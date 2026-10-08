"""Loader, tokenizer fix, text chunking and generation for rumik-ai/rumik-oss-1.

Importable outside ComfyUI: every ``comfy`` / ``folder_paths`` import is guarded.
"""
from __future__ import annotations

import gc
import logging
import os
import re
from typing import Any

import torch

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None

try:
    import comfy.model_management as mm  # type: ignore
except Exception:
    mm = None

logger = logging.getLogger("ComfyUI-Rumik-TTS")

DEFAULT_REPO = "rumik-ai/rumik-oss-1"
PINNED_REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"
SAMPLE_RATE = 24000
SPEAKERS = ["Ira", "Aisha", "Siya", "Zoya"]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DOWNLOAD_PATTERNS = ["*.json", "*.py", "*.safetensors", "codec/*"]


def models_dir() -> str | None:
    if folder_paths is None:
        return None
    return os.path.join(folder_paths.models_dir, "tts", "rumik-oss-1")


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return mm.get_torch_device() if mm is not None else torch.device("cuda")
    if mm is not None:
        return mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(repo_or_path: str) -> str:
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"allow_patterns": DOWNLOAD_PATTERNS}
    if repo_or_path == DEFAULT_REPO:
        kwargs["revision"] = PINNED_REVISION
    target = models_dir() if repo_or_path == DEFAULT_REPO else None
    if target:
        kwargs["local_dir"] = target
    return snapshot_download(repo_or_path, **kwargs)


def load_tokenizer(path: str):
    """Load tokenizer.json exactly as saved.

    On transformers 5.x ``tokenizer_class: CohereTokenizer`` rebuilds the
    pre-tokenizer and yields different ids for most Indic text, and appends
    <SEP>/<CLS> ids beyond the embedding table. ``PreTrainedTokenizerFast``
    uses the saved tokenizer.json untouched. Files on disk are not modified.
    """
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(path, "tokenizer.json"),
        bos_token="<BOS_TOKEN>",
        eos_token="<|END_OF_TURN_TOKEN|>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        clean_up_tokenization_spaces=False,
    )
    bos = tok.bos_token_id
    ids = tok("a")["input_ids"]
    tok._rumik_prepend_bos = bos is not None and (not ids or ids[0] != bos)
    return tok


def encode_prompt(tok, prompt: str) -> list[int]:
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    if getattr(tok, "_rumik_prepend_bos", False):
        ids = [tok.bos_token_id] + ids
    return ids


def build_prompt(speaker: str, description: str, text: str) -> str:
    description = (description or "").strip()
    desc = f'<description="{description}"> ' if description else ""
    return f"<text>{speaker}: {desc}{text}<audio>"


def load_bundle(repo_or_path: str, dtype: str, device: str) -> dict:
    from transformers import AutoModelForCausalLM, MimiModel

    path = resolve_path(repo_or_path)
    dev = resolve_device(device)
    tok = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=DTYPES[dtype]
    ).eval()
    mimi = MimiModel.from_pretrained(path, subfolder="codec", dtype=torch.float32).eval()
    model.to(dev)
    mimi.to(dev)
    return {"model": model, "mimi": mimi, "tokenizer": tok, "device": dev, "path": path}


def offload(bundle: dict) -> None:
    bundle["model"].to("cpu")
    bundle["mimi"].to("cpu")
    gc.collect()
    if mm is not None:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_on_device(bundle: dict) -> None:
    dev = bundle["device"]
    if next(bundle["model"].parameters()).device != dev:
        bundle["model"].to(dev)
        bundle["mimi"].to(dev)


_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:،])\s+")
CHARS_PER_SECOND = 14.0
PACE = {"slow": 0.8, "fast": 1.2}


def estimate_seconds(text: str, description: str = "") -> float:
    plain = re.sub(r"<[a-z]+>", "", text)
    rate = CHARS_PER_SECOND
    for key, mult in PACE.items():
        if key in (description or "").lower():
            rate *= mult
    return len(plain.strip()) / rate


def chunk_text(text: str, description: str = "", max_seconds: float = 25.0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if estimate_seconds(sent, description) <= max_seconds:
            pieces.append(sent)
            continue
        for clause in _CLAUSE_SPLIT.split(sent):
            if estimate_seconds(clause, description) <= max_seconds:
                pieces.append(clause)
                continue
            words, cur = clause.split(), []
            for w in words:
                if cur and estimate_seconds(" ".join(cur + [w]), description) > max_seconds:
                    pieces.append(" ".join(cur))
                    cur = []
                cur.append(w)
            if cur:
                pieces.append(" ".join(cur))
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        cand = f"{cur} {p}".strip()
        if cur and estimate_seconds(cand, description) > max_seconds:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def generate_chunk(bundle: dict, speaker: str, description: str, text: str,
                   temperature: float, top_k: int, max_new_tokens: int) -> torch.Tensor:
    model, mimi, tok, dev = bundle["model"], bundle["mimi"], bundle["tokenizer"], bundle["device"]
    ids = torch.tensor([encode_prompt(tok, build_prompt(speaker, description, text))], device=dev)
    mask = torch.ones_like(ids)
    out = model.generate_audio(
        input_ids=ids, attention_mask=mask, max_new_tokens=int(max_new_tokens),
        temperature=float(temperature), top_k=int(top_k), do_sample=temperature > 0,
    )
    tail = out[0].tolist()[ids.shape[1]:]
    codes = model.audio_tokens_to_codes(tail).to(mimi.device)
    with torch.inference_mode():
        wav = mimi.decode(codes).audio_values[0, 0]
    return wav.float().cpu()


def synthesize(bundle: dict, speaker: str, description: str, text: str, seed: int,
               temperature: float, top_k: int, max_new_tokens: int,
               chunk_long_text: bool, unload_after: bool, gap_seconds: float = 0.25) -> dict:
    ensure_on_device(bundle)
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available() and bundle["device"].type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        chunks = chunk_text(text, description) if chunk_long_text else [text.strip()]
        if not chunks:
            raise ValueError("text is empty")
        gap = torch.zeros(int(gap_seconds * SAMPLE_RATE))
        parts: list[torch.Tensor] = []
        for i, c in enumerate(chunks):
            if i:
                parts.append(gap)
            parts.append(generate_chunk(bundle, speaker, description, c,
                                        temperature, top_k, max_new_tokens))
        wav = torch.cat(parts)
    finally:
        if unload_after:
            offload(bundle)
    return {"waveform": wav.reshape(1, 1, -1).contiguous(), "sample_rate": SAMPLE_RATE}
