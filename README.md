# ComfyUI-Rumik-TTS

ComfyUI nodes for [`rumik-ai/rumik-oss-1`](https://huggingface.co/rumik-ai/rumik-oss-1), a text-to-speech model for Indian
languages. Pinned to revision `0ed3c98684e14350c910129c0efa179841c41ad2`. Output is 24 kHz mono, returned as a standard
ComfyUI `AUDIO`.

## Nodes (category `Rumik TTS`)

**Rumik TTS Loader** -> `RUMIK_MODEL` (LM + Mimi codec + tokenizer)
- `repo_or_path`: HF repo id (default `rumik-ai/rumik-oss-1`) or a local directory. A repo id is downloaded (about 6.7 GB) to
  `ComfyUI/models/tts/rumik-oss-1`, or to the HF cache when run outside ComfyUI.
- `dtype`: bf16 / fp16 / fp32. `device`: auto / cuda / cpu.

**Rumik TTS Generate** -> `AUDIO`
- `speaker`: Ira, Aisha, Siya, Zoya.
- `description`: optional, e.g. `happy, Hindi accent, steady pace`. Empty omits the `<description>` prefix.
  Tone: happy/sad/angry/excited/professional. Accent: Hindi/Telugu/Tamil/Kannada/Bengali/Punjabi/Indian English.
  Pace: slow/fast/steady.
- `text`: multiline. Inline tags `<laugh>`, `<chuckle>`, `<sigh>` are supported.
- `seed`, `temperature` (0.8), `top_k` (30), `max_new_tokens` (3072; about 100 tokens per second of audio, per chunk).
- `chunk_long_text`: the model is unreliable past roughly 30-35 s per utterance. When on, text is split on sentence
  punctuation (`. ? ! ।`, then commas, then words) into chunks estimated at <= 25 s, each generated with the same
  speaker and description, and joined with 0.25 s of silence. The estimate is a character-rate heuristic (about 14
  chars/s, scaled for slow/fast).
- `unload_after` (default on): moves the LM and codec to CPU and empties the cache after generating, so the VRAM is free
  for other models (H3, Higgs). The next run moves them back.

## Prompt format

```
<text>{SPEAKER}: <description="{DESCRIPTION}"> {TEXT}<audio>
```
The tokenizer adds BOS itself. Example:
`<text>Ira: <description="happy, Hindi accent, steady pace"> नमस्ते, आज आपका दिन कैसा रहा?<audio>`

## Tokenizer fix (transformers 5.x)

The repo's `tokenizer_config.json` sets `tokenizer_class: CohereTokenizer`. On transformers 5.x that class replaces the
saved pre-tokenizer, so most Indic text gets different ids than on 4.57 (and `<SEP>`/`<CLS>` can appear beyond the
277404-row embedding). See https://github.com/Rishxb-arch/rumik-tokenizer-fix. The loader instead builds a
`PreTrainedTokenizerFast` straight from the saved `tokenizer.json` (no downloaded file is edited), and prepends BOS
itself only if the tokenizer does not. Measured, token counts for the wire-format prompt:

| prompt | 4.57 stock | 5.12 stock | 5.12 fixed (this pack) |
|---|---|---|---|
| Hindi | 27 | 37 | 27 |
| Tamil | 27 | 52 | 27 |
| Telugu | 32 | 46 | 32 |
| Bengali | 29 | 41 | 29 |
| Hindi+English | 34 | 39 | 34 |
| English | 31 | 31 | 31 |

Fixed ids on 5.12.1 are identical to 4.57.6 for every prompt. `tests/tok_parity.py` reproduces this.

## VRAM

The LM is about 6 GB in bf16/fp16 (fp32 doubles it); Mimi is small (kept in fp32). KV cache is small relative to that.
Use `unload_after` to coexist with large video models. CPU works but is slow (see `tests/e2e.py`).

## Install

Put this folder in `ComfyUI/custom_nodes/`. Needs `transformers>=4.57` (tested on 5.12.1), `huggingface_hub`,
`accelerate` — nothing else beyond ComfyUI's own torch.

## License

- Model weights: **CC-BY-NC 4.0**. Non-commercial only, including self-hosted commercial deployment.
- Mimi codec: CC-BY-4.0.
- This node pack's code ships without a license grant on the weights; you are responsible for complying with theirs.
