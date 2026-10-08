from __future__ import annotations

from . import rumik_core as core


class RumikTTSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_or_path": ("STRING", {"default": core.DEFAULT_REPO}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("RUMIK_MODEL",)
    RETURN_NAMES = ("rumik_model",)
    FUNCTION = "load"
    CATEGORY = "Rumik TTS"

    def load(self, repo_or_path, dtype, device):
        return (core.load_bundle(repo_or_path.strip(), dtype, device),)


class RumikTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rumik_model": ("RUMIK_MODEL",),
                "speaker": (core.SPEAKERS, {"default": "Ira"}),
                "description": ("STRING", {"default": "", "multiline": False}),
                "text": ("STRING", {"default": "नमस्ते, आज आपका दिन कैसा रहा?", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000}),
                "max_new_tokens": ("INT", {"default": 3072, "min": 16, "max": 8192}),
                "chunk_long_text": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Rumik TTS"

    def generate(self, rumik_model, speaker, description, text, seed, temperature,
                 top_k, max_new_tokens, chunk_long_text, unload_after):
        audio = core.synthesize(rumik_model, speaker, description, text, seed, temperature,
                                top_k, max_new_tokens, chunk_long_text, unload_after)
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "RumikTTSLoader": RumikTTSLoader,
    "RumikTTSGenerate": RumikTTSGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RumikTTSLoader": "Rumik TTS Loader",
    "RumikTTSGenerate": "Rumik TTS Generate",
}
