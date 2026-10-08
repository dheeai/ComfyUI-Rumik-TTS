import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
import transformers
P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights/rumik-oss-1")
PROMPTS = {
 "hindi": 'नमस्ते, आज आपका दिन कैसा रहा?',
 "tamil": 'வணக்கம், இன்று உங்கள் நாள் எப்படி இருந்தது?',
 "telugu": 'నమస్కారం, ఈ రోజు మీ రోజు ఎలా గడిచింది?',
 "bengali": 'নমস্কার, আজ আপনার দিনটা কেমন কাটল?',
 "codeswitch": 'अरे yaar, meeting कल morning में है, <chuckle> don\'t forget करना।',
 "english": 'Hello, how was your day today? <sigh> It was long.',
}
def prompt(t): return f'<text>Ira: <description="happy, Hindi accent, steady pace"> {t}<audio>'
mode = sys.argv[1]
out = {}
if mode == "fixed":
    from rumik_core import load_tokenizer, encode_prompt
    tok = load_tokenizer(P)
    for k, t in PROMPTS.items(): out[k] = encode_prompt(tok, prompt(t))
else:
    tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
    for k, t in PROMPTS.items(): out[k] = tok(prompt(t))["input_ids"]
json.dump(out, open(f"/tmp/ids_{mode}_{transformers.__version__}.json", "w"))
print(mode, transformers.__version__, {k: len(v) for k, v in out.items()}, "bos", {k: v[0] for k, v in out.items()}, "max", max(max(v) for v in out.values()))
