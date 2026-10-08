import os, sys, time, importlib
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(root))
import torch, soundfile as sf
m = importlib.import_module(os.path.basename(root))
n = sys.modules[os.path.basename(root) + ".nodes"]
dtype = sys.argv[1] if len(sys.argv) > 1 else "bf16"
t0 = time.time()
(b,) = n.RumikTTSLoader().load(os.path.join(root, "weights/rumik-oss-1"), dtype, "cpu")
print("load s", round(time.time() - t0, 1), flush=True)
os.makedirs(os.path.join(root, "test_out"), exist_ok=True)
cases = [("hindi", "Ira", "happy, Hindi accent, steady pace", "नमस्ते, आज आपका दिन कैसा रहा?"),
         ("codeswitch", "Aisha", "excited, Indian English, fast", "अरे yaar, <chuckle> meeting कल morning में है!")]
for name, spk, desc, text in cases:
    t0 = time.time()
    (a,) = n.RumikTTSGenerate().generate(b, spk, desc, text, 42, 0.8, 30, 400, True, False)
    dt = time.time() - t0
    secs = a["waveform"].shape[-1] / a["sample_rate"]
    p = os.path.join(root, f"test_out/{name}_{dtype}.wav")
    sf.write(p, a["waveform"][0, 0].numpy(), a["sample_rate"])
    print(name, "wall s", round(dt, 1), "audio s", round(secs, 2), "tok/s", round(secs * 100 / dt, 2), p, flush=True)
