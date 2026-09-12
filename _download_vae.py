# sd-vae-ft-mse 并行下载 (unet.pth 串行脚本阻塞时单独拉 VAE)
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

BASE = r"d:\FasterLivePortrait\causal_forcing_poc\musetalk_models\sd-vae-ft-mse"
os.makedirs(BASE, exist_ok=True)
for f in ["config.json", "diffusion_pytorch_model.safetensors"]:
    try:
        p = hf_hub_download("stabilityai/sd-vae-ft-mse", f, local_dir=BASE)
        print("OK vae", f, "->", p, flush=True)
    except Exception as e:
        print("FAIL vae", f, repr(e), flush=True)
        if f.endswith(".safetensors"):
            p = hf_hub_download("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", local_dir=BASE)
            print("OK vae bin fallback ->", p, flush=True)
print("ALL DONE", flush=True)
