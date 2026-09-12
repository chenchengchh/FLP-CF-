# MuseTalk 权重下载 (走 hf-mirror 镜像, 国内可达); 后台运行, 完成后打印 ALL DONE
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

BASE = r"d:\FasterLivePortrait\causal_forcing_poc\musetalk_models"
os.makedirs(BASE, exist_ok=True)

# 1) MuseTalk V15 UNet (fp32 ~3.2GB, 落盘后另转 fp16) + 配置
p = hf_hub_download("TMElyralab/MuseTalk", "musetalkV15/unet.pth", local_dir=BASE)
print("OK unet.pth ->", p, flush=True)
p = hf_hub_download("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", local_dir=BASE)
print("OK musetalk.json ->", p, flush=True)

# 2) sd-vae-ft-mse (MuseTalk 的 latent 编解码 VAE, ~335MB)
for f in ["config.json", "diffusion_pytorch_model.safetensors"]:
    try:
        p = hf_hub_download("stabilityai/sd-vae-ft-mse", f, local_dir=os.path.join(BASE, "sd-vae-ft-mse"))
        print("OK vae", f, "->", p, flush=True)
    except Exception as e:
        print("FAIL vae", f, repr(e), flush=True)
        if f.endswith(".safetensors"):
            p = hf_hub_download("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", local_dir=os.path.join(BASE, "sd-vae-ft-mse"))
            print("OK vae bin fallback ->", p, flush=True)

# 3) whisper tiny.pt (音频特征提取器, ~72MB, openai 官方 CDN)
import urllib.request
dst = os.path.join(BASE, "whisper", "tiny.pt")
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.exists(dst):
    urllib.request.urlretrieve(
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.pt", dst)
print("OK whisper tiny.pt ->", dst, flush=True)
print("ALL DONE", flush=True)
