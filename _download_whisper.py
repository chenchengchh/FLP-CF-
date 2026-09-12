# whisper-tiny (transformers 格式) 下载, 供 MuseTalk V15 音频特征提取
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

BASE = r"d:\FasterLivePortrait\causal_forcing_poc\musetalk_models\whisper-tiny"
os.makedirs(BASE, exist_ok=True)
for f in ["config.json", "preprocessor_config.json", "model.safetensors"]:
    p = hf_hub_download("openai/whisper-tiny", f, local_dir=BASE)
    print("OK", f, "->", p, flush=True)
print("ALL DONE", flush=True)
