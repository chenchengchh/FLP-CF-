# CPU 干跑: 验证 unet.pth state_dict 与 musetalkV15.json 构建的 UNet 键完全匹配 (不碰 GPU)
import json, torch
MT_MODELS = r"d:\FasterLivePortrait\causal_forcing_poc\musetalk_models"
from diffusers import UNet2DConditionModel
cfg = json.load(open(MT_MODELS + r"\musetalkV15.json"))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
m = UNet2DConditionModel(**cfg)
sd = torch.load(MT_MODELS + r"\musetalkV15\unet.pth", map_location="cpu", weights_only=True)
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing:", list(missing)[:5], len(missing))
print("unexpected:", list(unexpected)[:5], len(unexpected))
print("keys:", len(sd), "| model params:", sum(p.numel() for p in m.parameters()) // 10**6, "M")
print("OK" if not missing and not unexpected else "MISMATCH")
