# MuseTalk 逐帧 303ms 优化对比: 定位瓶颈 + 验证候选优化收益
#   baseline: enc(batch2) / unet(compiled) / vae dec / CPU 贴回
#   优化候选: compile VAE decoder+encoder / 解码裁剪(只解口型椭圆周边 latent 行) /
#             channels_last / INTER_CUBIC 替代 LANCZOS4 / 遮罩缓存
import os, sys, time
sys.path.insert(0, r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source")
import numpy as np
import torch, cv2
torch.set_grad_enabled(False)
from PIL import Image
from lipsync_musetalk import MuseTalkLipSync

SRC = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\i2v_std\15-26\portrait.png"
ls = MuseTalkLipSync(device="cuda:1")

img = Image.open(SRC).convert("RGB").resize((480, 832))
img_u8 = np.asarray(img).copy()
ls._frame_no = 0; ls._box_ema = None
box = ls._face_box(img_u8); box = ls._face_box(img_u8)   # 两轮收敛 EMA
H, W = img_u8.shape[:2]
x, y, x1, y1 = [int(v) for v in box]
cx, cy = (x + x1) // 2, (y + y1) // 2
s = int(max(x1 - x, y1 - y) // 2 * ls.EXPAND)
xs, ys, xe, ye = max(0, cx - s), max(0, cy - s), min(W, cx + s), min(H, cy + s)
crop = img_u8[ys:ye, xs:xe]
crop_r = cv2.resize(crop, (ls.CROP, ls.CROP), interpolation=cv2.INTER_LANCZOS4)
t256 = torch.from_numpy(crop_r.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(ls.device, ls.dtype) * 2.0 - 1.0
masked = t256 * (1.0 - ls.half_mask)[None, None]
# 假 whisper chunk (whisper-tiny hidden_states 5 层, chunk [1,50,384] 与线上一致)
feat = torch.zeros(1, 50, 5, 384, device=ls.device, dtype=ls.dtype)
chunk = feat[:, :10].reshape(1, 50, 384)
emb = ls.pe(chunk)

def sync(): torch.cuda.synchronize(ls.device)
def timeit(fn, n=30):
    fn(); sync(); t = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t) / n * 1000

# ---------- baseline (当前线上路径) ----------
dt_enc = timeit(lambda: torch.cat(ls._enc(torch.cat([masked, t256], dim=0)).chunk(2), dim=1))
lat = torch.cat(ls._enc(torch.cat([masked, t256], dim=0)).chunk(2), dim=1)
dt_unet = timeit(lambda: ls.unet(lat, ls.timesteps, encoder_hidden_states=emb).sample)
pred = ls.unet(lat, ls.timesteps, encoder_hidden_states=emb).sample
dt_dec = timeit(lambda: ls.vae.decode(pred / ls.scaling).sample)
def _dec_cpu(): return (ls.vae.decode(pred / ls.scaling).sample / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
dt_dec_cpu = timeit(_dec_cpu)
out256 = _dec_cpu()
out_u8 = (out256 * 255).astype(np.uint8)
dt_up_lz = timeit(lambda: cv2.resize(out_u8, (xe - xs, ye - ys), interpolation=cv2.INTER_LANCZOS4))
dt_up_cb = timeit(lambda: cv2.resize(out_u8, (xe - xs, ye - ys), interpolation=cv2.INTER_CUBIC))
mask = np.zeros(out_u8.shape[:2], np.float32)
fx, fy = x - xs, y - ys
fw, fh = x1 - x, y1 - y
ax, ay = int(fw * 0.52), int(fh * 0.28)
cv2.ellipse(mask, (fx + fw // 2, fy + int(fh * 0.72)), (ax, ay), 0, 0, 360, 1.0, -1)
dt_blur = timeit(lambda: cv2.GaussianBlur(mask, (0, 0), 9))
blend_dst = crop.astype(np.float32).copy()
mask_b = cv2.GaussianBlur(mask, (0, 0), 9)[..., None]
dt_blend = timeit(lambda: cv2.ellipse.__self__ if False else None) if False else 0

print("== baseline (ms/帧, 3050 单独) ==")
print(f"enc batch2      {dt_enc:7.1f}")
print(f"unet (compiled) {dt_unet:7.1f}")
print(f"vae dec GPU     {dt_dec:7.1f}")
print(f"vae dec + D2H   {dt_dec_cpu:7.1f}  (D2H {dt_dec_cpu-dt_dec:.1f})")
print(f"resize up LANCZ {dt_up_lz:7.1f} | CUBIC {dt_up_cb:7.1f}")
print(f"mask GaussianBt {dt_blur:7.1f}  (可缓存, 每帧可省)")
print(f"GPU小计 {dt_enc+dt_unet+dt_dec:.1f} | 全路径近似 {dt_enc+dt_unet+dt_dec_cpu+dt_up_lz+dt_blur:.1f}")

# ---------- 优化 1: 解码裁剪 (只解 latent 行 12:32, 椭圆上缘余量 ~23px) ----------
pred_c = pred[:, :, 12:, :]
dt_dec_crop = timeit(lambda: ls.vae.decode(pred_c / ls.scaling).sample)
print(f"\n== 优化1: 解码裁剪 rows 12:32 ({pred_c.shape[2]}/32) ==")
print(f"vae dec crop    {dt_dec_crop:7.1f}  (省 {dt_dec - dt_dec_crop:.1f})")

# ---------- 优化 2: channels_last ----------
vae_cl = ls.vae.to(memory_format=torch.channels_last)
p_cl = pred.to(memory_format=torch.channels_last)
dt_dec_cl = timeit(lambda: vae_cl.decode(p_cl / ls.scaling).sample)
unet_cl = torch.compile(ls.unet) if os.environ.get("LIPSYNC_PLAIN") == "1" else None
print(f"\n== 优化2: channels_last decoder ==")
print(f"vae dec CL      {dt_dec_cl:7.1f}  (baseline {dt_dec:.1f})")

# ---------- 优化 3: compile VAE decoder (裁剪输入, 静态 shape) ----------
print("\n== 优化3: compile VAE decoder (首次编译 ~1-2min) ==")
dec_fn = ls.vae.decoder
try:
    dec_c = torch.compile(dec_fn)
    t0 = time.time()
    out_c = dec_c(pred_c / ls.scaling)
    sync(); print(f"编译耗时 {time.time()-t0:.0f}s")
    dt_dec_cc = timeit(lambda: dec_c(pred_c / ls.scaling))
    print(f"dec crop+compile {dt_dec_cc:7.1f}  (baseline {dt_dec:.1f}, 裁剪 {dt_dec_crop:.1f})")
except Exception as e:
    print(f"decoder compile 失败: {type(e).__name__}: {e}")

# ---------- 优化 4: compile VAE encoder (batch2) ----------
print("\n== 优化4: compile VAE encoder ==")
try:
    enc_c = torch.compile(ls.vae.encoder)
    t0 = time.time()
    _ = torch.cat(enc_c(torch.cat([masked, t256], dim=0)).chunk(2), dim=1) * ls.scaling
    sync(); print(f"编译耗时 {time.time()-t0:.0f}s")
    dt_enc_c = timeit(lambda: torch.cat(enc_c(torch.cat([masked, t256], dim=0)).chunk(2), dim=1) * ls.scaling)
    print(f"enc batch2+compile {dt_enc_c:7.1f}  (baseline {dt_enc:.1f})")
except Exception as e:
    print(f"encoder compile 失败: {type(e).__name__}: {e}")

print("\n== 汇总: 预计优化后全路径 ==")
best_enc = dt_enc_c if 'dt_enc_c' in dir() else dt_enc
best_dec = dt_dec_cc if 'dt_dec_cc' in dir() else (dt_dec_crop if 'dt_dec_crop' in dir() else dt_dec)
total = best_enc + dt_unet + best_dec + (dt_dec_cpu - dt_dec) + dt_up_cb + 2  # D2H+blend≈2ms余量
print(f"enc {best_enc:.0f} + unet {dt_unet:.0f} + dec {best_dec:.0f} + CPU~{dt_up_cb+2:.0f} = {total:.0f} ms -> {1000/total:.2f} fps")
