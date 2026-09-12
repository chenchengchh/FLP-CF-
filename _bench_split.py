# 分段计时: 定位 431ms/帧 瓶颈分布 (enc x2 / unet / dec / CPU 贴回)
import os, sys, time
sys.path.insert(0, r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source")
import numpy as np
import torch, cv2
torch.set_grad_enabled(False)
from PIL import Image
from lipsync_musetalk import MuseTalkLipSync, mp3_to_pcm16k

SRC = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\i2v_std\15-26\portrait.png"
OUT = r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source\lipsync_bench"
mp3 = os.path.join(OUT, "test_tts.mp3")

ls = MuseTalkLipSync(device="cuda:1")
pcm = mp3_to_pcm16k(mp3)
feat = ls.extract_feature(pcm)

img = Image.open(SRC).convert("RGB").resize((480, 832))
base = np.asarray(img).astype(np.float32) / 255.0
f = torch.from_numpy(base).permute(2, 0, 1)   # [3,832,480] cpu

# 准备一段完整 process_frame 的输入 (模拟真实路径)
img_u8 = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
box = ls._face_box(img_u8); ls._box_ema = None; ls._frame_no = 0
box = ls._face_box(img_u8)
H, W = img_u8.shape[:2]
x, y, x1, y1 = [int(v) for v in box]
cx, cy = (x + x1) // 2, (y + y1) // 2
s = int(max(x1 - x, y1 - y) // 2 * ls.EXPAND)
xs, ys, xe, ye = max(0, cx - s), max(0, cy - s), min(W, cx + s), min(H, cy + s)
crop = img_u8[ys:ye, xs:xe]
crop_r = cv2.resize(crop, (ls.CROP, ls.CROP), interpolation=cv2.INTER_LANCZOS4)

t256 = torch.from_numpy(crop_r.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(ls.device, ls.dtype) * 2.0 - 1.0
masked = t256 * (1.0 - ls.half_mask)[None, None]
chunk = ls.chunk_at(feat, 1.0)
emb = ls.pe(chunk)

def sync(): torch.cuda.synchronize(ls.device)
def timeit(fn, n=20):
    sync(); t = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t) / n

N = 20
dt_enc1 = timeit(lambda: ls._enc(masked))
lat_m = ls._enc(masked)
dt_enc2 = timeit(lambda: ls._enc(t256))
lat_r = ls._enc(t256)
lat = torch.cat([lat_m, lat_r], dim=1)
dt_unet = timeit(lambda: ls.unet(lat, ls.timesteps, encoder_hidden_states=emb).sample)
pred = ls.unet(lat, ls.timesteps, encoder_hidden_states=emb).sample
dt_dec = timeit(lambda: ls.vae.decode(pred / ls.scaling).sample)
dt_dec_cpu = timeit(lambda: ls._dec(pred))
out256 = ls._dec(pred)
out_full_cache = (out256 * 255).astype(np.uint8)
dt_rs_up = timeit(lambda: cv2.resize(out_full_cache, (xe - xs, ye - ys), interpolation=cv2.INTER_LANCZOS4))
dt_rs_dn = timeit(lambda: cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4))
dt_h2d = timeit(lambda: torch.from_numpy(crop_r.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(ls.device, ls.dtype) * 2.0 - 1.0)
def _haar():
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return ls.cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
dt_haar = timeit(_haar, 5)

print(f"enc(masked)  {dt_enc1*1000:7.1f} ms")
print(f"enc(ref)     {dt_enc2*1000:7.1f} ms")
print(f"unet step    {dt_unet*1000:7.1f} ms")
print(f"vae dec(GPU) {dt_dec*1000:7.1f} ms")
print(f"dec + D2H    {dt_dec_cpu*1000:7.1f} ms  (D2H 开销 {(dt_dec_cpu-dt_dec)*1000:.1f})")
print(f"resize up    {dt_rs_up*1000:7.1f} ms")
print(f"resize down  {dt_rs_dn*1000:7.1f} ms")
print(f"H2D crop     {dt_h2d*1000:7.1f} ms")
print(f"haar detect  {dt_haar*1000:7.1f} ms")
print(f"GPU 小计(enc2+unet+dec) {(dt_enc1+dt_enc2+dt_unet+dt_dec)*1000:.1f} ms")
