# 口型贴回"口部上方色差"诊断:
#   ① 椭圆坐标系 bug 假设: 线上用裁剪区内原图尺度参数直接画进 256 空间,
#      裁剪区 !=256px 时椭圆偏大偏上 -> 侵入鼻翼/脸颊 (用户报"口部上面有色差"的位置)
#   ② VAE 重建固有肤色偏: 纯往返 dec(enc(x)) vs x 的逐通道/分行差 (不走 UNet, 排除嘴形结构差)
import os, sys
sys.path.insert(0, r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source")
import numpy as np, torch, cv2
torch.set_grad_enabled(False)
from PIL import Image
from lipsync_musetalk import MuseTalkLipSync

SRC = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\i2v_std\15-26\portrait.png"
OUT = r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source\lipsync_bench"
os.makedirs(OUT, exist_ok=True)
ls = MuseTalkLipSync(device="cuda:1")

img_u8 = np.asarray(Image.open(SRC).convert("RGB").resize((480, 832))).copy()
ls._frame_no = 0; ls._box_ema = None
box = ls._face_box(img_u8); box = ls._face_box(img_u8)   # EMA 收敛
H, W = img_u8.shape[:2]
x, y, x1, y1 = [int(v) for v in box]
cx, cy = (x + x1) // 2, (y + y1) // 2
s = int(max(x1 - x, y1 - y) // 2 * ls.EXPAND)
xs, ys, xe, ye = max(0, cx - s), max(0, cy - s), min(W, cx + s), min(H, cy + s)
crop = img_u8[ys:ye, xs:xe]
ch_, cw_ = crop.shape[:2]
scx, scy = ls.CROP / cw_, ls.CROP / ch_
print(f"[diag] face box=({x},{y},{x1},{y1}) {x1-x}x{y1-y} | crop {cw_}x{ch_} | scale x{scx:.3f} y{scy:.3f}")

# --- 线上同款 256 化 ---
crop_r = cv2.resize(crop, (ls.CROP, ls.CROP), interpolation=cv2.INTER_LANCZOS4)
t256 = torch.from_numpy(crop_r.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(ls.device, ls.dtype) * 2 - 1

# ② 纯 VAE 往返色偏 (不走 UNet: 无嘴形结构差, 即贴回椭圆边界的色偏下限)
lat_ref = ls._enc(t256)
dec_ref = ls.vae.decode(lat_ref / ls.scaling).sample
ref256 = (dec_ref / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
diff_ref = crop_r.astype(np.float32) - ref256 * 255
print(f"[diag] VAE往返 逐通道均值差(src-dec) R{diff_ref[...,0].mean():+.1f} G{diff_ref[...,1].mean():+.1f} B{diff_ref[...,2].mean():+.1f} /255")
rows = diff_ref.mean(axis=(1, 2))
print("[diag] VAE往返 分行均值差(每24行, 0=脸顶):", " ".join(f"{rows[i]:+.0f}" for i in range(0, 256, 24)))

# --- 线上同款 UNet+裁剪解码 (带内) ---
masked = t256 * (1 - ls.half_mask)[None, None]
lat = torch.cat(ls._enc(torch.cat([masked, t256], 0)).chunk(2), 1)
feat = torch.zeros(1, 50, 5, 384, device=ls.device, dtype=ls.dtype)
emb = ls.pe(feat[:, :10].reshape(1, 50, 384))
pred = ls.unet(lat, ls.timesteps, encoder_hidden_states=emb).sample
dec = ls.vae.decode(pred[:, :, ls.DEC_R0:, :] / ls.scaling).sample
img_dec = (dec / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()  # [192,256,3]
band = ls.CROP * ls.DEC_R0 // 32          # 64
src_band = crop_r[band:]
dec_band = (img_dec * 255).astype(np.uint8)
d = src_band.astype(np.float32) - dec_band.astype(np.float32)

# ① 椭圆坐标系: 线上(原图尺度直画) vs 修正(256 空间换算)
fx, fy = x - xs, y - ys
fw, fh = x1 - x, y1 - y
ax0, ay0 = int(fw * 0.52), int(fh * 0.28)
c0 = (fx + fw // 2, fy + int(fh * 0.72))
ax1, ay1 = int(fw * scx * 0.52), int(fh * scy * 0.28)
c1 = (int(fx * scx + fw * scx / 2), int(fy * scy + fh * scy * 0.72))
top0, top1 = c0[1] - ay0, c1[1] - ay1
print(f"[diag] 椭圆线上: c={c0} a=({ax0},{ay0}) 上缘y={top0} | 修正: c={c1} a=({ax1},{ay1}) 上缘y={top1} | 上缘偏差 {top1-top0}px (正=线上侵入上方)")

# 椭圆环带 (嘴外脸内, 无嘴形结构差) 的色差 = 拼接可见色差的主项
m0 = np.zeros((ls.CROP, ls.CROP), np.float32)
cv2.ellipse(m0, c0, (ax0, ay0), 0, 0, 360, 1.0, -1)
ring = (m0[band:] > 0.4) & (m0[band:] < 0.95)
if ring.any():
    print(f"[diag] 线上椭圆环带色差 R{d[ring][:,0].mean():+.1f} G{d[ring][:,1].mean():+.1f} B{d[ring][:,2].mean():+.1f} /255 (|d|均值 {np.abs(d[ring]).mean():.1f})")

# --- 可视化: 行0=原图带+两椭圆叠加 / 行1=解码带 / 行2=差值热图 ---
vis0 = src_band.copy()
cv2.ellipse(vis0, (c0[0], c0[1] - band), (ax0, ay0), 0, 0, 360, (255, 60, 60), 2)   # 红=线上
cv2.ellipse(vis0, (c1[0], c1[1] - band), (ax1, ay1), 0, 0, 360, (60, 255, 60), 2)   # 绿=修正
heat = np.clip(np.abs(d) * 6, 0, 255).astype(np.uint8)
panel = np.concatenate([vis0, dec_band, cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]], axis=0)
cv2.imwrite(os.path.join(OUT, "diag_color.png"), panel[:, :, ::-1])
print(f"[diag] 可视化 -> {os.path.join(OUT, 'diag_color.png')} (上=原图+椭圆叠加 红=线上/绿=修正, 中=VAE解码带, 下=|差|x6 热图)")

# --- 修复后验证: 完整 process_frame 一帧, 椭圆应落在嘴周 (鼻区保持原样不重建) ---
ls._mask_cache.clear()
frame = torch.from_numpy(img_u8.astype(np.float32) / 255.).permute(2, 0, 1)
out = ls.process_frame(frame, feat, 0.5).permute(1, 2, 0).numpy()
out_u8 = (out * 255).astype(np.uint8)
face_vis = out_u8[max(0, y - 30):y1 + 40, max(0, x - 40):x1 + 40].copy()
face_vis = cv2.resize(face_vis, (face_vis.shape[1] * 2, face_vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
src_vis = img_u8[max(0, y - 30):y1 + 40, max(0, x - 40):x1 + 40].copy()
src_vis = cv2.resize(src_vis, (src_vis.shape[1] * 2, src_vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(OUT, "diag_fixed.png"), np.concatenate([src_vis, face_vis], axis=1)[:, :, ::-1])
print(f"[diag] 修复后贴回 -> {os.path.join(OUT, 'diag_fixed.png')} (左=原图, 右=修复后)")

# --- 横带定位: 修复后帧 vs 原图, 脸部列范围内分行差 (找横带确切行) ---
dfull = out_u8.astype(np.float32) - img_u8.astype(np.float32)
rows = np.abs(dfull[:, x:x1]).mean(axis=(1, 2))
print("[diag] 贴回帧 vs 原图 分行|差| (y: 值 直方):")
for i in range(140, 260, 4):
    print(f"  y={i:3d} {rows[i]:6.1f} {'#' * int(rows[i] / 1.5)}")
