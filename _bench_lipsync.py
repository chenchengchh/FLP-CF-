# MuseTalk 3050 benchmark + 烟测: 真实 TTS 音频 → 真实特征 → 逐帧口型 → mp4 目检
import os, sys, time, subprocess
sys.path.insert(0, r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source")
import numpy as np
import torch, cv2
from PIL import Image
from lipsync_musetalk import MuseTalkLipSync, mp3_to_pcm16k

SRC = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\i2v_std\15-26\portrait.png"
PY = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp_env\python.exe"
OUT = r"d:\FasterLivePortrait\causal_forcing_poc\cfpp\source\lipsync_bench"
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
ls = MuseTalkLipSync(device="cuda:1")
print(f"[bench] model load {time.time()-t0:.1f}s", flush=True)

# 1) 真实 TTS 音频
text = "大家好，欢迎来到直播间，今天给大家介绍一款非常好用的产品，感兴趣的朋友可以点个关注。"
mp3 = os.path.join(OUT, "test_tts.mp3")
if not os.path.exists(mp3):
    subprocess.run([PY, "-m", "edge_tts", "--voice", "zh-CN-XiaoxiaoNeural",
                    "--text", text, "--write-media", mp3], check=True)
pcm = mp3_to_pcm16k(mp3)
dur = len(pcm) / 16000
print(f"[bench] audio {dur:.1f}s pcm {len(pcm)}", flush=True)

# 2) 音频特征 (计时)
torch.cuda.synchronize(ls.device)
t0 = time.time()
feat = ls.extract_feature(pcm)
torch.cuda.synchronize(ls.device)
print(f"[bench] extract_feature {time.time()-t0:.2f}s shape={tuple(feat.shape)}", flush=True)

# 3) 模拟帧: 源图 + 轻微平移缩放扰动 (模拟 CF++ 帧间运动)
img = Image.open(SRC).convert("RGB").resize((480, 832))
base = np.asarray(img).astype(np.float32) / 255.0
base_t = torch.from_numpy(base).permute(2, 0, 1)          # [3,832,480] cpu float

def frame_with(i):
    M = cv2.getRotationMatrix2D((240, 380), 0.4 * np.sin(i * 0.2), 1.0 + 0.004 * np.sin(i * 0.13))
    M[0, 2] += 1.2 * np.sin(i * 0.17); M[1, 2] += 0.8 * np.cos(i * 0.11)
    return torch.from_numpy(cv2.warpAffine(base, M, (480, 832))).permute(2, 0, 1)

# 4) 暖机 3 帧
for i in range(3):
    ls.process_frame(frame_with(i), feat, 0.5)
torch.cuda.synchronize(ls.device)

# 5) 计时 60 帧
N = 60
torch.cuda.reset_peak_memory_stats(ls.device)
t0 = time.time()
outs = []
for i in range(N):
    off = float(min(dur * 0.95, i * dur / N))
    outs.append(ls.process_frame(frame_with(i), feat, off))
torch.cuda.synchronize(ls.device)
dt = (time.time() - t0) / N
peak_alloc = torch.cuda.max_memory_allocated(ls.device) / 1024**2
free_b, total_b = torch.cuda.mem_get_info(ls.device)
print(f"[bench] per-frame {dt*1000:.0f}ms | alloc peak {peak_alloc:.0f}MB | "
      f"process used {total_b/1024**3 - free_b/1024**3:.2f}GB / {total_b/1024**3:.1f}GB", flush=True)

# 6) 出对比帧 + mp4 目检 (口型动起来 + 与音频粗对齐)
cv2.imwrite(os.path.join(OUT, "before.jpg"),
            cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
png_pat = os.path.join(OUT, "f%04d.jpg")
for i, o in enumerate(outs):
    a = np.clip(o.permute(1, 2, 0).numpy(), 0, 1)
    cv2.imwrite(png_pat % i, cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15",
                "-i", png_pat, "-i", mp3,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest",
                os.path.join(OUT, "lipsync_test.mp4")], check=True)
for i in range(N):
    os.remove(png_pat % i)
print(f"[bench] DONE -> {OUT}\\lipsync_test.mp4", flush=True)
