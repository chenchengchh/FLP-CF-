import argparse, json, os, sys, time, queue, threading, subprocess, types, importlib.util
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--segments", type=int, default=4)
parser.add_argument("--blocks", type=int, default=20, help="每段 latent 块数; 越短 KV 缓存越浅, 去噪越快")
parser.add_argument("--mult", type=int, default=4)
parser.add_argument("--target_fps", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--taehv", action="store_true", help="用 TAEHV 轻量解码器替换 Wan VAE (decode ~0.44s -> ~0.05s/块)")
parser.add_argument("--out", type=str, default=r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\output\stream\stream_live.mp4")
args = parser.parse_args()
torch.manual_seed(args.seed)

SEGMENTS, MULT, TARGET_FPS = args.segments, args.mult, args.target_fps
BLOCKS_PER_SEG = args.blocks
LAT_H, LAT_W = 60, 104
H, W = 480, 832
CAPTION = ("A young woman holds a heart-shaped gesture with both hands in front of her chest, "
           "then moves her hands gently and naturally, upper body view, simple clean gray background, calm expression.")
IMG_PATH = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\i2v_input\26-15\000001.png"
CKPT = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\cfpp_git\causal-forcing++\framewise-2step.pt"
RIFE_ARCH = r"F:\2026-ComfyUI-V8.3\custom_nodes\comfyui-frame-interpolation\vfi_models\rife\rife_arch.py"
RIFE_CKPT = r"F:\2026-ComfyUI-V8.3\models\frame_interpolation\rife47.pth"
# PATH 里的 ffmpeg 是 TRAE 裁剪版(无 rawvideo demuxer/pipe 协议)，必须用完整版
FFMPEG_CANDIDATES = [
    r"C:\Users\kt\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
    "ffmpeg",
]
FFMPEG = next((p for p in FFMPEG_CANDIDATES if os.path.exists(p) or p == "ffmpeg"), "ffmpeg")

torch.set_grad_enabled(False)
gpu = torch.device("cuda")
t_script = time.time()

# ---------------- RIFE (standalone) ----------------
_stub = types.ModuleType("comfy.model_management")
_stub.get_torch_device = lambda: torch.device("cuda")
sys.modules["comfy"] = types.ModuleType("comfy")
sys.modules["comfy.model_management"] = _stub
_spec = importlib.util.spec_from_file_location("rife_arch", RIFE_ARCH)
_ra = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ra)
rife_model = _ra.IFNet(arch_ver="4.7")
rife_model.load_state_dict(torch.load(RIFE_CKPT, map_location="cpu"))
rife_model.eval().cuda()
_ = rife_model(torch.rand(1,3,H,W).cuda(), torch.rand(1,3,H,W).cuda(), 0.5, [8,4,2,1], training=False, fastmode=True, ensemble=False)
print(f"[setup] RIFE loaded ({time.time()-t_script:.1f}s)")

# ---------------- CF++ pipeline ----------------
from pipeline import CausalInferencePipeline
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

config = OmegaConf.load("configs/causal_forcing_dmd_framewise_2step_1p3b_384.yaml")
config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), config)
pipeline = CausalInferencePipeline(config, device=gpu)
sd = torch.load(CKPT, map_location="cpu")
try:
    pipeline.generator.load_state_dict(sd["generator_ema"])
except RuntimeError:
    fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in sd["generator_ema"].items()}
    pipeline.generator.load_state_dict(fixed, strict=False)
pipeline = pipeline.to(dtype=torch.bfloat16)
low_memory = get_cuda_free_memory_gb(gpu) < 40
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)
print(f"[setup] pipeline loaded ({time.time()-t_script:.1f}s)")

# 流式 VAE 解码器: TAEHV(轻量, 快) 或 Wan VAE 增量解码(保真)
lat_cache = None  # TAEHV 的 3 帧 latent 上下文缓存
if args.taehv:
    from demo_utils.taehv import TAEHV
    decoder = TAEHV(checkpoint_path="checkpoints/taew2_1.pth").eval().to(dtype=torch.float16).to(gpu)
    vae_cache = []
    print(f"[setup] TAEHV decoder loaded ({time.time()-t_script:.1f}s)")
else:
    decoder = VAEDecoderWrapper()
    vae_sd = torch.load("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu")
    decoder.load_state_dict({k: v for k, v in vae_sd.items() if "decoder." in k or "conv2" in k})
    decoder.eval().to(dtype=torch.float16).to(gpu)
    vae_cache = [c.to(device=gpu, dtype=torch.float16) for c in ZERO_VAE_CACHE]
    print(f"[setup] stream VAE decoder loaded ({time.time()-t_script:.1f}s)")

# 文本条件(一次) + 首帧 latent
conditional_dict = pipeline.text_encoder(text_prompts=[CAPTION])
transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
image = transform(Image.open(IMG_PATH).convert("RGB")).unsqueeze(0).unsqueeze(2).to(device=gpu, dtype=torch.bfloat16)
initial_latent_0 = pipeline.vae.encode_to_latent(image).to(device=gpu, dtype=torch.bfloat16)
pipeline.vae.model.clear_cache()
torch.cuda.empty_cache()  # 释放文本编码器的瞬态缓存块
print(f"[setup] text+image encoded ({time.time()-t_script:.1f}s)")

# ---------------- 共享状态 ----------------
gen_q = queue.Queue(maxsize=400)
out_q = queue.Queue(maxsize=4000)
stats = {
    "lock": threading.Lock(),
    "latents": 0, "pixel_supply": 0, "rife_out": 0,
    "gen_t0": None, "gen_t1": None,
    "delivered": 0, "first_frame_latency": None,
    "e2e": [], "stall_s": 0.0, "peak_vram_gb": 0.0,
    "err": None,
}

def vram_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3

# ---------------- 生成线程 ----------------
def reset_kv():
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(1, torch.bfloat16, gpu)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, gpu)
    else:
        for b in pipeline.crossattn_cache: b["is_init"] = False
        for b in pipeline.kv_cache1:
            b["global_end_index"].zero_(); b["local_end_index"].zero_()

# 关键: PyTorch 梯度模式是线程本地状态, 模块级的 set_grad_enabled(False) 不作用于子线程。
# 子线程若带梯度运行, autograd 会保留全部激活(每次前向约 3.6GB), 显存冲满 24GB 后
# 在 WDDM 下溢出到共享内存, GPU 以约 1/100 速度挣扎。
@torch.no_grad()
def generator_thread():
    global lat_cache
    try:
        hot = initial_latent_0
        with stats["lock"]: stats["gen_t0"] = time.time()
        for seg in range(SEGMENTS):
            reset_kv()
            pipeline.generator(noisy_image_or_video=hot, conditional_dict=conditional_dict,
                               timestep=torch.zeros([1,1], device=gpu, dtype=torch.int64),
                               kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                               current_start=0)
            noise = torch.randn([1, BLOCKS_PER_SEG, 16, LAT_H, LAT_W], device=gpu, dtype=torch.bfloat16)
            for idx in range(BLOCKS_PER_SEG):
                t_blk = time.time()
                csf = 1 + idx
                dl = pipeline.denoising_step_list_first_chunk if idx == 0 else pipeline.denoising_step_list
                noisy_input = noise[:, idx:idx+1]
                for index, cur_t in enumerate(dl):
                    timestep = torch.ones([1,1], device=gpu, dtype=torch.int64) * cur_t
                    _, pred = pipeline.generator(noisy_image_or_video=noisy_input,
                                                 conditional_dict=conditional_dict, timestep=timestep,
                                                 kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                                                 current_start=csf * pipeline.frame_seq_length)
                    if index < len(dl) - 1:
                        nxt = dl[index+1]
                        noisy_input = pipeline.scheduler.add_noise(
                            pred.flatten(0,1), torch.randn_like(pred.flatten(0,1)),
                            nxt * torch.ones([1], device=gpu, dtype=torch.long)).unflatten(0, pred.shape[:2])
                if idx != BLOCKS_PER_SEG - 1:
                    pipeline.generator(noisy_image_or_video=pred, conditional_dict=conditional_dict,
                                       timestep=torch.zeros([1,1], device=gpu, dtype=torch.int64),
                                       kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                                       current_start=csf * pipeline.frame_seq_length)
                t_den = time.time() - t_blk
                t_dec = time.time()
                if args.taehv:
                    # TAEHV: 每块喂 [3帧latent上下文 + 新latent], 输出取末尾4帧(新内容)
                    p16 = pred.half()
                    if lat_cache is None:
                        px = decoder.decode_video(p16, parallel=True)          # NTCHW [0,1]
                        pixels = (px * 2.0 - 1.0)[:, 3:]                       # 首块: 1帧有效
                        lat_cache = p16.clone()
                    else:
                        lat_in = torch.cat([lat_cache, p16], dim=1)            # [1,4,16,60,104]
                        px = decoder.decode_video(lat_in, parallel=True)
                        pixels = (px * 2.0 - 1.0)[:, -4:]                      # 后续: 4帧新内容
                        lat_cache = lat_in[:, -3:]
                else:
                    pixels, vae_cache_new = decoder(pred.half(), *vae_cache)
                    vae_cache.clear(); vae_cache.extend(vae_cache_new)
                    if seg == 0 and idx == 0:
                        pixels = pixels[:, 3:]
                t_gen = time.time()
                for fi in range(pixels.shape[1]):
                    f = ((pixels[0, fi].float() + 1.0) * 0.5).clamp(0, 1).cpu()
                    gen_q.put((f, t_gen))
                with stats["lock"]:
                    stats["latents"] += 1
                    stats["pixel_supply"] += pixels.shape[1]
                hot = pred
                print(f"[gen] seg{seg} blk{idx:02d} denoise {t_den:.2f}s decode {time.time()-t_dec:.2f}s +{pixels.shape[1]}px")
        with stats["lock"]: stats["gen_t1"] = time.time()
        gen_q.put(None)
    except Exception as e:
        import traceback; traceback.print_exc()
        with stats["lock"]: stats["err"] = f"gen: {e}"
        gen_q.put(None)

# ---------------- RIFE 插帧线程 ----------------
# RIFE 跑在独立 CUDA 流上: 与生成线程的默认流并行执行,
# 避免两线程 kernel 在同一条流里串行排队互相阻塞(MULT越大争用越重)
rife_stream = torch.cuda.Stream()
@torch.no_grad()
def rife_thread():
    try:
        prev = None
        while True:
            item = gen_q.get()
            if item is None:
                # prev 已在上一轮输出, 无需重复推送
                out_q.put(None); break
            f, t_gen = item
            if prev is not None:
                p, _ = prev
                with torch.cuda.stream(rife_stream):
                    x0 = p.unsqueeze(0).cuda(); x1 = f.unsqueeze(0).cuda()
                    for k in range(1, MULT):
                        mid = rife_model(x0, x1, k / MULT, [8,4,2,1], training=False, fastmode=True, ensemble=False)
                        out_q.put((mid[0].float().cpu(), t_gen))
                    del x0, x1
                with stats["lock"]: stats["rife_out"] += MULT  # (MULT-1)中间帧 + 当前帧
            else:
                with stats["lock"]: stats["rife_out"] += 1
            out_q.put((f, t_gen))
            prev = (f, t_gen)
    except Exception as e:
        import traceback; traceback.print_exc()
        with stats["lock"]: stats["err"] = f"rife: {e}"
        out_q.put(None)

# ---------------- 输出线程(模拟直播播放器) ----------------
def writer_thread():
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-framerate", str(int(TARGET_FPS)), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)
    t0 = None; delivered = 0; stall = 0.0
    try:
        while True:
            item = out_q.get()
            if item is None: break
            f, t_gen = item
            now = time.time()
            if t0 is None:
                t0 = now
                with stats["lock"]:
                    if stats["gen_t0"] is not None:
                        stats["first_frame_latency"] = now - stats["gen_t0"]
            sched = t0 + delivered / TARGET_FPS
            if now < sched:
                time.sleep(sched - now)
            else:
                stall += now - sched
            rgb = (f.permute(1,2,0).numpy() * 255).astype(np.uint8)
            proc.stdin.write(rgb.tobytes())
            delivered += 1
            with stats["lock"]:
                stats["delivered"] = delivered
                stats["stall_s"] = stall
                stats["e2e"].append(time.time() - t_gen)
    except Exception as e:
        import traceback; traceback.print_exc()
        with stats["lock"]: stats["err"] = f"writer: {e}"
    finally:
        try: proc.stdin.close()
        except Exception: pass
        proc.wait()

# ---------------- 监控线程 ----------------
def monitor_thread():
    while any(t.is_alive() for t in threads):
        with stats["lock"]:
            v = vram_gb()
            stats["peak_vram_gb"] = max(stats["peak_vram_gb"], v)
            print(f"[mon] vram {v:.1f}GB | gen_q {gen_q.qsize()} out_q {out_q.qsize()} | "
                  f"supply {stats['pixel_supply']} rife {stats['rife_out']} delivered {stats['delivered']}")
        time.sleep(5)

threads = [threading.Thread(target=generator_thread), threading.Thread(target=rife_thread), threading.Thread(target=writer_thread)]
mon = threading.Thread(target=monitor_thread, daemon=True)
for t in threads: t.start()
mon.start()
for t in threads: t.join()

# ---------------- 汇总 ----------------
with stats["lock"]:
    s = dict(stats)
wall = time.time() - s["gen_t0"] if s["gen_t0"] else 0
gen_wall = (s["gen_t1"] or time.time()) - s["gen_t0"] if s["gen_t0"] else 0
supply_fps = s["pixel_supply"] / gen_wall if gen_wall > 0 else 0
delivered_fps = s["delivered"] / wall if wall > 0 else 0
rt_factor = (s["pixel_supply"] / 16.0) / wall if wall > 0 else 0
e2e = s["e2e"]
print("\n" + "=" * 60)
print("直播流式管线实测汇总 (CF++ 2-step 1.3B @832x480 + RIFE x%d)" % MULT)
print("=" * 60)
print(f"运行时长(生成开始→播放结束): {wall:.1f}s | 段数: {SEGMENTS}x{BLOCKS_PER_SEG} latent帧")
print(f"latent生成速率: {s['latents']/gen_wall:.2f} 帧/s | 像素供给: {supply_fps:.2f} 帧/s")
print(f"RIFE输出: {s['rife_out']} 帧 | 播放器交付: {s['delivered']} 帧")
print(f"★ 可持续直播FPS(交付/墙钟): {delivered_fps:.2f}")
print(f"★ 实时率(内容速度/墙钟): {rt_factor:.2f}x  (1.0x=真实时; <1 为慢动作/延迟累积)")
print(f"首帧延迟(生成开始→首帧上屏): {s['first_frame_latency']:.2f}s" if s['first_frame_latency'] else "首帧延迟: n/a")
if e2e:
    print(f"端到端延迟(生成完→上屏): 均值 {sum(e2e)/len(e2e):.2f}s / 最大 {max(e2e):.2f}s")
print(f"播放等待(卡顿累计): {s['stall_s']:.1f}s | 峰值显存: {s['peak_vram_gb']:.1f}GB")
if s["err"]: print(f"⚠ 错误: {s['err']}")
print(f"输出视频: {args.out}")
with open(os.path.join(os.path.dirname(args.out), "stream_stats.json"), "w") as fp:
    json.dump({"wall_s": wall, "supply_fps": supply_fps, "delivered_fps": delivered_fps,
               "rt_factor": rt_factor, "first_frame_latency": s["first_frame_latency"],
               "stall_s": s["stall_s"], "peak_vram_gb": s["peak_vram_gb"], "err": s["err"]}, fp, indent=2)
