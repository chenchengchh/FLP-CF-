# 常驻直播服务: CF++ 2-step 1.3B @480x832 竖屏 + TAEHV + RIFE xN
# 浏览器打开 http://127.0.0.1:8390 实时观看, 页面统计 播放FPS/延迟/卡顿
# 只使用 CF++ 模型直播: 每段从源图重锚定(画质/位置每段重置) + 段间 crossfade
import argparse, base64, json, os, sys, time, queue, threading, types, importlib.util, itertools, logging
import torch
import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf
from flask import Flask, Response, request, jsonify, send_file
import concurrent.futures

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, default=20, help="每段 latent 块数")
parser.add_argument("--mult", type=int, default=10, help="RIFE 插帧倍数")
parser.add_argument("--rife_gpu", type=int, default=1,
                    help="RIFE 所在 GPU 序号 (默认 1=3050; CF++/TAEHV 固定 cuda:0=3090, 解除插帧对生成的 GPU 竞争)")
parser.add_argument("--lipsync", action="store_true",
                    help="启用 MuseTalk V15 实时口型 (fp16, ~2.5GB 显存); "
                         "说话段 RIFE 自动降档到 --lipsync_mult, 静默段零开销")
parser.add_argument("--lipsync_gpu", type=int, default=0,
                    help="MuseTalk 所在 GPU 序号 (默认 0=3090, 与 RIFE 的 3050 解耦: "
                         "实测同卡争抢是口型 3.3fps 的根源, 138ms+RIFE170ms 同卡串行; "
                         "3090 单帧 ~60ms, 3050 独占给 RIFE)")
parser.add_argument("--lipsync_mult", type=int, default=6,
                    help="说话段 RIFE 插帧倍数 (口型节流后源帧 2.85fps x6 = 17.1fps ≥ 17 播放节拍)")
parser.add_argument("--target_fps", type=float, default=17.0, help="前端播放节拍 fps")
parser.add_argument("--port", type=int, default=8390)
parser.add_argument("--skip_commit", action="store_true",
                    help="实验: 跳过 KV 提交前向(省1前向/块)——上下文留噪声KV, 画质可能崩, 仅用于鲁棒性验证")
parser.add_argument("--height", type=int, default=832, help="竖屏直播高 (px, 8 的倍数)")
parser.add_argument("--width", type=int, default=480, help="竖屏直播宽 (px, 8 的倍数); 320x560 约 2x 提速但非原生分辨率, 画质待验证")
parser.add_argument("--jpeg_q", type=int, default=92)
parser.add_argument("--caption", type=str, default=None,
                    help="覆盖默认提示词; 手势描述越简单(如双手交握/自然下垂)手部越不容易变形")
parser.add_argument("--ckpt", type=str, default=None,
                    help="CF++ 权重路径 (默认 framewise-2step.pt; 1-step 用 framewise-1step.pt 提速约2x)")
parser.add_argument("--cfg", type=str, default=None,
                    help="config yaml 路径 (1-step 配套 configs/causal_forcing_dmd_framewise_1step_1p3b_384.yaml)")
parser.add_argument("--cudagraphs", action="store_true",
                    help="torch.compile(backend=cudagraphs) 捕获 DiT 前向 (不依赖 triton, Windows 可用); "
                         "KV 序列每块变化会逐 shape 捕获, 第二段起全命中; 消除 launch-bound 开销")
parser.add_argument("--reanchor", choices=["source", "hot"], default="source",
                    help="source=每段从源图干净重锚定(画质每段重置); hot=自回归传递上一段末帧(动作连续但缓慢漂移)")
parser.add_argument("--anchor_every", type=int, default=6,
                    help="仅 reanchor=hot 时生效: 每隔 N 段把当前帧与源图 latent 混合重注入; 0=关闭")
parser.add_argument("--anchor_alpha", type=float, default=0.15,
                    help="仅 reanchor=hot 时生效: 锚帧混合强度 (0~1)")
parser.add_argument("--fade", type=int, default=12,
                    help="段边界 crossfade 帧数 (12帧=0.8s@15fps 溶解过渡, 避免重锚定姿态跳变被看成瞬移)")
args = parser.parse_args()
torch.manual_seed(0)
ANCHOR_EVERY = args.anchor_every
ANCHOR_ALPHA = args.anchor_alpha
REANCHOR = args.reanchor
FADE = args.fade

BLOCKS_PER_SEG = args.blocks
SKIP_COMMIT = args.skip_commit
MULT = args.mult
TARGET_FPS = args.target_fps
H, W = args.height, args.width   # 竖屏直播分辨率 (8 的倍数); 非原生分辨率画质自担
LAT_H, LAT_W = H // 8, W // 8   # TAEHV 8x 空间压缩, 随分辨率联动
# pipeline.frame_seq_length 在 pipeline 初始化后联动 (每帧 KV token 数 = LAT_H*LAT_W/4)
# 手部防变形/防瞬移: 双手交握且几乎静止(almost still) + 手指自然成形 + 动作幅度压到最小
# (1.3B 模型手部时序一致性弱, 手动得越少越不变形不瞬移; 蒸馏模型无 CFG, 负面词无效, 靠正面引导)
CAPTION = ("A young woman keeps both hands gently clasped together and almost still in front of her chest, "
           "fingers naturally formed and clearly detailed, only very subtle slow body movements, "
           "upper body view, simple clean gray background, calm expression.")
if args.caption:
    CAPTION = args.caption
IMG_PATH = r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\i2v_std\15-26\portrait.png"
CKPT = args.ckpt or r"D:\FasterLivePortrait\causal_forcing_poc\cfpp\cfpp_git\causal-forcing++\framewise-2step.pt"
CFG = args.cfg or "configs/causal_forcing_dmd_framewise_2step_1p3b_384.yaml"
RIFE_ARCH = r"F:\2026-ComfyUI-V8.3\custom_nodes\comfyui-frame-interpolation\vfi_models\rife\rife_arch.py"
RIFE_CKPT = r"F:\2026-ComfyUI-V8.3\models\frame_interpolation\rife47.pth"

torch.set_grad_enabled(False)
gpu = torch.device("cuda")
t_script = time.time()

# ---------------- RIFE ----------------
_stub = types.ModuleType("comfy.model_management")
# rife_arch.py 模块导入时用 get_torch_device() 固定 backwarp 网格缓存设备, 必须指向 RIFE 实际 GPU
_stub.get_torch_device = lambda: torch.device(f"cuda:{args.rife_gpu}")
sys.modules["comfy"] = types.ModuleType("comfy")
sys.modules["comfy.model_management"] = _stub
_spec = importlib.util.spec_from_file_location("rife_arch", RIFE_ARCH)
_ra = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ra)
rife_model = _ra.IFNet(arch_ver="4.7")
rife_model.load_state_dict(torch.load(RIFE_CKPT, map_location="cpu"))
rife_dev = torch.device(f"cuda:{args.rife_gpu}")
rife_model.eval().to(rife_dev)
# RIFE cudagraphs 不可行: forward 的 timestep 是 python float (k/MULT 变化) 触发反复 recompile;
# fp16 被内部 float 强转阻断; launch-bound 只能靠降 MULT 减少每源帧推理次数 (MULT=4: 3次x70ms=210ms -> 4.8fps 消费)
_ = rife_model(torch.rand(1,3,H,W).to(rife_dev), torch.rand(1,3,H,W).to(rife_dev), 0.5, [8,4,2,1], training=False, fastmode=True, ensemble=False)
print(f"[setup] RIFE loaded on cuda:{args.rife_gpu} ({time.time()-t_script:.1f}s)")

# ---------------- MuseTalk 口型 (可选, 与 RIFE 同卡, 静默段零开销) ----------------
lipsync = None
if args.lipsync:
    try:
        from lipsync_musetalk import MuseTalkLipSync
        lipsync = MuseTalkLipSync(device=f"cuda:{args.lipsync_gpu}", use_fp16=True)
        # 预热: unet torch.compile 在首次真实推理时才编译(~1min), 不预热则第一句口型线程停摆,
        # gen_q 积压数十帧 (实测句释放点 off 从 10.6s 恶化到 21.2s)。用源图(含人脸)走完整链路
        _wimg = np.asarray(Image.open(IMG_PATH).convert("RGB").resize((W, H))).astype(np.float32) / 255.0
        _warm = torch.from_numpy(_wimg).permute(2, 0, 1)
        _wfeat = lipsync.extract_feature(np.zeros(16000, dtype=np.float32))
        for _ in range(3):
            lipsync.process_frame(_warm, _wfeat, 0.5)
        print(f"[setup] MuseTalk loaded+warm ({time.time()-t_script:.1f}s)", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[setup] MuseTalk 加载失败, 口型关闭: {e}", flush=True)
        lipsync = None

# ---------------- CF++ pipeline ----------------
from pipeline import CausalInferencePipeline
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from demo_utils.taehv import TAEHV

config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load("configs/causal_forcing_dmd_framewise_2step_1p3b_384.yaml"))
# 384 config 未带滚动窗口/sink 配置; 直播段内也开启: local 窗口 21 帧 + sink 1 帧(源图锚),
# 抑制段内漂移(不修改 config 文件, 非侵入注入)
config.model_kwargs.local_attn_size = 21
config.model_kwargs.sink_size = 1
pipeline = CausalInferencePipeline(config, device=gpu)
sd = torch.load(CKPT, map_location="cpu")
try:
    pipeline.generator.load_state_dict(sd["generator_ema"])
except RuntimeError:
    fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in sd["generator_ema"].items()}
    pipeline.generator.load_state_dict(fixed, strict=False)
pipeline = pipeline.to(dtype=torch.bfloat16)
if get_cuda_free_memory_gb(gpu) < 40:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)
if args.cudagraphs:
    # denoise 实测 0.82s/块 vs 理论 flops ~0.17s -> launch-bound;
    # backend=cudagraphs 只做图捕获不需要 triton/inductor, Windows 可用
    print("[setup] torch.compile(generator.model, backend=cudagraphs) ...", flush=True)
    pipeline.generator.model = torch.compile(pipeline.generator.model, backend="cudagraphs", fullgraph=False)
# 每帧 KV token 数 = LAT_H*LAT_W/4 (Wan latent patch 2x2), 必须随分辨率联动否则 KV cache 布局崩溃
pipeline.frame_seq_length = LAT_H * LAT_W // 4
print(f"[setup] pipeline loaded ({time.time()-t_script:.1f}s) | {W}x{H} latent {LAT_W}x{LAT_H} seq/frame {pipeline.frame_seq_length}")

# TAEHV 轻量解码 (稳态 ~75ms/块, Wan VAE 为 ~440ms/块)
decoder = TAEHV(checkpoint_path="checkpoints/taew2_1.pth").eval().to(dtype=torch.float16).to(gpu)
print(f"[setup] TAEHV decoder loaded ({time.time()-t_script:.1f}s)")

# ---------------- 动态 caption: 文本→动作 (下一段生效, ~20s) ----------------
# CF++ 扩散模型无音频口型能力(口型同步需 TTS+FLP 音频驱动, 属 FLP+CF++ 组合管线);
# 此处把用户文本关键词映射为动作短语, 靠正面提示词驱动简单手部/表情动作
ACTION_MAP = [
    ("招手", "waves her right hand in a friendly greeting"),
    ("挥手", "waves her right hand in a friendly greeting"),
    ("打招呼", "waves her right hand in a friendly greeting"),
    ("再见", "waves her right hand goodbye"),
    ("点头", "nods her head gently in agreement"),
    ("同意", "nods her head gently in agreement"),
    ("摇头", "shakes her head gently"),
    ("点赞", "gives a thumbs up with her right hand"),
    ("竖大拇指", "gives a thumbs up with her right hand"),
    ("鼓掌", "claps her hands gently a few times"),
    ("拍手", "claps her hands gently a few times"),
    ("思考", "rests her chin lightly on one hand, thinking"),
    ("托腮", "rests her chin lightly on one hand, thinking"),
    ("说话", "talks expressively with small natural hand gestures"),
    ("讲解", "talks expressively with small natural hand gestures"),
    ("比心", "makes a heart shape with both hands in front of her chest"),
    ("微笑", "smiles warmly"),
    ("大笑", "laughs happily"),
    ("开心", "smiles warmly"),
    ("静止", "keeps both hands gently clasped together and almost still"),
]
DEFAULT_CAPTION = ("A young woman keeps both hands gently clasped together and almost still in front of her chest, "
                   "fingers naturally formed and clearly detailed, only very subtle slow body movements, "
                   "upper body view, simple clean gray background, calm expression.")

def build_caption(text):
    """用户文本 → 动作描述: 命中关键词用首个动作短语, 未命中用说话/默认兜底; 保留人物+质量底板."""
    t = (text or "").strip()
    if not t:
        return DEFAULT_CAPTION
    action = None
    for kw, phrase in ACTION_MAP:
        if kw in t:
            action = phrase
            break
    if action is None:
        # 未命中任何动作词: 视为说话内容, 用讲解姿态(模拟说话感, 非真实口型)
        action = "talks expressively with small natural hand gestures"
    return (f"A young woman {action}, "
            "fingers naturally formed and clearly detailed, upper body view, "
            "simple clean gray background, calm expression.")

caption_state = {"lock": threading.Lock(), "text": "", "caption": DEFAULT_CAPTION}

_cond_cache = {"cap": None, "cond": None}

def get_conditional():
    """caption 未变化时复用已编码结果, 省去每段 ~0.5-1s 的 UMT5 重编码"""
    with caption_state["lock"]:
        cap = caption_state["caption"]
    if _cond_cache["cap"] != cap:
        _cond_cache["cond"] = pipeline.text_encoder(text_prompts=[cap])
        _cond_cache["cap"] = cap
        print(f"[caption] (re)encoded: {cap[:60]}...", flush=True)
    return _cond_cache["cond"], cap

conditional_dict = pipeline.text_encoder(text_prompts=[DEFAULT_CAPTION])
if args.caption:
    DEFAULT_CAPTION = args.caption
    caption_state["caption"] = DEFAULT_CAPTION
    conditional_dict = pipeline.text_encoder(text_prompts=[DEFAULT_CAPTION])
from torchvision import transforms
transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
image = transform(Image.open(IMG_PATH).convert("RGB")).unsqueeze(0).unsqueeze(2).to(device=gpu, dtype=torch.bfloat16)
initial_latent_0 = pipeline.vae.encode_to_latent(image).to(device=gpu, dtype=torch.bfloat16)
pipeline.vae.model.clear_cache()
torch.cuda.empty_cache()
print(f"[setup] text+image encoded ({time.time()-t_script:.1f}s)")

# ---------------- 共享状态 ----------------
gen_q = queue.Queue(maxsize=400)
out_q = queue.Queue(maxsize=4000)
seq_counter = itertools.count()
stats = {
    "lock": threading.Lock(),
    "pixel_supply": 0,       # 累计像素帧
    "rife_out": 0,
    "gen_t0": None,
    "err": None,
    "peak_vram_gb": 0.0,
    # 滑动窗口供采样
    "samples": [],           # [(t, cum_supply)]
}

def vram_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3

# ---------------- 语音口型状态 (Flask 线程写, 生成/口型/RIFE 线程读) ----------------
# t0 语义: 说话段首帧的 t_gen (生成钟); 帧 f 的音频偏移 = f.t_gen - t0, 前端播放延迟 L 恒定时
# 音画天然对齐 (帧显示于 t_gen+L, 音频起点=首帧显示时刻=t0+L)
speech_state = {
    "lock": threading.Lock(),
    "feat": None,    # whisper 特征 [1,T50,5,384] (cuda:rife_gpu), None=静默
    "t0": None,      # 说话段首帧 t_gen; None=已合成等段首
    "dur": 0.0,      # 音频时长秒
    "url": "",       # /audio/xxx.mp3 (前端播放)
    "seg": None,     # 说话段起始 seg (SSE 事件用)
    "id": 0,         # 句 id (SSE 去重)
}

def _mark_speech_start(seg, t_gen):
    """说话段首帧钩子: 登记句起始时刻; 返回句 id (无 pending 返回 None)"""
    with speech_state["lock"]:
        if speech_state["feat"] is not None and speech_state["t0"] is None:
            speech_state["t0"] = t_gen
            speech_state["seg"] = seg
            return speech_state["id"]
    return None

def reset_kv():
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(1, torch.bfloat16, gpu)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, gpu)
    else:
        for b in pipeline.crossattn_cache: b["is_init"] = False
        for b in pipeline.kv_cache1:
            b["global_end_index"].zero_(); b["local_end_index"].zero_()

# PyTorch 梯度模式是线程本地状态, 子线程必须显式 no_grad
@torch.no_grad()
def generator_thread():
    lat_cache = None
    tail = None      # 上一段末尾的原生帧(未混合), 用于下一段开头 crossfade
    try:
        hot = initial_latent_0
        with stats["lock"]: stats["gen_t0"] = time.time()
        seg = 0
        while True:
            reset_kv()
            # 每段开始重新编码当前 caption (文本→动作 下一段生效, ~0.5s/段)
            conditional_dict, _cap_now = get_conditional()
            fade_k = 0       # 本段已 crossfade 帧计数
            seg_tail = []    # 本段原生尾帧滚动缓存(末 FADE 帧)
            first_put = True  # 段首帧钩子标记 (口型句起始登记)
            if REANCHOR == "source":
                # 分段重锚定: 每段从源图干净重启, 画质/位置/尺度每段重置;
                # 动作跳变由段间 crossfade 平滑; TAEHV 上下文属于上一段内容, 必须重置
                feed = initial_latent_0
                lat_cache = None
                tag = " [REANCHOR]"
            else:
                # 自回归连续模式: 传上一段末帧保持动作连续, 周期性做源图 latent 混合拉回漂移
                anchor = ANCHOR_EVERY > 0 and (seg % ANCHOR_EVERY) == 0
                feed = hot
                if anchor:
                    feed = ANCHOR_ALPHA * initial_latent_0 + (1.0 - ANCHOR_ALPHA) * hot
                tag = " [ANCHOR]" if anchor else ""
            pipeline.generator(noisy_image_or_video=feed, conditional_dict=conditional_dict,
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
                if idx != BLOCKS_PER_SEG - 1 and not SKIP_COMMIT:
                    # KV 提交前向: 干净 pred(t=0) 覆盖本块 slot 里的噪声 KV, 保证下块上下文干净.
                    # SKIP_COMMIT 实验: 验证 DMD 蒸馏模型对噪声 KV 上下文的鲁棒性 (省 1 前向/块, ~2x 潜力)
                    pipeline.generator(noisy_image_or_video=pred, conditional_dict=conditional_dict,
                                       timestep=torch.zeros([1,1], device=gpu, dtype=torch.int64),
                                       kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                                       current_start=csf * pipeline.frame_seq_length)
                t_den = time.time() - t_blk
                t_dec = time.time()
                # TAEHV 增量解码: [3帧latent上下文 + 新latent] -> 取末尾4帧新内容
                p16 = pred.half()
                if lat_cache is None:
                    px = decoder.decode_video(p16, parallel=True)
                    pixels = (px * 2.0 - 1.0)[:, 3:]
                    lat_cache = p16.clone()
                else:
                    lat_in = torch.cat([lat_cache, p16], dim=1)
                    px = decoder.decode_video(lat_in, parallel=True)
                    pixels = (px * 2.0 - 1.0)[:, -4:]
                    lat_cache = lat_in[:, -3:]
                t_gen = time.time()
                for fi in range(pixels.shape[1]):
                    f = ((pixels[0, fi].float() + 1.0) * 0.5).clamp(0, 1).cpu()
                    seg_tail.append(f)
                    out = f
                    if tail is not None and fade_k < FADE:
                        # 段间 crossfade: 新段前 FADE 帧与旧段尾帧线性混合, 平滑姿态/位置跳变
                        w = (fade_k + 1) / (FADE + 1)
                        old = tail[min(fade_k, len(tail) - 1)]
                        out = (1.0 - w) * old + w * f
                        fade_k += 1
                    if first_put:
                        first_put = False
                        _mark_speech_start(seg, t_gen)   # 口型: 登记说话段首帧时刻
                    # 帧戳按 2.8fps 源率块内错开: 4 帧共享 t_gen 会让口型 offset 量化到
                    # 1.43s 步进 (块内 4 帧 mouth 冻结后跳变, 音画偏差最大 ~1s)
                    gen_q.put((out, t_gen + fi / 2.8, seg))
                with stats["lock"]:
                    stats["pixel_supply"] += pixels.shape[1]
                    stats["samples"].append((time.time(), stats["pixel_supply"]))
                    stats["samples"] = stats["samples"][-30:]
                hot = pred
                print(f"[gen] seg{seg}{tag if idx==0 else ''} blk{idx:02d} denoise {t_den:.2f}s decode {time.time()-t_dec:.2f}s +{pixels.shape[1]}px", flush=True)
                # 说话段源帧节流: 口型重绘 3.2fps 跟不上 4.4fps 源流, pacing 到 2.85fps
                # (x6 插帧 = 17.1fps > 17 播放节拍, 前端 2s 缓冲不空心; 16.8<17 会缓慢耗尽缓冲致卡顿)
                if LIPSYNC_ACTIVE:
                    with speech_state["lock"]:
                        speaking = speech_state["feat"] is not None
                    if speaking:
                        budget = pixels.shape[1] / 2.85
                        el = time.time() - t_blk
                        if el < budget:
                            time.sleep(budget - el)
            tail = seg_tail[-FADE:] if seg_tail else None
            seg += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        with stats["lock"]: stats["err"] = f"gen: {e}"
        gen_q.put(None)

# RIFE 在 3050 (cuda:1) 独立 GPU 上, 与 3090 的 CF++ 天然并行, 无需独立 CUDA 流
# (cudagraphs 要求默认流, 自定义流会使其失效回退 eager)

# ---------------- 口型线程: gen_q → [逐帧口型重绘] → gen_q2 → RIFE ----------------
LIPSYNC_ACTIVE = lipsync is not None
rife_in_q = gen_q
if LIPSYNC_ACTIVE:
    gen_q2 = queue.Queue(maxsize=400)
    rife_in_q = gen_q2

@torch.no_grad()
def lipsync_thread():
    """消费 gen_q: 说话段内逐帧 MuseTalk 重绘口型 (3050), 静默帧直通; 句尾 +0.6s 闭嘴过渡"""
    stat = {"n": 0, "t": 0.0}   # 本句重绘帧数/累计耗时 (实测服务内 MuseTalk fps)
    try:
        while True:
            # 反压保护: 口型重绘暂时落后时丢最旧源帧防延迟累积
            if gen_q.qsize() > 60:
                while gen_q.qsize() > 30:
                    try: gen_q.get_nowait()
                    except queue.Empty: break
            item = gen_q.get()
            if item is None:
                gen_q2.put(None); break
            f, t_gen, seg = item
            with speech_state["lock"]:
                feat, t0, dur = speech_state["feat"], speech_state["t0"], speech_state["dur"]
                # 句已播完 (+2s 容忍): 释放 whisper 特征防显存积累, 回到静默零开销
                # (url/seg 一并清空: 残留会让新连接的观众收到过期 spk 事件错播旧音频)
                if feat is not None and t0 is not None and t_gen - t0 > dur + 2.0:
                    speech_state["feat"] = None
                    speech_state["url"] = ""
                    speech_state["seg"] = None
                    extra = (f" 重绘{stat['n']}帧 avg {stat['t']/stat['n']*1000:.0f}ms"
                             f" ({stat['n']/stat['t']:.2f}fps)") if stat["n"] else ""
                    print(f"[lipsync] 句结束, 回静默 (off={t_gen - t0:.1f}s){extra}", flush=True)
                    stat = {"n": 0, "t": 0.0}
                    feat = None
            if feat is not None and t0 is not None and 0.0 <= t_gen - t0 <= dur + 0.6:
                try:
                    t_p = time.time()
                    f = lipsync.process_frame(f, feat, t_gen - t0)
                    stat["n"] += 1; stat["t"] += time.time() - t_p
                except Exception as e:
                    print(f"[lipsync] 帧处理失败, 跳过重绘: {type(e).__name__}: {e}", flush=True)
            gen_q2.put((f, t_gen, seg))
    except Exception as e:
        import traceback; traceback.print_exc()
        with stats["lock"]: stats["err"] = f"lipsync: {e}"
        gen_q2.put(None)

def out_put(item):
    """队满时丢最旧帧: 无观众时 RIFE 照常消费 gen_q 会把 out_q 堆满(4000帧≈4.5分钟),
    观众重连后从队头吐旧帧导致延迟爆炸; drop-oldest 保证重连延迟有界."""
    if out_q.full():
        try: out_q.get_nowait()
        except queue.Empty: pass
    out_q.put(item)

@torch.no_grad()
def rife_thread():
    try:
        prev = None
        cur_mult = MULT
        while True:
            # 反压保护: 消费暂时落后时丢最旧源帧防延迟累积 (口型线程启用时 gen_q 的反压在 lipsync_thread)
            if rife_in_q.qsize() > 60:
                while rife_in_q.qsize() > 30:
                    try: rife_in_q.get_nowait()
                    except queue.Empty: break
                prev = None   # 丢帧后两帧跨度过大, 重置插帧链
            item = rife_in_q.get()
            if item is None:
                out_put(("err", None)); break
            f, t_gen, seg = item
            # 说话段动态降档: MULT 10→4, 让出 3050 算力给口型重绘 (4.4源fps x4=17.6fps 仍达标)
            if LIPSYNC_ACTIVE:
                with speech_state["lock"]:
                    speaking = speech_state["feat"] is not None
                want = args.lipsync_mult if speaking else MULT
                if want != cur_mult:
                    cur_mult = want
                    print(f"[rife] MULT -> {cur_mult} ({'口型段' if speaking else '静默段'})", flush=True)
            if prev is not None:
                p, pt_gen, pseg = prev
                x0 = p.unsqueeze(0).to(rife_dev); x1 = f.unsqueeze(0).to(rife_dev)
                if cur_mult > 3:
                    # batch 化 + 降采样插帧 (mult>=4 才需要: 3050 逐帧消费跟不上供给)
                    B = cur_mult - 1
                    x0s = torch.nn.functional.interpolate(x0, size=(560, 320), mode="bilinear", align_corners=False)
                    x1s = torch.nn.functional.interpolate(x1, size=(560, 320), mode="bilinear", align_corners=False)
                    ts = (torch.arange(1, cur_mult, device=rife_dev, dtype=torch.float32) / cur_mult).view(-1, 1, 1, 1)  # [B,1,1,1] 契合 IFNet L394 repeat(1,1,h,w)
                    mids = rife_model(x0s.expand(B, -1, -1, -1), x1s.expand(B, -1, -1, -1), ts,
                                      [8, 4, 2, 1], training=False, fastmode=True, ensemble=False)
                    for j in range(B):
                        mid = torch.nn.functional.interpolate(mids[j:j+1].float(), size=(H, W), mode="bilinear", align_corners=False)
                        # 插值帧时间戳线性内插 + 沿用前帧 seg: 说话段首帧前的 5 张插值帧若带新 seg,
                        # SSE 会提前 5 拍(~0.29s)触发音频起播 → 声音领先口型
                        out_put(("frame", (mid[0].cpu(), pt_gen + (j + 1) / cur_mult * (t_gen - pt_gen), pseg)))
                elif cur_mult > 1:
                    # 逐帧插帧 (mult<=3): 产帧节奏均匀 (~每57.4ms 1帧), 与节拍对齐避免突发导致的顿挫
                    for k in range(1, cur_mult):
                        mid = rife_model(x0, x1, k / cur_mult, [8, 4, 2, 1], training=False, fastmode=True, ensemble=False)
                        out_put(("frame", (mid[0].float().cpu(), pt_gen + (k + 1) / cur_mult * (t_gen - pt_gen), pseg)))
                del x0, x1
            out_put(("frame", (f, t_gen, seg)))
            prev = (f, t_gen, seg)
    except Exception as e:
        import traceback; traceback.print_exc()
        with stats["lock"]: stats["err"] = f"rife: {e}"
        out_q.put(("err", None))

def monitor_thread():
    while True:
        with stats["lock"]:
            v = vram_gb()
            stats["peak_vram_gb"] = max(stats["peak_vram_gb"], v)
            print(f"[mon] vram {v:.1f}GB | gen_q {gen_q.qsize()} out_q {out_q.qsize()} | supply {stats['pixel_supply']}", flush=True)
        time.sleep(10)

# ---------------- Flask + SSE ----------------
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CF++ 竖屏直播</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0d1117; color:#e6edf3; font-family:"Segoe UI",system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; min-height:100vh; padding:16px; }
  .topbar { width:auto; display:flex; align-items:center; gap:10px; margin-bottom:10px;
            font-size:14px; color:#8b949e; }
  .dot { width:9px; height:9px; border-radius:50%; background:#f85149; animation:blink 1.2s infinite; }
  @keyframes blink { 50% { opacity:.25; } }
  .wrap { position:relative; }
  canvas { background:#000; height:78vh; width:auto; max-width:96vw; border-radius:10px; display:block;
           box-shadow:0 0 40px rgba(248,81,73,.12); }
  .badge { position:absolute; top:12px; left:12px; background:rgba(248,81,73,.92); color:#fff;
           font-size:12px; font-weight:700; padding:3px 9px; border-radius:4px; letter-spacing:1px; }
  #segTag { position:absolute; top:12px; right:12px; background:rgba(13,17,23,.72); color:#e6edf3;
           font-size:12px; padding:3px 9px; border-radius:4px; font-variant-numeric:tabular-nums; }
  #stallTip { position:absolute; bottom:12px; left:50%; transform:translateX(-50%);
           background:rgba(248,81,73,.95); color:#fff; font-size:12px; padding:3px 10px;
           border-radius:4px; display:none; }
  #spkTag { position:absolute; bottom:12px; right:12px; background:rgba(35,134,54,.92); color:#fff;
           font-size:12px; font-weight:700; padding:3px 10px; border-radius:4px; display:none;
           align-items:center; gap:6px; }
  #spkTag::before { content:''; width:7px; height:7px; border-radius:50%; background:#7ee787;
           animation:blink 1s infinite; }
  .grid { width:min(480px,96vw); display:grid; grid-template-columns:repeat(3,1fr);
          gap:8px; margin-top:10px; }
  .cell { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:8px 10px; }
  .cell .k { font-size:11px; color:#8b949e; margin-bottom:2px; }
  .cell .v { font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; }
  .good { color:#3fb950; } .warn { color:#d29922; } .bad { color:#f85149; }
  .hint { width:min(480px,96vw); margin-top:10px; font-size:12px; color:#8b949e; line-height:1.6; }
  .ctrl { width:min(480px,96vw); margin-top:14px; }
  .ctrl-row { display:flex; gap:8px; }
  #capIn { flex:1; padding:10px 12px; border-radius:10px; border:1px solid #30363d;
           background:#0d1117; color:#e6edf3; font-size:14px; outline:none; }
  #capIn:focus { border-color:#58a6ff; }
  #capBtn { padding:10px 18px; border-radius:10px; border:none; background:#238636;
            color:#fff; font-size:14px; cursor:pointer; }
  #capBtn:hover { background:#2ea043; }
  #micBtn { padding:10px 14px; border-radius:10px; border:1px solid #30363d; background:#161b22;
            color:#e6edf3; font-size:14px; cursor:pointer; }
  #micBtn.rec { background:#da3633; border-color:#da3633; animation:pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:0.55; } }
  .chips { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }
  .chip { padding:6px 12px; border-radius:999px; border:1px solid #30363d; background:#161b22;
          color:#c9d1d9; font-size:12px; cursor:pointer; }
  .chip:hover { border-color:#58a6ff; color:#58a6ff; }
  .cap-msg { margin-top:8px; font-size:12px; color:#8b949e; min-height:16px; }
</style>
</head>
<body>
<div class="topbar"><span class="dot"></span>LIVE · CF++ 2-step 1.3B · 480×832 竖屏 · TAEHV + RIFE ×__MULT__ · __TARGET_FPS__ fps · 每段重锚定</div>
<div class="wrap">
  <canvas id="stage" width="480" height="832"></canvas>
  <div class="badge">LIVE</div>
  <div id="segTag">SEG –</div>
  <div id="stallTip">卡顿中…</div>
  <div id="spkTag">说话中</div>
</div>
<div class="grid">
  <div class="cell"><div class="k">服务器供给 FPS</div><div class="v" id="sSupply">–</div></div>
  <div class="cell"><div class="k">播放 FPS</div><div class="v" id="sPlay">–</div></div>
  <div class="cell"><div class="k">端到端延迟</div><div class="v" id="sLat">–</div></div>
  <div class="cell"><div class="k">缓冲帧</div><div class="v" id="sBuf">–</div></div>
  <div class="cell"><div class="k">卡顿次数</div><div class="v" id="sStall">0</div></div>
  <div class="cell"><div class="k">服务器显存</div><div class="v" id="sVram">–</div></div>
</div>
<div class="hint">
  只使用 CF++ 模型直播: 每段 __BLOCKS__ latent 帧从源图重锚定(画质每段重置), 段间 __FADE__ 帧溶解过渡;
  KV 窗口 21 帧 + sink 1 帧(源图锚) 抑制段内漂移。缓冲超过 3 秒自动丢旧帧追最新。
</div>
<div class="ctrl">
    <div class="ctrl-row">
      <input id="capIn" type="text" maxlength="200" placeholder="输入文本驱动动作+语音播报, 如: 向大家挥手打招呼 (下一段生效, 约20秒后)">
      <button id="micBtn" title="语音输入 (Chrome)">🎤</button>
      <button id="capBtn">发送</button>
    </div>
    <div class="ctrl-row" style="margin-top:6px; align-items:center;">
      <label style="font-size:12px; color:#8b949e; display:flex; align-items:center; gap:6px; cursor:pointer;">
        <input type="checkbox" id="ttsOn" checked> 🔊 语音播报 (edge-tts)
      </label>
      <label style="font-size:12px; color:#8b949e; display:flex; align-items:center; gap:6px; cursor:pointer;">
        <input type="checkbox" id="autoSend" checked> 语音识别后自动发送
      </label>
    </div>
    <div class="chips">
      <button class="chip" data-t="向大家挥手打招呼">挥手</button>
      <button class="chip" data-t="点头表示同意">点头</button>
      <button class="chip" data-t="点赞">点赞</button>
      <button class="chip" data-t="鼓掌">鼓掌</button>
      <button class="chip" data-t="思考一下">思考</button>
      <button class="chip" data-t="讲一段话">说话</button>
      <button class="chip" data-t="双手静止">静止</button>
    </div>
    <div id="capMsg" class="cap-msg">当前: 默认动作 (双手交握几乎静止)。发送后下一段生效 (约 20 秒)。</div>
  </div>
<script>
// ---- 文本→动作 (+ 口型同步语音播报 + 语音输入) ----
const capIn = document.getElementById('capIn'), capMsg = document.getElementById('capMsg');
const ttsOn = document.getElementById('ttsOn'), autoSend = document.getElementById('autoSend');
let curAudio = null;     // 当前播报音频 (新句顶旧句)
let pendingSpeak = null; // {url,id}: 口型句音频就绪, 等对应段帧真正显示时起播
let nextFrameSpk = false; // spk 事件后的第一个帧消息 = 说话段首帧 (SSE 顺序保证)
let pendingSpeakText = ''; // 待播报文本 (状态提示用)
const spkTag = document.getElementById('spkTag');

function stopAudio() {
  if (curAudio) { curAudio.onended = null; curAudio.pause(); curAudio.src = ''; curAudio = null; }
  spkTag.style.display = 'none';
}

function startAudio(url) {
  stopAudio();
  curAudio = new Audio(url);
  curAudio.onplay  = () => { spkTag.style.display = 'flex'; };
  curAudio.onended = () => { stopAudio(); capMsg.textContent = '🔊 播报完成'; };
  curAudio.play().catch(() => {});
}

function sendCaption(text, speak) {
  text = (text || capIn.value || '').trim();
  if (!text) return;
  const doSpeak = speak !== undefined ? speak : ttsOn.checked;   // chips 动作词默认不播报
  const url = doSpeak ? '/caption_speak' : '/caption';
  fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({text})})
    .then(r => r.json())
    .then(d => { if (d.ok) { capIn.value = '';
                 if (doSpeak) { capMsg.textContent = '⏳ 已提交: ' + text + ' — 画面切到说话段后自动播报(口型同步)';
                                pendingSpeakText = text; }
                 else capMsg.textContent = '已发送: ' + text + ' → 下一段生效 (约 20 秒后)'; }
               else capMsg.textContent = '发送失败: ' + (d.msg || ''); })
    .catch(() => { capMsg.textContent = '发送失败 (网络)'; });
}

// ---- 语音输入 (Web Speech API, Chrome/Edge) ----
const micBtn = document.getElementById('micBtn');
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recog = null;
if (SR) {
  recog = new SR();
  recog.lang = 'zh-CN'; recog.interimResults = false; recog.maxAlternatives = 1;
  recog.onresult = e => {
    const t = e.results[e.results.length-1][0].transcript.trim();
    capIn.value = t;
    if (autoSend.checked) sendCaption(t); else capMsg.textContent = '语音识别: ' + t;
  };
  recog.onend = () => { micBtn.classList.remove('rec'); };
  recog.onerror = e => { micBtn.classList.remove('rec');
                         capMsg.textContent = '语音识别失败: ' + (e.error==='not-allowed' ? '请允许麦克风权限' : e.error); };
  micBtn.onclick = () => {
    if (micBtn.classList.contains('rec')) { recog.stop(); return; }
    stopAudio();   // 停掉播报防回环: 扬声器声音被麦克风拾音会造成误识别
    try { recog.start(); micBtn.classList.add('rec');
          capMsg.textContent = '🎤 正在聆听… (说完自动识别)'; }
    catch (err) { capMsg.textContent = '识别器启动失败: ' + err; }
  };
} else {
  micBtn.onclick = () => { capMsg.textContent = '当前浏览器不支持语音识别 (请用 Chrome/Edge)'; };
}

document.getElementById('capBtn').onclick = () => sendCaption();
capIn.addEventListener('keydown', e => { if (e.key === 'Enter') sendCaption(); });
document.querySelectorAll('.chip').forEach(b => b.onclick = () => sendCaption(b.dataset.t, false));

const TARGET = __TARGET_FPS__, INIT_BUF = Math.round(__TARGET_FPS__ * 2);
const es = new EventSource('/stream');
const ctx = document.getElementById('stage').getContext('2d');
let q = [];                 // FIFO [{img, tGen, seg}]
let stallCnt = 0, stallStreak = false;
let recvT = [], showT = [], latArr = [], started = false, playT0 = null;
let lastShow = 0, serverStats = null, curSeg = -1;

es.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.t === 's') { serverStats = m; return; }
  if (m.t === 'e') { document.getElementById('stallTip').textContent = '服务器错误: ' + m.msg;
                     document.getElementById('stallTip').style.display = 'block'; return; }
  if (m.t === 'spk') { pendingSpeak = {url: m.url, id: m.id}; nextFrameSpk = true; return; }   // 口型句音频就绪
  const img = new Image();
  const spkFlag = nextFrameSpk; nextFrameSpk = false;   // 只标记紧随其后的首帧
  img.onload = () => {
    q.push({img, tGen: m.g, seg: m.s, spk: spkFlag});
    if (!started && q.length >= INIT_BUF) { started = true; playT0 = performance.now(); }
  };
  img.src = 'data:image/jpeg;base64,' + m.i;
  recvT.push(performance.now());
};
function trim(arr, now) {
  while (arr.length && now - arr[0] > 3000) arr.shift();
}
function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  // 缓冲 >3s: 丢旧帧追最新 (说话段首帧若被丢, 把起播标记带到下一帧防播报丢失)
  while (q.length > TARGET * 3) {
    const d = q.shift();
    if (d.spk && q.length) q[0].spk = true;
  }
  if (started && now - lastShow >= 1000 / TARGET) {
    if (q.length) {
      const fr = q.shift();
      ctx.drawImage(fr.img, 0, 0, 480, 832);
      // 说话段首帧显示瞬间起播音频 → 音画+口型对齐 (2s 缓冲延迟被自然吸收)
      if (fr.spk && pendingSpeak) {
        startAudio(pendingSpeak.url);
        capMsg.textContent = '🔊 播报中: ' + pendingSpeakText;
        pendingSpeak = null;
      }
      if (fr.seg !== curSeg) {
        curSeg = fr.seg;
        document.getElementById('segTag').textContent = 'SEG ' + (curSeg + 1);
      }
      lastShow = now;
      showT.push(now);
      // latArr 存 {t: 时间戳, v: 延迟ms}; trim 按 t 清理, 不能直接存延迟值(会被当时间戳误杀清空)
      latArr.push({t: now, v: Date.now() - fr.tGen * 1000});
      if (stallStreak) stallStreak = false;
    } else {
      // 节拍到点但无帧: 冻结画面 = 卡顿
      if (!stallStreak) { stallCnt++; stallStreak = true; }
    }
  }
  document.getElementById('stallTip').style.display = stallStreak ? 'block' : 'none';
  trim(recvT, now); trim(showT, now); trim(latArr, now);
  const recv = recvT.length / 3, play = showT.length / 3;
  const lats = latArr.filter(e => isFinite(e.v) && e.v > 0).map(e => e.v);
  const lat = lats.length ? lats.reduce((a,b)=>a+b,0) / lats.length / 1000 : 0;
  const cls = v => v >= 15 ? 'good' : (v >= 12 ? 'warn' : 'bad');
  document.getElementById('sPlay').textContent = play.toFixed(1);
  document.getElementById('sPlay').className = 'v ' + cls(play);
  document.getElementById('sLat').textContent = lats.length ? lat.toFixed(2) + 's' : '–';
  document.getElementById('sLat').className = 'v ' + (lats.length && lat < 2 ? 'good' : 'bad');
  document.getElementById('sBuf').textContent = q.length;
  document.getElementById('sStall').textContent = stallCnt;
  document.getElementById('sStall').className = 'v ' + (stallCnt === 0 ? 'good' : (stallCnt < 5 ? 'warn' : 'bad'));
  if (serverStats) {
    document.getElementById('sSupply').textContent = serverStats.sp.toFixed(1);
    document.getElementById('sSupply').className = 'v ' + cls(serverStats.sp);
    document.getElementById('sVram').textContent = serverStats.vm.toFixed(1) + 'GB';
  }
  // 初始缓冲提示
  if (!started) {
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,480,832);
    ctx.fillStyle = '#8b949e'; ctx.font = '16px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('初始缓冲中… (' + q.length + '/' + INIT_BUF + ')', 240, 420);
  }
}
loop();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return (HTML.replace("__TARGET_FPS__", str(TARGET_FPS))
                .replace("__MULT__", str(MULT))
                .replace("__BLOCKS__", str(BLOCKS_PER_SEG))
                .replace("__FADE__", str(FADE)))

@app.post("/caption")
def set_caption():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))[:200]
    cap = build_caption(text)
    with caption_state["lock"]:
        caption_state["text"] = text
        caption_state["caption"] = cap
    print(f"[caption] text={text!r} -> {cap[:80]}...")
    return jsonify({"ok": True, "caption": cap, "text": text})

@app.get("/caption")
def get_caption():
    with caption_state["lock"]:
        return jsonify({"text": caption_state["text"], "caption": caption_state["caption"]})

# ---------------- TTS 语音播报 (edge-tts, CPU, 与直播渲染零 GPU 争抢) ----------------
import hashlib, asyncio
try:
    import edge_tts
    # Windows 默认 ProactorEventLoop 与 aiohttp 清理不兼容, 间歇性报 "Event loop is closed";
    # 换 Selector 策略是 edge-tts 官方推荐修法 (仅影响本进程新建的事件循环)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    TTS_VOICE = "zh-CN-XiaoxiaoNeural"   # 自然中文女声
    TTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
    os.makedirs(TTS_DIR, exist_ok=True)
    _tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # 串行合成, 防并发打满 CPU
except ImportError:
    edge_tts = None
    print("[tts] edge-tts 未安装 (pip install edge-tts), 语音播报不可用", flush=True)

def _tts_sync(text, path):
    """先写 .part 临时文件再原子替换: 网络中断不留半截 mp3 污染缓存"""
    tmp = path + ".part"
    async def _run():
        cm = edge_tts.Communicate(text, TTS_VOICE)
        await cm.save(tmp)
    try:
        asyncio.run(_run())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
        raise

@app.post("/speak")
def speak():
    """文本 → edge-tts 合成 mp3 (带缓存); 返回 /audio/<file> 供前端 <audio> 播放"""
    if edge_tts is None:
        return jsonify({"ok": False, "msg": "server: edge-tts 未安装"}), 503
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()[:300]
    if not text:
        return jsonify({"ok": False, "msg": "empty"}), 400
    key = hashlib.md5((TTS_VOICE + "|" + text).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(TTS_DIR, key + ".mp3")
    if not os.path.exists(path):                    # 缓存命中直接返回
        try:
            _tts_pool.submit(_tts_sync, text, path).result(timeout=60)
        except Exception as e:
            return jsonify({"ok": False, "msg": f"合成失败: {type(e).__name__}"}), 502
    return jsonify({"ok": True, "url": f"/audio/{key}.mp3"})

@app.get("/audio/<name>")
def audio(name):
    safe = os.path.basename(name)
    p = os.path.join(TTS_DIR, safe)
    if not os.path.exists(p):
        return jsonify({"ok": False}), 404
    return send_file(p, mimetype="audio/mpeg")

@app.post("/caption_speak")
def caption_speak():
    """说话内容一站式: 文本→动作 caption(下一段生效) + TTS + whisper 特征预计算 + 注册口型句。
    帧偏移按段首帧 t_gen 计, 前端在该段画面出现时播音频 → 句级音画+口型对齐。"""
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()[:200]
    if not text:
        return jsonify({"ok": False, "msg": "empty"}), 400
    # 1) 动作 caption (与 /caption 同逻辑)
    cap = build_caption(text)
    with caption_state["lock"]:
        caption_state["text"] = text
        caption_state["caption"] = cap
    # 2) TTS 合成 (带缓存)
    if edge_tts is None:
        return jsonify({"ok": False, "msg": "server: edge-tts 未安装"}), 503
    key = hashlib.md5((TTS_VOICE + "|" + text).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(TTS_DIR, key + ".mp3")
    if not os.path.exists(path):
        try:
            _tts_pool.submit(_tts_sync, text, path).result(timeout=60)
        except Exception as e:
            return jsonify({"ok": False, "msg": f"合成失败: {type(e).__name__}"}), 502
    # 3) 口型特征预计算 (lipsync_gpu, ~百ms)
    if lipsync is not None:
        try:
            from lipsync_musetalk import mp3_to_pcm16k
            pcm = mp3_to_pcm16k(path)
            feat = lipsync.extract_feature(pcm)
            with speech_state["lock"]:
                speech_state["id"] += 1
                speech_state["feat"] = feat
                speech_state["t0"] = None          # 等说话段首帧登记
                speech_state["dur"] = len(pcm) / 16000.0
                speech_state["url"] = f"/audio/{key}.mp3"
                speech_state["seg"] = None
            print(f"[speak] '{text[:30]}' 特征就绪 {len(pcm)/16000.0:.1f}s, 等待口型段", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"ok": False, "msg": f"特征提取失败: {type(e).__name__}"}), 502
    return jsonify({"ok": True, "caption": cap, "text": text, "url": f"/audio/{key}.mp3",
                    "lipsync": lipsync is not None})

@app.route("/stream")
def stream():
    def gen():
        # 新连接: 丢弃积压, 只保留最新 ~2s, 避免观众接入时先看长历史
        drop = out_q.qsize() - int(TARGET_FPS * 2)
        for _ in range(max(0, drop)):
            try: out_q.get_nowait()
            except queue.Empty: break
        last_stat = 0.0
        # 节拍 + drain-to-latest: 每拍把队列排干只取最新帧。
        # RIFE 输出(3.5源帧x8=28fps) > 播放节拍(15fps), 若从队头逐帧拿会拿到积压旧帧, 延迟随观看时间无界增长;
        # drain-to-latest 使观众永远看到"当前最新画面", 延迟恒定≈前端缓冲(2s)+编码传输, 多余插帧帧被丢弃(视觉等价)
        frame_dt = 1.0 / TARGET_FPS
        dt = frame_dt
        next_beat = time.time()
        last_yield = None   # 上一帧 payload, 用于帧脉冲间隙重发保持恒定节拍
        spk_seg_seen = None   # 已检查过口型事件的段 (每段查一次)
        spk_sent = set()      # 已下发的句 id (本连接去重)
        while True:
            lag = next_beat - time.time()
            if lag > 0:
                time.sleep(lag)
            # 自适应节拍: 队列空→放宽拍间隔(减少空拍重发), 队列积压→加速消化(防延迟累积)。
            # 让节拍跟随供给自然波动, 每拍必中有新帧, 消除固定节拍下空拍/跳帧交替的顿挫
            q = out_q.qsize()
            if q == 0:
                dt = min(dt * 1.06, 0.070)
            elif q > 2:
                dt = max(dt * 0.94, 0.045)
            else:
                dt += (frame_dt - dt) * 0.3   # 回归名义节拍
            next_beat = max(next_beat + dt, time.time())
            # 软节流: 平时每拍只取 1 帧(高节拍下不丢插帧帧); 仅当积压>8帧(0.4s)才排到4帧防延迟累积。
            # 旧 drain-to-latest 在 RIFE 突发(4帧/160ms)下每拍丢3帧, 20fps 节拍实际只播到 17fps
            if out_q.qsize() > 8:
                while out_q.qsize() > 4:
                    try: out_q.get_nowait()
                    except queue.Empty: break
            latest = None
            try:
                kind, payload = out_q.get_nowait()
                if kind == "err":
                    yield f"data: {json.dumps({'t':'e','msg': stats.get('err') or 'unknown'})}\n\n"
                    return
                latest = payload
            except queue.Empty:
                pass
            if latest is None:
                # 本拍无新帧(帧脉冲间隙): 重发上一帧保持恒定节拍 (66ms 冻结肉眼无感),
                # 否则前端缺帧导致卡顿计数累积
                if last_yield is not None:
                    f, t_gen, seg = last_yield
                    rgb = (f.permute(1,2,0).numpy() * 255).astype(np.uint8)
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
                    b64 = base64.b64encode(buf.tobytes()).decode()
                    yield f"data: {json.dumps({'t':'f','g': round(t_gen,3),'s': seg,'i': b64})}\n\n"
                else:
                    now = time.time()
                    if now - last_stat > 1.0:
                        last_stat = now
                        yield f"data: {json.dumps({'t':'s','sp': supply_fps(),'vm': vram_gb()})}\n\n"
                continue
            f, t_gen, seg = latest
            last_yield = latest
            now = time.time()
            if now - last_stat > 1.0:
                last_stat = now
                yield f"data: {json.dumps({'t':'s','sp': supply_fps(),'vm': vram_gb()})}\n\n"
            # 口型事件: 段首到达前端时下发音频 url (前端在该段画面显示瞬间播放 → 音画+口型对齐)
            if seg != spk_seg_seen:
                spk_seg_seen = seg
                with speech_state["lock"]:
                    sseg, surl, sid = speech_state["seg"], speech_state["url"], speech_state["id"]
                if sid not in spk_sent and sseg is not None and seg >= sseg and surl:
                    spk_sent.add(sid)
                    if len(spk_sent) > 32: spk_sent.discard(min(spk_sent))
                    yield f"data: {json.dumps({'t':'spk','url': surl,'id': sid})}\n\n"
            rgb = (f.permute(1,2,0).numpy() * 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
            b64 = base64.b64encode(buf.tobytes()).decode()
            yield f"data: {json.dumps({'t':'f','g': round(t_gen,3),'s': seg,'i': b64})}\n\n"
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

def supply_fps():
    """最近 ~10s 滑动窗口的像素供给 fps"""
    with stats["lock"]:
        samples = stats["samples"]
    if len(samples) < 2: return 0.0
    t1, c1 = samples[0]
    t2, c2 = samples[-1]
    if t2 - t1 < 1.0: return 0.0
    return (c2 - c1) / (t2 - t1)

# ---------------- 启动 ----------------
threads = [threading.Thread(target=generator_thread)]
if LIPSYNC_ACTIVE:
    threads.append(threading.Thread(target=lipsync_thread))
threads.append(threading.Thread(target=rife_thread))
mon = threading.Thread(target=monitor_thread, daemon=True)
for t in threads: t.start()
mon.start()
print(f"\n[live] 服务启动: http://127.0.0.1:{args.port}  口型={'开' if LIPSYNC_ACTIVE else '关'}  (Ctrl+C 停止)\n", flush=True)
app.run(host="127.0.0.1", port=args.port, threaded=True)
