# MuseTalk V15 最小实时口型核心 (跑在 3050/cuda:1, fp16)
# 数据流: CF++ 源帧(RGB float 0..1, CHW) → Haar人脸框 → 扩展裁剪→256² → VAE双编码(遮罩+净图)
#        → UNet单步(t=0, cross-attn=whisper chunk) → VAE解码 → 回缩 → 椭圆羽化贴回
# 音频:   ffmpeg 解码 mp3 → 16k PCM → whisper-tiny mel → encoder 全33层 hidden states
#        → 每源帧按 音频偏移秒 取10帧窗口 → [1,330,384] 位置编码
import os, sys, math, json, subprocess
import numpy as np
import torch
import cv2

MT_SRC = r"d:\FasterLivePortrait\causal_forcing_poc\MuseTalk_src\MuseTalk-main"
MT_MODELS = r"d:\FasterLivePortrait\causal_forcing_poc\musetalk_models"
FFMPEG = "ffmpeg"

def mp3_to_pcm16k(mp3_path):
    """mp3 → 16k 单声道 float32 PCM (ffmpeg CLI, 无新 python 依赖)"""
    raw = subprocess.run([FFMPEG, "-i", mp3_path, "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-loglevel", "error", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

class PositionalEncoding(torch.nn.Module):
    """与官方 realtime_inference.py 一致 (d_model=384)"""
    def __init__(self, d_model=384, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.dtype)

class MuseTalkLipSync:
    CROP = 256          # MuseTalk 工作分辨率
    WIN = 10            # 每源帧 whisper 窗口 (10帧@50Hz=200ms, 官方 audio_padding 2+2)
    EXPAND = 1.5        # 裁剪框相对人脸框扩展 (官方 get_crop_box expand)

    def __init__(self, device="cuda:1", use_fp16=True):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import WhisperModel, WhisperFeatureExtractor
        self.device = torch.device(device)
        self.dtype = torch.float16 if use_fp16 else torch.float32
        wt = os.path.join(MT_MODELS, "musetalkV15", "unet.pth")
        wj = os.path.join(MT_MODELS, "musetalkV15.json")
        cfg = json.load(open(wj))
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}   # 滤掉 _class_name 等 diffusers 元数据
        self.unet = UNet2DConditionModel(**cfg)
        sd = torch.load(wt, map_location="cpu", weights_only=True)
        self.unet.load_state_dict(sd)
        self.unet.to(device=self.device, dtype=self.dtype).eval()
        self.vae = AutoencoderKL.from_pretrained(os.path.join(MT_MODELS, "sd-vae-ft-mse")
                        ).to(device=self.device, dtype=self.dtype).eval()
        self.scaling = self.vae.config.scaling_factor
        wdir = os.path.join(MT_MODELS, "whisper-tiny")
        self.whisper = WhisperModel.from_pretrained(wdir).to(device=self.device, dtype=self.dtype).eval()
        self.feat_ext = WhisperFeatureExtractor.from_pretrained(wdir)
        self.pe = PositionalEncoding(384).to(self.device, self.dtype)
        self.timesteps = torch.zeros(1, device=self.device, dtype=torch.long)
        # UNet 编译加速 (default 模式不用 cudagraph, 不干扰 RIFE); VAE enc/dec 同样编译
        # (静态 shape; 实测 _bench_opt.py: enc 69→48ms, dec 73→43ms)
        if os.environ.get("LIPSYNC_PLAIN") != "1":
            try:
                self.unet = torch.compile(self.unet)
                self.vae.encoder = torch.compile(self.vae.encoder)
                self.vae.decoder = torch.compile(self.vae.decoder)
                print("[lipsync] unet + vae enc/dec torch.compile enabled", flush=True)
            except Exception as e:
                print(f"[lipsync] compile 不可用: {type(e).__name__}", flush=True)
        # 解码带: 只解 latent 行 8:32 (口型椭圆上缘典型 ~109px@256 空间, 带上缘 64px+羽化 18px 余量足)
        self.DEC_R0 = 8
        # 半脸遮罩 (上1下0): 遮罩输入 latent 告诉 UNet 上半部分保留原样 (官方 mask_tensor 同款)
        m = torch.zeros(self.CROP, self.CROP); m[:self.CROP // 2, :] = 1
        self.half_mask = m.to(self.device, self.dtype)
        # Haar 人脸检测 (固定数字人, 每30帧检一次 + EMA 平滑)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._box_ema = None
        self._frame_no = 0
        self._mask_cache = {}   # 椭圆羽化遮罩缓存 (key=量化椭圆参数): 实测每帧省 ~1.2ms
        print(f"[lipsync] MuseTalk V15 loaded on {device} fp16={use_fp16}", flush=True)

    # ---------- 音频特征 (每句一次, GPU ~百ms级) ----------
    @torch.no_grad()
    def extract_feature(self, pcm16k):
        """PCM → whisper 全层 hidden states [1, T50, 33, 384] fp16 (GPU 常驻备用)"""
        mel = self.feat_ext(pcm16k, sampling_rate=16000, return_tensors="pt").input_features
        mel = mel.to(self.device, self.dtype)
        hs = self.whisper.encoder(mel, output_hidden_states=True).hidden_states  # (1+层数) × [1,T50,384]
        feat = torch.stack(hs, dim=2)                                            # [1,T50,1+层数,384]
        return feat

    def chunk_at(self, feat, offset_sec):
        """音频偏移秒 → 该源帧的 cross-attn chunk [1, 10×(层数+1), 384] (tiny: 50, 边界零填充防越界)"""
        if getattr(self, "_pad_id", None) != id(feat):   # 每句特征只拼一次填充 (帧级热点路径免 cat)
            pad = torch.zeros_like(feat[:, :self.WIN])
            self._pad_feat = torch.cat([pad, feat, pad, pad], dim=1)
            self._pad_id = id(feat)
        f = self._pad_feat
        idx = int(offset_sec * 50)
        idx = max(0, min(idx, f.size(1) - self.WIN))
        clip = f[:, idx: idx + self.WIN]                     # [1,10,33,384]
        b, c, h, w = clip.shape
        return clip.reshape(b, c * h, w)                     # [1,330,384] (官方 rearrange 等价)

    # ---------- 人脸框 ----------
    def _face_box(self, frame_rgb_u8):
        self._frame_no += 1
        gray = cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2GRAY)
        if self._frame_no % 30 == 1 or self._box_ema is None:
            faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                self._box_ema = box if self._box_ema is None else 0.4 * box + 0.6 * self._box_ema
        return self._box_ema

    # ---------- 逐帧口型 ----------
    @torch.no_grad()
    def process_frame(self, frame_chw, feat=None, offset=0.0):
        """CF++ 源帧 [3,H,W] float 0..1 (RGB) → 口型重绘帧; 无人脸/无特征时原样返回"""
        f = frame_chw
        if feat is None:
            return f
        if f.dim() == 3 and f.shape[0] == 3:
            img = (f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)   # HWC RGB
        else:
            return f
        box = self._face_box(img)
        if box is None:
            return f
        H, W = img.shape[:2]
        x, y, x1, y1 = [int(v) for v in box]
        cx, cy = (x + x1) // 2, (y + y1) // 2
        s = int(max(x1 - x, y1 - y) // 2 * self.EXPAND)
        xs, ys, xe, ye = cx - s, cy - s, cx + s, cy + s
        # 裁剪区映射回画面 (越界裁剪后在贴回时对齐)
        xs_c, ys_c = max(0, xs), max(0, ys)
        xe_c, ye_c = min(W, xe), min(H, ye)
        crop = img[ys_c:ye_c, xs_c:xe_c]
        if crop.size == 0:
            return f
        crop_r = cv2.resize(crop, (self.CROP, self.CROP), interpolation=cv2.INTER_LANCZOS4)
        # --- VAE 双编码 + UNet 单步 ---
        t256 = torch.from_numpy(crop_r.astype(np.float32) / 255.0
                ).permute(2, 0, 1)[None].to(self.device, self.dtype) * 2.0 - 1.0   # [1,3,256,256]
        masked = t256 * (1.0 - self.half_mask)[None, None]     # 上半清零 → 待重绘区
        # 编码必须用原 VAE (TAESD encoder latent 偏差会让 UNet 输出严重伪影, 已实测);
        # 两次编码合并 batch=2 一次前向
        lat = torch.cat(self._enc(torch.cat([masked, t256], dim=0)).chunk(2), dim=1)  # [1,8,32,32]
        chunk = self.chunk_at(feat, offset)                                       # [1,50,384]
        pred = self.unet(lat, self.timesteps, encoder_hidden_states=self.pe(chunk)).sample
        # 只解码口型带 (latent 行 8:32): 全图解码是最大瓶颈, 裁剪+编译 73→~51ms
        dec = self.vae.decode(pred[:, :, self.DEC_R0:, :] / self.scaling).sample
        img_dec = (dec / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        # 低频亮度校准: 解码带相对原图有 +1~5/255 低/中频色偏 (half_mask 边界附近尤甚, 眼下横带;
        # 人眼对平滑肤色 1-2% 亮度差极敏感, 并排实测原图无此带)。sigma10 低通差限幅校正:
        # 覆盖横带所在的 8-15px 中频段, 只修色偏 — 唇纹等细节 (<10px) 与嘴形动态不受影响
        src_band_f = crop_r[self.DEC_R0 * 8:].astype(np.float32) / 255.0
        bd = cv2.GaussianBlur(img_dec, (0, 0), 10)
        bs = cv2.GaussianBlur(src_band_f, (0, 0), 10)
        img_dec = np.clip(img_dec + np.clip(bs - bd, -4.0 / 255, 4.0 / 255), 0, 1)
        # --- 回缩 + 椭圆羽化贴回 (只贴解码带) ---
        # 椭圆参数必须换算到 256 工作空间: 裁剪区 !=256px 时 (实测人脸框137→裁剪204px, scale1.255)
        # 原图尺度参数直画 256 会偏上 ~24px 且偏小 25%, 罩住鼻区重建 → 口部上方可见色差 (_diag_color.py)
        scx = self.CROP / (xe_c - xs_c)
        scy = self.CROP / (ye_c - ys_c)
        fx, fy = (x - xs_c) * scx, (y - ys_c) * scy       # 人脸框在 256 空间坐标
        fw, fh = (x1 - x) * scx, (y1 - y) * scy
        ax, ay = int(fw * 0.52), int(fh * 0.17)           # 椭圆半轴: 上缘 0.55fh=鼻翼 (UNet 遮罩重绘区眼眶/鼻梁
                                                          #  偏差大必须避开; 下缘 0.89fh 覆盖张嘴下唇)
        ccx, ccy = int(fx + fw / 2), int(fy + fh * 0.72)
        band_px = self.CROP * self.DEC_R0 // 32           # 256 空间解码带起点行 (64)
        key = (ccx // 2, ccy // 2, ax // 2, ay // 2)      # 2px 量化: 人脸框 EMA 微动也命中缓存
        mask = self._mask_cache.get(key)
        if mask is None:
            m = np.zeros((self.CROP, self.CROP), np.float32)
            cv2.ellipse(m, (ccx, ccy), (ax, ay), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(m, (0, 0), 6)[band_px:, :][..., None]   # 羽化 6: 3σ=18px, 重绘偏差带(<135行)混入≤20%
            if len(self._mask_cache) > 16:
                self._mask_cache.clear()
            self._mask_cache[key] = mask
        bh = ye_c - ys_c
        y0 = ys_c + int(round(bh * band_px / self.CROP))  # 解码带在原图的起始行
        band_h = ye_c - y0
        out_band = cv2.resize((img_dec * 255).astype(np.uint8), (xe_c - xs_c, band_h),
                              interpolation=cv2.INTER_CUBIC)
        m_band = cv2.resize(mask[..., 0], (xe_c - xs_c, band_h),
                            interpolation=cv2.INTER_LINEAR)[..., None]
        blend = crop[y0 - ys_c:].astype(np.float32) * (1 - m_band) + out_band.astype(np.float32) * m_band
        img[y0:ye_c, xs_c:xe_c] = np.clip(blend, 0, 255).astype(np.uint8)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(f.device, f.dtype)

    def _enc(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.scaling

    def _dec(self, lat):
        # 只用原 VAE 解码: TAESD 嘴周色块不可接受 (直播嘴部是视线焦点), 实测对照见 lipsync_bench/
        img = self.vae.decode(lat / self.scaling).sample
        return (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
