# =============================================================================
#  FILE INI KHUSUS UNTUK SCREENSHOT SKRIPSI — JANGAN DIEDIT
#  Buka di VS Code, zoom secukupnya, lalu SS tiap bagian
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  4.3.1  IMPLEMENTASI PREPROCESSING DATA
# ─────────────────────────────────────────────────────────────────────────────

# [SS 1] Ekstraksi 16 Frame dengan Margin 15%
def extract_frames(video_path, num_frames, img_size=(224, 224),
                   is_train=False, margin=0.15):
    cap = cv2.VideoCapture(video_path)
    total       = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    start_frame = int(total * margin)
    end_frame   = int(total * (1.0 - margin))
    valid_total = max(end_frame - start_frame, 1)
    seg_size    = valid_total / float(num_frames)

    indices = []
    for i in range(num_frames):
        s = int(start_frame + i * seg_size)
        e = int(start_frame + (i + 1) * seg_size)
        if is_train and e > s:
            idx = np.random.randint(s, e)   # jitter: acak dalam segmen
        else:
            idx = s + (e - s) // 2          # center: titik tengah segmen
        indices.append(min(idx, total - 1))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, img_size))
    cap.release()
    return frames[:num_frames]


# [SS 2] Segmentasi Background (MediaPipe Selfie Segmentation)
def center_crop_square(bgr):
    h, w = bgr.shape[:2]
    s  = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return bgr[y0:y0+s, x0:x0+s]

def segment_frame(bgr, mode='blur'):
    bgr  = center_crop_square(bgr)         # seragamkan proporsi portrait/landscape
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = selfie_segmentation.process(rgb).segmentation_mask
    binm = cv2.dilate((mask > 0.3).astype(np.uint8), dilate_kernel, iterations=1)
    mask = np.clip(np.maximum(mask, binm.astype(np.float32) * 0.8), 0.0, 1.0)
    return composite(bgr, mask, mode)      # background diganti blur (kernel 41×41)


# [SS 3] Normalisasi ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# diterapkan ke setiap frame setelah augmentasi:
tensor = TF.to_tensor(pil_image)
tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


# ─────────────────────────────────────────────────────────────────────────────
#  4.3.2  IMPLEMENTASI AUGMENTASI DATA
# ─────────────────────────────────────────────────────────────────────────────

# [SS 4] Augmentasi Konsisten Per-Video (nilai dipilih sekali, dipakai 16 frame)
class ConsistentVideoAugment:
    def __call__(self, frames_bgr):
        if self.train:
            # Spasial (sama untuk semua frame → makna isyarat tidak berubah)
            angle     = random.uniform(-7, 7)
            translate = (random.uniform(-0.07, 0.07) * W,
                         random.uniform(-0.07, 0.07) * W)
            scale     = random.uniform(0.88, 1.12)

            # Fotometrik — diperlebar v10 agar robust terhadap variasi kamera
            b = random.uniform(0.6, 1.4)     # brightness  (v9: 0.8-1.2)
            c = random.uniform(0.6, 1.4)     # contrast    (v9: 0.8-1.2)
            s = random.uniform(0.8, 1.2)     # saturation  (v9: 0.9-1.1)
            h = random.uniform(-0.08, 0.08)  # hue         (v9: ±0.05)
            g = random.uniform(0.7, 1.5)     # gamma jitter — BARU di v10

        out = []
        for pil in frames_pil:
            if self.train:
                pil = TF.affine(pil, angle=angle,
                                translate=list(translate), scale=scale, shear=[0.0])
                pil = TF.adjust_brightness(pil, b)
                pil = TF.adjust_contrast(pil, c)
                pil = TF.adjust_saturation(pil, s)
                pil = TF.adjust_hue(pil, h)
                pil = TF.adjust_gamma(pil, g)   # simulasi respons cahaya kamera nyata
            tensor = TF.normalize(TF.to_tensor(pil), IMAGENET_MEAN, IMAGENET_STD)
            out.append(tensor)
        return torch.stack(out)   # (T=16, 3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
#  4.3.3  IMPLEMENTASI ARSITEKTUR MODEL
# ─────────────────────────────────────────────────────────────────────────────

# [SS 5] Temporal Shift Module (TSM)
class TemporalShift(nn.Module):
    def __init__(self, n_segment=16, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = n_div            # 1/8 channel digeser maju, 1/8 mundur

    def forward(self, x):
        BT, C, H, W = x.size()
        B    = BT // self.n_segment
        x    = x.view(B, self.n_segment, C, H, W)
        fold = C // self.fold_div
        out  = torch.zeros_like(x)
        out[:, 1:,    :fold]     = x[:, :-1, :fold]      # geser maju  (frame t-1 → t)
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold] # geser mundur (frame t+1 → t)
        out[:, :,   2*fold:]     = x[:, :,   2*fold:]     # sisanya tetap
        return out.view(BT, C, H, W)

def inject_tsm(model, n_segment=16, n_div=8):
    for block in model.features:
        if isinstance(block, InvertedResidual):
            orig = block.conv[0]
            block.conv[0] = nn.Sequential(TemporalShift(n_segment, n_div), orig)
    return model


# [SS 6] CBAM (Convolutional Block Attention Module)
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(),
                                  nn.Linear(ch//r, ch, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        a = self.fc(self.avg(x).view(B, C))
        m = self.fc(self.mx(x).view(B, C))
        return x * self.sig(a + m).view(B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.channel_att = ChannelAttention(ch, r)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# [SS 7] Arsitektur Lengkap BISINDOClassifier + forward()
class BISINDOClassifier(nn.Module):
    def __init__(self, num_classes=24, num_frames=16, dropout=0.5):
        super().__init__()
        self.num_frames = num_frames
        backbone        = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        backbone        = inject_tsm(backbone, n_segment=num_frames, n_div=8)
        self.features   = backbone.features          # MobileNetV2 + TSM
        feat_dim        = backbone.last_channel       # 1280

        self.cbam       = CBAM(feat_dim)             # Channel + Spatial Attention
        self.pool       = nn.AdaptiveAvgPool2d(1)    # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))        # 1280 → 24 kelas

    def forward(self, x):          # x: (B, T=16, 3, 224, 224)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)         # gabung batch & waktu
        x = self.features(x)               # MobileNetV2 + TSM → (B*T, 1280, 7, 7)
        x = self.cbam(x)                   # CBAM attention
        x = self.pool(x).view(B, T, -1)   # GAP → (B, T, 1280)
        x = x.mean(dim=1)                  # temporal mean pooling → (B, 1280)
        return self.classifier(x)          # → (B, 24)


# ─────────────────────────────────────────────────────────────────────────────
#  4.3.4  IMPLEMENTASI PELATIHAN MODEL
# ─────────────────────────────────────────────────────────────────────────────

# [SS 8] Konfigurasi Hyperparameter
NUM_CLASSES         = 24
NUM_FRAMES          = 16
BATCH_SIZE          = 8
EPOCH_STAGE1        = 12
EPOCH_STAGE2        = 45
LR_STAGE1           = 1e-3
LR_STAGE2           = 1e-4
WEIGHT_DECAY        = 1e-3
EARLY_STOP_PATIENCE = 7
LABEL_SMOOTHING     = 0.1
DROPOUT             = 0.5

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


# [SS 9] Phase 1 — Frozen Backbone
def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False

freeze_backbone(model)
opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min',
                   factor=0.5, patience=3)
es1  = EarlyStopping(patience=EARLY_STOP_PATIENCE, path='best_stage1_v10.pth')
# Latih 12 epoch — hanya classifier + CBAM yang diperbarui


# [SS 10] Phase 2 — Partial Unfreeze (Layer ≥ 14)
def unfreeze_partial(model, from_layer=14):
    for p in model.features.parameters():
        p.requires_grad = False
    for layer in list(model.features.children())[from_layer:]:
        for p in layer.parameters():
            p.requires_grad = True

unfreeze_partial(model, from_layer=14)
opt2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=LR_STAGE2, weight_decay=WEIGHT_DECAY * 5)
sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min',
                   factor=0.5, patience=3)
es2  = EarlyStopping(patience=EARLY_STOP_PATIENCE, path='best_stage2_v10.pth')
# Latih maks 45 epoch — layer 14+ + classifier + CBAM diperbarui


# [SS 11] Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, path='best.pth'):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.best_loss = float('inf')
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)   # simpan bobot terbaik
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                model.load_state_dict(torch.load(self.path))  # muat bobot terbaik


# ─────────────────────────────────────────────────────────────────────────────
#  4.3.5  IMPLEMENTASI SERVER INFERENSI
# ─────────────────────────────────────────────────────────────────────────────

# [SS 12] State Machine Client (prepare → collect → hold)
class ClientState:
    def __init__(self):
        self.state       = "prepare"
        self.phase_start = time.time()
        self.frames      = []       # frame BGR 224×224 tersegmentasi
        self.hand_frames = 0        # frame yang terdeteksi ada tangan

PREPARE_SEC = 2.0   # detik jeda sebelum rekam
COLLECT_SEC = 5.0   # durasi rekam gestur
HOLD_SEC    = 3.0   # durasi tampil hasil


# [SS 13] Endpoint /predict — Siklus State Machine
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Decode JPEG dari base64
    jpeg = base64.b64decode(body["frame"])
    bgr  = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    # 2. Segmentasi background per frame
    seg, coverage = segment_frame(bgr, mode='blur')
    n_hands       = count_hands(bgr)
    pose_ok       = coverage > 0.04

    # 3. State machine per client
    if cs.state == "prepare":
        if elapsed >= PREPARE_SEC:
            cs.state = "collect"             # mulai kumpulkan frame

    elif cs.state == "collect":
        cs.frames.append(cv2.resize(seg, (224, 224)))
        if elapsed >= COLLECT_SEC:
            label, conf, top3 = predict_clip(cs.frames)  # inferensi
            cs.state = "hold"

    else:  # hold — tampilkan hasil selama HOLD_SEC
        if elapsed >= HOLD_SEC:
            cs.state = "prepare"             # mulai siklus baru


# [SS 14] Inferensi Clip (trim + 16 frame + model)
def predict_clip(frames224):
    src, peak = trim_active(frames224)      # potong frame idle di awal/akhir
    if peak < 0.5:
        return "Tidak terdeteksi", 0.0, []

    f16 = sample_clip_eval(src, 16, margin=0.15)   # ambil 16 frame representatif
    x   = frames_to_tensor(f16, device)             # → tensor (1, 16, 3, 224, 224)

    with torch.inference_mode():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()

    order = probs.argsort()[::-1][:3]
    label = CLASSES[order[0]]
    conf  = float(probs[order[0]]) * 100.0

    if conf < 22.0:                         # threshold kepercayaan minimum
        return "Belum dikenali", conf, []
    return label, conf, [[CLASSES[i], round(float(probs[i]), 4)] for i in order]


# ─────────────────────────────────────────────────────────────────────────────
#  4.3.6  IMPLEMENTASI APLIKASI MOBILE (Flutter/Dart)
# ─────────────────────────────────────────────────────────────────────────────

# [SS 15] Konversi Frame Kamera YUV→RGB + Rotasi + Flip  (detection_service.dart)
"""
Uint8List? _convertDataToJpeg(_CameraData data) {
  // YUV420 → RGB (konversi BT.601)
  img.Image rgbImage = _yuv420DataToImage(data);

  // Rotasi agar orang tampak tegak (sensor Android landscape)
  img.Image upright = rgbImage;
  final int rot = ((data.sensorOrientation % 360) + 360) % 360;
  if (rot == 90)       upright = img.copyRotate(rgbImage, angle: 90);
  else if (rot == 270) upright = img.copyRotate(rgbImage, angle: 270);

  // Flip horizontal — kamera depan Android ter-mirror secara default
  if (data.isFront) upright = img.flipHorizontal(upright);

  return Uint8List.fromList(img.encodeJpg(upright, quality: 90));
}
"""


# [SS 16] Kirim Frame ke Server via HTTP POST  (detection_service.dart)
"""
Future<Map<String, dynamic>> runInference(CameraImage image, ...) async {
  // Konversi frame kamera → JPEG di isolate terpisah (tidak block UI)
  final jpegBytes   = await compute(_convertDataToJpeg, data);
  final base64Image = base64Encode(jpegBytes);

  // POST ke Flask server
  final response = await http.post(
    Uri.parse('$apiBaseUrl/predict'),
    headers: {'Content-Type': 'application/json'},
    body: json.encode({'frame': base64Image, 'client_id': 'flutter_mobile'}),
  ).timeout(const Duration(seconds: 12));

  // Parse respons: phase, remaining, label, confidence, top3
  final j = json.decode(response.body);
  return {
    'phase'     : j['phase'],
    'label'     : j['label'],
    'confidence': j['confidence'] / 100.0,
    'remaining' : j['remaining'],
    'poseOk'    : j['pose_ok'],
    'hands'     : j['hands'],
  };
}
"""


# [SS 17] Cooldown 120ms + Proses Frame Masuk  (detection_provider.dart)
"""
final Duration _detectionCooldown = const Duration(milliseconds: 120);

void _processCameraImage(CameraImage image) {
  final now = DateTime.now();
  // Lewati frame jika masih dalam cooldown (hindari request berlebih)
  if (now.difference(_lastDetectionTime) < _detectionCooldown) return;
  if (!_isModelReady || _isDetecting) return;

  _runDetection(image);
}

Future<void> _runDetection(CameraImage image) async {
  _isDetecting       = true;
  _lastDetectionTime = DateTime.now();

  final result = await _service.runInference(
    image,
    sensorOrientation: _cameraController!.description.sensorOrientation,
    isFront: _isFrontCamera,
  );

  _phase    = result['phase'];       // collect / hold / prepare
  _hands    = result['hands'];       // jumlah tangan terdeteksi
  _poseOk   = result['poseOk'];      // tubuh terlihat di frame
  if (_phase == 'hold') {
    _hasilDeteksi = result['label'];  // tampilkan hasil prediksi
    _akurasi      = result['confidence'];
  }
}
"""
