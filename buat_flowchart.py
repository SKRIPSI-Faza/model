"""
Flowchart Metode SnapSign BISINDO v10 — Simbol Standar
  Oval          : Mulai / Selesai (terminal)
  Persegi       : Proses
  Persegi ganda : Proses model / predefined
  Jajaran genjang: Input / Output
  Belah ketupat : Keputusan
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.patches import Ellipse
import numpy as np, os
os.chdir(r'd:\v10_lighting')

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── Warna ──────────────────────────────────────────────────────────────────
C = dict(
    term  = '#D5D8DC',   # abu  – terminal
    inp   = '#AED6F1',   # biru – input/output
    proc  = '#A9DFBF',   # hijau – proses umum
    train = '#FAD7A0',   # oranye – proses training
    model = '#FDEBD0',   # krem – model (predefined)
    dec   = '#F9E79F',   # kuning – keputusan
    res   = '#D7BDE2',   # ungu – hasil
    edge  = '#2C3E50',
    txt   = '#17202A',
    arr   = '#2C3E50',
)

# ── Primitif ───────────────────────────────────────────────────────────────
def _txt(ax, cx, cy, s, fs=9, fw='normal'):
    ax.text(cx, cy, s, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=C['txt'], multialignment='center', zorder=6)

def oval(ax, cx, cy, w=3.4, h=0.62, lbl='', fs=10):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=C['term'], ec=C['edge'], lw=1.6, zorder=4))
    _txt(ax, cx, cy, lbl, fs=fs, fw='bold')
    return cy+h/2, cy-h/2

def rect(ax, cx, cy, w, h, clr, lbl='', fs=9):
    ax.add_patch(mpatches.Rectangle(
        (cx-w/2, cy-h/2), w, h,
        fc=clr, ec=C['edge'], lw=1.3, zorder=4))
    _txt(ax, cx, cy, lbl, fs=fs)
    return cy+h/2, cy-h/2

def predef(ax, cx, cy, w, h, clr, lbl='', fs=9):
    """Persegi dengan garis ganda vertikal di kiri-kanan (predefined process)."""
    ax.add_patch(mpatches.Rectangle(
        (cx-w/2, cy-h/2), w, h,
        fc=clr, ec=C['edge'], lw=1.3, zorder=4))
    m = 0.24
    for xv in [cx-w/2+m, cx+w/2-m]:
        ax.plot([xv, xv], [cy-h/2, cy+h/2], C['edge'], lw=1.0, zorder=5)
    _txt(ax, cx, cy, lbl, fs=fs, fw='bold')
    return cy+h/2, cy-h/2

def parallelogram(ax, cx, cy, w, h, clr, lbl='', fs=9):
    """Jajaran genjang untuk I/O."""
    sk = 0.32
    vx = [cx-w/2+sk, cx+w/2+sk, cx+w/2-sk, cx-w/2-sk]
    vy = [cy+h/2,    cy+h/2,    cy-h/2,    cy-h/2]
    ax.add_patch(Polygon(list(zip(vx,vy)), closed=True,
                         fc=clr, ec=C['edge'], lw=1.3, zorder=4))
    _txt(ax, cx, cy, lbl, fs=fs)
    return cy+h/2, cy-h/2

def diamond(ax, cx, cy, w, h, lbl='', fs=9):
    """Belah ketupat untuk keputusan."""
    vx = [cx,    cx+w/2, cx,    cx-w/2]
    vy = [cy+h/2, cy,    cy-h/2, cy]
    ax.add_patch(Polygon(list(zip(vx,vy)), closed=True,
                         fc=C['dec'], ec=C['edge'], lw=1.3, zorder=4))
    _txt(ax, cx, cy, lbl, fs=fs)
    return cy+h/2, cy-h/2

def arrow_down(ax, cx, y1, y2, lbl='', lside='right', fs=8):
    ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle='->', color=C['arr'],
                                lw=1.3, mutation_scale=14), zorder=5)
    if lbl:
        dx = 0.14 if lside=='right' else -0.14
        ax.text(cx+dx, (y1+y2)/2, lbl, fontsize=fs, color='#922B21',
                ha='left' if lside=='right' else 'right', va='center', zorder=7)

def arrow_side(ax, x1, y, x2, lbl='', lside='top', fs=8):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=C['arr'],
                                lw=1.2, mutation_scale=12), zorder=5)
    if lbl:
        ax.text((x1+x2)/2, y+0.07, lbl, fontsize=fs, color='#922B21',
                ha='center', va='bottom', zorder=7)

def line_seg(ax, pts, lw=1.2):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=C['arr'], lw=lw, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FLOWCHART PELATIHAN
# ══════════════════════════════════════════════════════════════════════════════
def make_training():
    fig, ax = plt.subplots(figsize=(9, 23))
    ax.set_xlim(0, 9); ax.set_ylim(0, 23)
    ax.axis('off'); fig.patch.set_facecolor('white')

    cx = 4.5
    W  = 7.4   # lebar persegi
    H  = 0.78  # tinggi persegi standar
    SP = 1.38  # jarak antar pusat node

    y = 22.3
    tops, bots = [], []

    def add(fn, *a, **kw):
        t, b = fn(*a, **kw)
        tops.append(t); bots.append(b)

    # 1. MULAI
    add(oval,         ax, cx, y, 3.4, 0.62, 'MULAI')
    y -= SP
    # 2. Dataset
    add(parallelogram,ax, cx, y, W, H, C['inp'], 'Dataset Video BISINDO\n(24 Kelas, Format .mp4)')
    y -= SP + 0.05
    # 3. Segmentasi
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Pra-Segmentasi Background\nMediaPipe Selfie Segmentation (Proses Offline)')
    y -= SP + 0.05
    # 4. Split
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Pembagian Data (Stratified Random Split)\nData Latih 70%  –  Validasi 15%  –  Uji 15%')
    y -= SP + 0.05
    # 5. Ekstraksi frame
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Ekstraksi 16 Frame per Video\nMargin 15% (Awal & Akhir), Uniform Temporal Sampling')
    y -= SP + 0.15
    # 6. Augmentasi
    add(rect,         ax, cx, y, W, H+0.18, C['proc'],
        'Augmentasi Konsisten (Hanya Data Latih)\n'
        'Spasial: Rotasi ±7°, Skala 0.88–1.12, Translasi ±7%\n'
        'Fotometrik: Kecerahan, Kontras, Saturasi, Rona, Gamma')
    y -= SP + 0.30
    # 7. Normalisasi
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Normalisasi ImageNet\nmean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]')
    y -= SP + 0.05
    # 8. Model
    add(predef,       ax, cx, y, W, H+0.12, C['model'],
        'Arsitektur Model\nMobileNetV2 + TSM (Temporal Shift Module) + CBAM')
    y -= SP + 0.18
    # 9. Stage 1
    add(rect,         ax, cx, y, W, H, C['train'],
        'Pelatihan Stage 1: Backbone Beku\nOptimizer AdamW,  LR = 10⁻³,  Maks 12 Epoch')
    y -= SP + 0.05
    # 10. Stage 2
    stage2_cy = y - SP + 0.05
    add(rect,         ax, cx, y, W, H, C['train'],
        'Pelatihan Stage 2: Partial Unfreeze (Layer ≥ 14)\nOptimizer AdamW,  LR = 10⁻⁴,  Maks 45 Epoch')
    stage2_bot = bots[-1]

    # 11. Decision: early stop
    y -= SP + 0.40
    dec_cy = y
    DW, DH = 4.8, 1.10
    add(diamond,      ax, cx, y, DW, DH, 'Early Stop\nTerpenuhi?')

    # Loop-back: Tidak → kanan → naik ke Stage 2
    lx = cx + DW/2 + 0.9
    y_loop_top = stage2_bot
    line_seg(ax, [(cx+DW/2, dec_cy), (lx, dec_cy), (lx, y_loop_top)])
    ax.annotate('', xy=(cx+W/2, y_loop_top), xytext=(lx, y_loop_top),
                arrowprops=dict(arrowstyle='->', color=C['arr'], lw=1.2, mutation_scale=12))
    ax.text(lx+0.08, dec_cy+0.14, 'Tidak', fontsize=8, color='#922B21', ha='left', va='bottom')

    # Ya → ke bawah (akan digambar oleh loop arrow_down)
    y -= SP + 0.22
    # 12. Evaluasi
    add(rect,         ax, cx, y, W, H, C['res'],
        'Evaluasi Model pada Data Uji')
    y -= SP
    # 13. Hasil
    add(parallelogram,ax, cx, y, W, H+0.12, C['res'],
        'Top-1 Accuracy: 97,22%  |  Macro-F1: 97,25%\nConfusion Matrix + Classification Report')
    y -= SP + 0.12
    # 14. Simpan
    add(parallelogram,ax, cx, y, W, H, C['inp'],
        'Simpan Model Terbaik: best_stage2_v10.pth')
    y -= SP
    # 15. SELESAI
    add(oval,         ax, cx, y, 3.4, 0.62, 'SELESAI')

    # ── Panah antar node ──────────────────────────────────────────────
    for i in range(len(bots)-1):
        if i == 10:   # decision → eval (Ya, ke bawah)
            arrow_down(ax, cx, bots[i], tops[i+1], lbl='Ya', lside='right')
        else:
            arrow_down(ax, cx, bots[i], tops[i+1])

    ax.set_title('Flowchart Pelatihan Model SnapSign BISINDO v10',
                 fontsize=13, fontweight='bold', color=C['edge'], pad=14)

    # Legenda
    patches = [
        mpatches.Patch(fc=C['term'],  ec=C['edge'], label='Terminal (Mulai/Selesai)'),
        mpatches.Patch(fc=C['inp'],   ec=C['edge'], label='Input / Output'),
        mpatches.Patch(fc=C['proc'],  ec=C['edge'], label='Proses Preprocessing'),
        mpatches.Patch(fc=C['train'], ec=C['edge'], label='Proses Pelatihan'),
        mpatches.Patch(fc=C['model'], ec=C['edge'], label='Proses Model (Predefined)'),
        mpatches.Patch(fc=C['dec'],   ec=C['edge'], label='Keputusan'),
        mpatches.Patch(fc=C['res'],   ec=C['edge'], label='Hasil / Evaluasi'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=7.5,
              framealpha=0.9, edgecolor='#BDC3C7', bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    plt.savefig('flowchart_training_v10.png', dpi=160, bbox_inches='tight')
    plt.close()
    print('[OK] flowchart_training_v10.png')


# ══════════════════════════════════════════════════════════════════════════════
# 2. FLOWCHART INFERENSI
# ══════════════════════════════════════════════════════════════════════════════
def make_inferensi():
    fig, ax = plt.subplots(figsize=(9, 20))
    ax.set_xlim(0, 9); ax.set_ylim(0, 20)
    ax.axis('off'); fig.patch.set_facecolor('white')

    cx = 4.5
    W  = 7.4
    H  = 0.78
    SP = 1.35

    y = 19.3
    tops, bots = [], []

    def add(fn, *a, **kw):
        t, b = fn(*a, **kw)
        tops.append(t); bots.append(b)

    # 1. MULAI
    add(oval,         ax, cx, y, 3.4, 0.62, 'MULAI')
    y -= SP
    # 2. Input frame
    add(parallelogram,ax, cx, y, W, H, C['inp'],
        'Input Frame dari Kamera (Desktop)\natau Stream Video dari Aplikasi HP')
    y -= SP + 0.05
    # 3. Norm cahaya
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Normalisasi Cahaya (Opsional)\nCLAHE + Gray-World White Balance')
    y -= SP + 0.05
    # 4. Segmentasi
    add(rect,         ax, cx, y, W, H+0.12, C['proc'],
        'Segmentasi Selfie\nMediaPipe Selfie Segmentation (model_sel=1)\nBackground diganti blur (kernel 41×41)')
    y -= SP + 0.22
    # 5. Kumpul frame
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Kumpulkan Frame Selama Fase Collect\n(Durasi 5 Detik per Siklus)')
    collect_cy = y

    # 6. Decision: gerak cukup?
    y -= SP + 0.30
    dec_cy = y
    DW, DH = 5.0, 1.10
    add(diamond,      ax, cx, y, DW, DH, 'Gerak Cukup?\n(Energi > Ambang Batas)')

    # Tidak → kanan → tampilkan belum dikenali → kembali ke collect
    bx, by = cx + DW/2 + 1.5, dec_cy
    bw, bh = 2.1, 0.72
    ax.add_patch(mpatches.Rectangle((bx-bw/2, by-bh/2), bw, bh,
                                    fc='#FDFEFE', ec=C['edge'], lw=1.2, zorder=4))
    _txt(ax, bx, by, 'Tampilkan:\n"Belum Dikenali"', fs=8.5)
    # Tidak → kotak
    line_seg(ax, [(cx+DW/2, dec_cy), (bx-bw/2, dec_cy)])
    ax.annotate('', xy=(bx-bw/2, dec_cy), xytext=(cx+DW/2, dec_cy),
                arrowprops=dict(arrowstyle='->', color=C['arr'], lw=1.2, mutation_scale=11))
    ax.text((cx+DW/2+bx-bw/2)/2, dec_cy+0.10, 'Tidak', fontsize=8, color='#922B21',
            ha='center', va='bottom')
    # kotak → kembali ke collect
    line_seg(ax, [(bx, by-bh/2), (bx, collect_cy), (cx+W/2, collect_cy)])
    ax.annotate('', xy=(cx+W/2, collect_cy), xytext=(bx, collect_cy),
                arrowprops=dict(arrowstyle='->', color=C['arr'], lw=1.1, mutation_scale=10))

    # Ya → ke bawah
    y -= SP + 0.22
    # 7. trim_active
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Pemotongan Segmen Aktif  (trim_active)\nHitung Energi Gerak, Ambil Puncak Aktivitas')
    y -= SP + 0.05
    # 8. sample
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Pengambilan Sampel 16 Frame Seragam  (sample_clip_eval)\nMargin 15% di Awal dan Akhir Klip')
    y -= SP + 0.05
    # 9. resize + norm
    add(rect,         ax, cx, y, W, H, C['proc'],
        'Resize 224×224 + Normalisasi ImageNet')
    y -= SP + 0.05
    # 10. inferensi
    add(predef,       ax, cx, y, W, H+0.12, C['model'],
        'Inferensi Model\nMobileNetV2 + TSM + CBAM  (best_stage2_v10.pth)')
    y -= SP + 0.18
    # 11. softmax + voting
    add(rect,         ax, cx, y, W, H, C['train'],
        'Softmax → Probabilitas 24 Kelas\nVoting Temporal (prob_hist, maxlen=3)')
    y -= SP + 0.05
    # 12. output
    add(parallelogram,ax, cx, y, W, H, C['res'],
        'Tampilkan: Label + Confidence (%) + Top-3 Prediksi')
    y -= SP
    # 13. SELESAI
    add(oval,         ax, cx, y, 3.4, 0.62, 'SELESAI')

    # ── Panah antar node ──────────────────────────────────────────────
    for i in range(len(bots)-1):
        if i == 5:   # decision → trim_active (Ya)
            arrow_down(ax, cx, bots[i], tops[i+1], lbl='Ya', lside='right')
        else:
            arrow_down(ax, cx, bots[i], tops[i+1])

    ax.set_title('Flowchart Inferensi Real-Time SnapSign BISINDO v10',
                 fontsize=13, fontweight='bold', color=C['edge'], pad=14)

    patches = [
        mpatches.Patch(fc=C['term'],  ec=C['edge'], label='Terminal'),
        mpatches.Patch(fc=C['inp'],   ec=C['edge'], label='Input / Output'),
        mpatches.Patch(fc=C['proc'],  ec=C['edge'], label='Proses Preprocessing'),
        mpatches.Patch(fc=C['model'], ec=C['edge'], label='Proses Model (Predefined)'),
        mpatches.Patch(fc=C['train'], ec=C['edge'], label='Proses Inferensi'),
        mpatches.Patch(fc=C['dec'],   ec=C['edge'], label='Keputusan'),
        mpatches.Patch(fc=C['res'],   ec=C['edge'], label='Hasil'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=7.5,
              framealpha=0.9, edgecolor='#BDC3C7', bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    plt.savefig('flowchart_inferensi_v10.png', dpi=160, bbox_inches='tight')
    plt.close()
    print('[OK] flowchart_inferensi_v10.png')


# ══════════════════════════════════════════════════════════════════════════════
# 3. FLOWCHART SISTEM KESELURUHAN (landscape)
# ══════════════════════════════════════════════════════════════════════════════
def make_sistem():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18); ax.set_ylim(0, 9)
    ax.axis('off'); fig.patch.set_facecolor('white')

    W = 4.0; H = 0.72; FS = 8.5

    # ── Label fase ────────────────────────────────────────────────────────────
    for xc, lbl in [(4.5,'FASE PELATIHAN'), (13.5,'FASE INFERENSI')]:
        ax.text(xc, 8.65, lbl, ha='center', fontsize=12,
                fontweight='bold', color='#1A5276')
    ax.axvline(9.0, color='#BDC3C7', lw=1.5, ls='--', zorder=1)

    # ── Kiri: Pelatihan ───────────────────────────────────────────────────────
    lx, SP = 4.5, 1.0
    ys_l = []
    y = 8.0
    nodes_l = [
        (parallelogram, C['inp'],   'Dataset Video BISINDO (24 Kelas)'),
        (rect,          C['proc'],  'Pra-Segmentasi Background (Offline)'),
        (rect,          C['proc'],  'Pembagian Data: Train 70% / Val 15% / Test 15%'),
        (rect,          C['proc'],  'Ekstraksi 16 Frame + Augmentasi Konsisten'),
        (rect,          C['proc'],  'Normalisasi ImageNet'),
        (predef,        C['model'], 'MobileNetV2 + TSM + CBAM'),
        (rect,          C['train'], 'Stage 1: Backbone Beku  (LR=10⁻³, 12 Epoch)'),
        (rect,          C['train'], 'Stage 2: Partial Unfreeze (LR=10⁻⁴, ≤45 Epoch)'),
        (parallelogram, C['res'],   'Top-1: 97,22%  |  Macro-F1: 97,25%'),
        (parallelogram, C['inp'],   'best_stage2_v10.pth'),
    ]
    for fn, clr, lbl in nodes_l:
        top, bot = fn(ax, lx, y, W, H, clr, lbl, fs=FS)
        ys_l.append((top, bot, y))
        y -= SP
    for i in range(len(ys_l)-1):
        arrow_down(ax, lx, ys_l[i][1], ys_l[i+1][0])

    # ── Kanan: Inferensi ──────────────────────────────────────────────────────
    rx = 13.5
    ys_r = []
    y = 8.0
    nodes_r = [
        (parallelogram, C['inp'],   'Input Frame (Kamera / HP)'),
        (rect,          C['proc'],  'Normalisasi Cahaya (CLAHE + Gray-World)'),
        (rect,          C['proc'],  'Segmentasi Selfie  (MediaPipe + Background Blur)'),
        (rect,          C['proc'],  'Kumpulkan Frame (5 Detik) → Deteksi Gerak'),
        (rect,          C['proc'],  'Sample 16 Frame Seragam (Margin 15%)'),
        (rect,          C['proc'],  'Resize 224×224 + Normalisasi ImageNet'),
        (predef,        C['model'], 'Inferensi Model (MobileNetV2 + TSM + CBAM)'),
        (rect,          C['train'], 'Softmax + Voting Temporal (3 Siklus)'),
        (parallelogram, C['res'],   'Label + Confidence (%) + Top-3 Prediksi'),
    ]
    for fn, clr, lbl in nodes_r:
        top, bot = fn(ax, rx, y, W, H, clr, lbl, fs=FS)
        ys_r.append((top, bot, y))
        y -= SP
    for i in range(len(ys_r)-1):
        arrow_down(ax, rx, ys_r[i][1], ys_r[i+1][0])

    # ── Panah "Muat Bobot" dari kiri ke kanan ─────────────────────────────────
    x_from = lx + W/2
    x_to   = rx - W/2
    y_mid  = (ys_l[-1][2] + ys_r[6][2]) / 2
    ax.annotate('', xy=(x_to, ys_r[6][2]),
                xytext=(x_from, ys_l[-1][2]),
                arrowprops=dict(arrowstyle='->', color='#C0392B',
                                lw=2.0, mutation_scale=16,
                                connectionstyle='arc3,rad=-0.3'), zorder=6)
    ax.text(9.0, y_mid + 0.1, 'Muat Bobot\nbest_stage2_v10.pth',
            ha='center', fontsize=8, color='#C0392B',
            style='italic', fontweight='bold', zorder=7)

    # ── Server / Flutter bar bawah ────────────────────────────────────────────
    bar_y = 0.45
    _, _ = rect(ax, 13.5, bar_y, 8.2, 0.58, C['inp'],
                'Flask API (server_v10.py)  ↔  Aplikasi Flutter SnapSign', fs=9)
    arrow_down(ax, rx, ys_r[-1][1], bar_y + 0.29)

    ax.set_title('Arsitektur Sistem SnapSign BISINDO v10  '
                 '(MobileNetV2 + TSM + CBAM | 24 Kelas | Real-Time)',
                 fontsize=13, fontweight='bold', color=C['edge'], pad=12)

    plt.tight_layout()
    plt.savefig('flowchart_sistem_v10.png', dpi=160, bbox_inches='tight')
    plt.close()
    print('[OK] flowchart_sistem_v10.png')


if __name__ == '__main__':
    make_training()
    make_inferensi()
    make_sistem()
    print('\nSelesai. File tersimpan di d:\\v10_lighting\\')
