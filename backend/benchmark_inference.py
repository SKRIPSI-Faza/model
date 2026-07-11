"""
Benchmark latensi inferensi model BISINDO (MobileNetV2 + TSM + CBAM)
======================================================================
Mengukur waktu forward pass MURNI model (tanpa segmentasi/decode) untuk
melengkapi pengujian black-box dengan angka performa kuantitatif.

CATATAN PENTING: ini BUKAN latensi end-to-end sistem. Latensi nyata yang
dirasakan pengguna juga mencakup segmentasi MediaPipe (~150-300ms untuk
16 frame, lihat timing log server) dan overhead jaringan (encode/decode
Base64 + HTTP). Gunakan angka ini khusus untuk membahas kecepatan model,
bukan klaim "sistem merespons dalam X ms".

Jalankan: python backend/benchmark_inference.py
"""
import os, csv, time
import numpy as np
import torch

import realtime_v10 as R   # pakai model + load_model asli, jamin arsitektur identik server

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_finetune_v2.pth")
N_WARMUP = 10        # run awal dibuang (cold-start)
N_RUNS   = 100        # run yang benar-benar diukur
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, classes = R.load_model(CHECKPOINT_PATH, DEVICE)
model.eval()

model_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
n_params = sum(p.numel() for p in model.parameters())
print(f"Checkpoint         : {CHECKPOINT_PATH}")
print(f"Ukuran checkpoint   : {model_size_mb:.2f} MB")
print(f"Jumlah parameter    : {n_params:,}")
print(f"Device pengujian    : {DEVICE}")

dummy_input = torch.randn(1, R.NUM_FRAMES, 3, R.IMG_SIZE[0], R.IMG_SIZE[1]).to(DEVICE)

with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = model(dummy_input)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

latencies = []
with torch.no_grad():
    for _ in range(N_RUNS):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy_input)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

latencies = np.array(latencies)

print("\n=== Hasil Benchmark Inferensi Model (forward pass murni) ===")
print(f"Jumlah run            : {N_RUNS}")
print(f"Rata-rata latensi     : {latencies.mean():.2f} ms")
print(f"Std deviasi           : {latencies.std():.2f} ms")
print(f"Latensi minimum       : {latencies.min():.2f} ms")
print(f"Latensi maksimum      : {latencies.max():.2f} ms")
print(f"Latensi P95           : {np.percentile(latencies, 95):.2f} ms")
print(f"Throughput            : {1000 / latencies.mean():.2f} klip/detik")

out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "latency_ms"])
    for i, lat in enumerate(latencies):
        writer.writerow([i, lat])
print(f"\nHasil disimpan ke {out_csv}")
