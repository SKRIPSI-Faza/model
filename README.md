# BISINDO v10 — Augmentasi Pencahayaan (ablasi vs v9)

Perbaikan terarah dari **v9** untuk mengatasi **domain gap pencahayaan**:
v9 mendapat test **97,78%** tapi flat di kamera baru karena kamera nyata
berbeda terang/kontras/semburat warna lampu dari data latih.

## Prinsip ablasi (jujur untuk skripsi)
Seperti v8→v9 yang hanya mengubah segmentasi, **v10 hanya mengubah augmentasi
fotometrik**. Semua yang lain identik dengan v9 → kalau akurasi kamera membaik,
penyebabnya jelas augmentasi cahaya (1 variabel berubah).

| Aspek | v9 | v10 | Alasan |
|---|---|---|---|
| Brightness | 0.8–1.2 | **0.6–1.4** | robust kamera gelap/terang |
| Contrast | 0.8–1.2 | **0.6–1.4** | variasi kontras ruangan |
| Saturation | 0.9–1.1 | **0.8–1.2** | warna lampu berbeda |
| Hue | ±0.05 | **±0.08** | semburat warna lampu |
| Gamma | — | **0.7–1.5 (BARU)** | respons cahaya kamera non-linear |
| Augmentasi **spasial** | ±7°/±7%/0.88–1.12 | **TETAP** | arah tangan = makna isyarat |
| Arsitektur / epoch / split / seed | — | **TETAP** | 1 variabel berubah |
| Dataset | WLBISINDO_seg | **SAMA** | hanya augmentasi yang beda |

## Cara menjalankan
Dataset **sama** dengan v9 (tidak perlu preprocess ulang). Edit path di
`train_v10.py` (`DATASET_PATH`, `SAVE_DIR`), lalu:
```bash
python train_v10.py
```
Output (di `SAVE_DIR`): `best_stage1_v10.pth`, `best_stage2_v10.pth`,
`bisindo_24_v10_checkpoint.pth`, `training_history_v10.png`,
`confusion_matrix_test_v10.png`.

## Evaluasi & perbandingan dengan v9
1. **Test set (subject-dependent):** bandingkan `confusion_matrix_test_v10.png`
   vs v9. Test mungkin **sedikit turun** (augmentasi lebih kuat = sedikit lebih
   sulit) — itu wajar dan bukan kegagalan.
2. **Yang benar-benar diukur = kamera nyata.** Rekam beberapa klip di kondisi
   kameramu (folder berlabel), lalu pakai `eval_folder.py` (di root project):
   ```bash
   # v9 (baseline) vs v10 pada data kamera yang SAMA
   python ../eval_folder.py --dataset kamera_uji --split all --segment \
       --checkpoint ../best_stage2_v9.pth
   python ../eval_folder.py --dataset kamera_uji --split all --segment \
       --checkpoint best_stage2_v10.pth
   ```
   Kalau v10 lebih tinggi di data kamera → augmentasi cahaya berhasil
   menjembatani domain gap. **Inilah angka untuk tabel ablasi Bab 4.**
3. **Realtime:** jalankan `realtime_test.py --checkpoint <best_stage2_v10.pth>`.

## Catatan untuk skripsi
- Klaim v10: *"memperkuat augmentasi fotometrik mengurangi domain gap
  pencahayaan kamera real-time"* — ditopang perbandingan terkontrol vs v9.
- `--light` (CLAHE/white-balance) di inference = pendekatan komplementer
  sisi-inference; v10 = pendekatan sisi-training. Keduanya bisa dilaporkan.
- Limitasi tetap berlaku: random split = subject-dependent (Bab 5).
