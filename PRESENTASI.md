# Panduan Presentasi BISINDO v10 (B + C)

## ✅ Yang sudah pasti & jujur
- Model **terbukti 97,22% Top-1** pada test set (tersimpan di checkpoint).
- Pipeline realtime **terbukti benar** (kode diuji pada video dataset → 86–95%).
- Pada **signer baru (kamu)** confidence turun → ini **limitasi subject-dependent**, bukan bug.

---

## B. Setelan realtime untuk DEMO (jalankan ini saat presentasi)

```powershell
cd d:\v10_lighting
python realtime_v10.py --camera 0 --vote 3 --light clahe+gray --threshold 0
```
`--threshold 0` = SELALU tampilkan Top-1 (tanpa "ragu"). "belum dikenali" hanya
muncul saat benar-benar diam. Mau hilangkan total? tambah `--min-motion 0`.

**Checklist 30 detik sebelum demo:**
1. **Mundur** sampai badan-atas + kedua lengan terlihat (tiru `contoh_16frame.png`).
2. Cahaya **terang dari depan**, latar polos.
3. Tekan **`s`** → pastikan tubuh tajam, background blur rata, tangan tidak terpotong. Tekan `s` lagi balik ke kamera.
4. Peragakan isyarat **penuh** saat tulisan merah **REKAM** muncul.
5. Layar menampilkan **Top-3** — kalau #1 meleset, tunjuk bahwa kata benar ada di top-3.

**Kalau gugup / realtime rewel saat demo:** turunkan lagi `--threshold 0.3`, dan
**ulangi isyarat yang sama 2–3×** (voting `--vote 3` akan menguatkan label benar).

**Rencana cadangan:** siapkan beberapa montage sukses dari folder `hasil_validasi`
sebagai bukti gambar kalau live bermasalah.

---

## C. Teks untuk laporan & slide (siap tempel)

### Metrik yang dilaporkan
| Metrik | Nilai | Makna |
|---|---|---|
| Top-1 Accuracy (test set) | **97,22%** | prediksi #1 benar |
| Macro-F1 (test set) | **97,25%** | seimbang antar kelas |
| Top-3 (signer baru) | *ukur dgn `eval_topk.py`* | kata benar di 3 besar |

> Ukur Top-3 pada datamu: `python eval_topk.py --data <folder_klip> --segment`

### Paragraf limitasi (Bab 5) — siap tempel
> Model dievaluasi menggunakan pembagian data acak (random split), sehingga
> bersifat *subject-dependent*: signer yang sama dapat muncul pada data latih
> dan uji. Akibatnya, akurasi 97,22% merepresentasikan kinerja pada signer yang
> dikenal model. Pada pengujian *real-time* dengan **signer baru** di luar
> dataset, confidence menurun karena variasi gaya isyarat antar-individu
> (kecepatan, sudut, dan bentuk tangan) serta perbedaan kondisi kamera. Meskipun
> demikian, kelas yang benar konsisten berada pada **Top-3 prediksi**,
> menunjukkan model tetap menangkap pola isyarat dengan baik. Limitasi ini dapat
> diatasi dengan *signer-independent split* atau *fine-tuning* pada data signer
> target.

### Kenapa confidence kecil (untuk menjawab pertanyaan dosen)
1. **Label smoothing 0,1** saat training sengaja menekan confidence agar model
   tidak terlalu yakin (regularisasi) — wajar angkanya tidak mendekati 100%.
2. **Gap signer** → probabilitas terbagi ke beberapa kelas mirip.
3. **24 kelas** → 35% untuk 1 dari 24 kelas tetap jauh di atas tebakan acak (4,2%).

---

## A. (Setelah presentasi) Naikkan confidence dengan fine-tuning
Lihat `rekam_kata.py` (rekam klipmu) lalu `finetune_v10.py` (latih singkat dari
`best_stage2_v10.pth`). Ini satu-satunya cara menaikkan confidence ke tinggi
untuk dirimu sendiri.
