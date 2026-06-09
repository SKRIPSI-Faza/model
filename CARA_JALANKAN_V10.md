# Cara Jalankan SnapSign v10 (Model → API → Flutter)

Model RGB v10 (MobileNetV2+TSM+CBAM) di HP lewat backend Flask.
**HP dan laptop WAJIB di WiFi yang sama. Jangan pakai data seluler.**

---

## Bagian 0 — Yang harus ada (sudah lengkap)
| Komponen | Lokasi | Status |
|---|---|---|
| Model (bobot) | `d:\v10_lighting\best_stage2_v10.pth` | ✅ ada |
| Server API | `d:\v10_lighting\server_v10.py` | ✅ ada |
| App Flutter | `d:\v10_lighting\flutter_application_isyarat\` | ✅ ada |
| IP laptop | `192.168.88.134` = sudah cocok di app | ✅ |

> Model **tidak perlu dilatih ulang** — server tinggal memuat `best_stage2_v10.pth`.

---

## Bagian 1 — Buka firewall port 8000 (SEKALI saja)
Buka **PowerShell sebagai Administrator** (klik-kanan Start → Terminal (Admin)):
```powershell
netsh advfirewall firewall add rule name="SnapSign v10 8000" dir=in action=allow protocol=TCP localport=8000
```
Atau klik-kanan `open_firewall_8000.bat` → **Run as administrator**.

---

## Bagian 2 — Jalankan server (model + API)

### 2a. STOP server lama dulu (PENTING)
Port 8000 mungkin masih dipakai server skeleton lama (`api.py`). Cek & matikan:
```powershell
netstat -ano | findstr :8000        REM lihat PID di kolom terakhir
taskkill /PID <PID_itu> /F           REM matikan (ganti <PID_itu>)
```
Atau kalau jendela tempat `api.py`/server lama jalan masih terbuka → tekan `Ctrl + C` di situ.

### 2b. Jalankan server v10
Terminal biasa:
```powershell
cd d:\v10_lighting
python server_v10.py --port 8000
```
**Berhasil** kalau muncul:
```
Endpoint : http://192.168.88.134:8000
Siklus   : prepare 2.0s -> collect 5.0s -> hold 3.0s
```
**Biarkan jendela ini TERBUKA.** Stop = `Ctrl + C`.

> Cahaya HP beda jauh? jalankan: `python server_v10.py --port 8000 --light clahe+gray`

---

## Bagian 3 — Cek koneksi (sebelum buka app)
Di **browser HP**, ketik: `http://192.168.88.134:8000/health`
- Muncul JSON `"status":"ok"` → koneksi beres, lanjut.
- Gagal → WiFi beda / firewall belum dibuka / IP berubah (lihat Troubleshoot).

---

## Bagian 4 — Jalankan app Flutter di HP
Sambungkan HP via USB (aktifkan **USB debugging** di HP), lalu:
```powershell
cd d:\v10_lighting\flutter_application_isyarat
flutter pub get
flutter devices          REM pastikan HP terdeteksi
flutter run              REM pilih HP kalau diminta
```
Atau build APK lalu install manual:
```powershell
flutter build apk --release
REM hasil: build\app\outputs\flutter-apk\app-release.apk  -> copy ke HP, install
```
Di app: buka halaman **Deteksi** → ikuti siklus **bersiap → rekam 5 dtk → hasil**.

---

## Bagian 5 — Urutan menyalakan (ringkas)
1. (sekali) buka firewall 8000.
2. `python server_v10.py --port 8000`  ← biarkan terbuka.
3. HP & laptop satu WiFi.
4. Cek `http://192.168.88.134:8000/health` di browser HP.
5. `flutter run` (atau buka apk) → halaman Deteksi.

---

## Troubleshoot
| Masalah | Sebab | Solusi |
|---|---|---|
| `/health` gagal di HP | WiFi beda / firewall | Samakan WiFi; jalankan firewall (Bagian 1) |
| IP bukan 192.168.88.134 | pindah WiFi | `ipconfig` → lihat IPv4 WiFi → ganti `apiBaseUrl` di `lib\core\app_constants.dart`, lalu `flutter run` ulang |
| App "API Not Ready" | server mati / IP salah | pastikan server jalan + IP cocok |
| Label selalu "motor"/ragu | gap signer (bukan bug) | wajar untuk signer baru; lihat PRESENTASI.md |
| `flutter` tidak dikenal | Flutter SDK tdk di PATH | jalankan dari folder Flutter / set PATH |

### Ganti IP di app (kalau berubah)
File `lib\core\app_constants.dart`:
```dart
static const String apiBaseUrl = 'http://<IP-BARU>:8000';
```
Simpan → `flutter run` ulang.
