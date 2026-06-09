@echo off
title SnapSign — Jalankan Server + Flutter
color 0A
cd /d "%~dp0"

REM Tambahkan Flutter ke PATH kalau belum ada
set PATH=C:\flutter\bin;%PATH%

echo.
echo ================================================
echo   SnapSign v10 — Setup Otomatis
echo ================================================
echo.

REM ── Langkah 1: Cek Python ──────────────────────────────────────────
echo [1/5] Cek Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan. Pastikan Python ada di PATH.
    pause & exit /b 1
)
python --version
echo.

REM ── Langkah 2: Cek Flutter ─────────────────────────────────────────
echo [2/5] Cek Flutter...
if not exist "C:\flutter\bin\flutter.bat" (
    echo [ERROR] Flutter tidak ditemukan di C:\flutter\bin\
    pause & exit /b 1
)
echo Flutter OK (C:\flutter\bin)
echo.

REM ── Langkah 3: Ambil IP WiFi otomatis ─────────────────────────────
echo [3/5] Ambil IP WiFi...
for /f "delims=" %%i in ('powershell -NoProfile -Command ^
    "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi' -ErrorAction SilentlyContinue).IPAddress"') do set WIFI_IP=%%i

if "%WIFI_IP%"=="" (
    echo [WARN] WiFi tidak terdeteksi otomatis. Coba cara manual...
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
        set RAW=%%a
        for /f "tokens=1" %%b in ("%%a") do set WIFI_IP=%%b
    )
)

if "%WIFI_IP%"=="" (
    echo [ERROR] Tidak bisa dapat IP WiFi.
    echo         Jalankan 'ipconfig' dan cari IPv4 WiFi, lalu update manual:
    echo         flutter_application_isyarat\lib\core\app_constants.dart
    pause & exit /b 1
)

echo IP WiFi ditemukan: %WIFI_IP%
echo.

REM ── Langkah 4: Update IP di Flutter ────────────────────────────────
echo [4/5] Update IP di app Flutter...
set DART_FILE=%~dp0flutter_application_isyarat\lib\core\app_constants.dart

powershell -NoProfile -Command ^
    "(Get-Content '%DART_FILE%') -replace 'http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:8000', 'http://%WIFI_IP%:8000' | Set-Content '%DART_FILE%'"

echo IP di app_constants.dart diubah ke: http://%WIFI_IP%:8000
echo.

REM ── Langkah 5: Jalankan Server di terminal baru ────────────────────
echo [5/5] Jalankan server Flask di terminal baru...
start "SnapSign Server" cmd /k "cd /d %~dp0 && echo. && echo Server berjalan di http://%WIFI_IP%:8000 && echo Biarkan jendela ini TERBUKA. Ctrl+C untuk stop. && echo. && python server_v10.py --port 8000"

echo Menunggu server siap (15 detik)...
timeout /t 15 /nobreak >nul

REM ── Cek server hidup ───────────────────────────────────────────────
echo Cek koneksi server...
powershell -NoProfile -Command ^
    "try { $r = Invoke-WebRequest -Uri 'http://%WIFI_IP%:8000/health' -TimeoutSec 5 -UseBasicParsing; Write-Host '[OK] Server merespons - status' $r.StatusCode } catch { Write-Host '[WARN] Server belum merespons, coba lanjut saja' }"
echo.

REM ── Jalankan Flutter ───────────────────────────────────────────────
echo ================================================
echo   Server   : http://%WIFI_IP%:8000
echo   Preview  : http://%WIFI_IP%:8000/preview
echo   Health   : http://%WIFI_IP%:8000/health
echo ================================================
echo.
echo Pastikan HP tersambung USB + USB Debugging aktif.
echo.
pause

echo Menjalankan Flutter...
cd /d "%~dp0flutter_application_isyarat"
flutter run

pause
