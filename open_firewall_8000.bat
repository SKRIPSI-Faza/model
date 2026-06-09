@echo off
REM Klik-kanan -> Run as administrator. Membuka port 8000 agar HP bisa konek.
title Buka Firewall Port 8000 (SnapSign v10)
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Harus dijalankan sebagai Administrator.
    echo         Klik-kanan file ini -> Run as administrator.
    pause & exit /b 1
)
netsh advfirewall firewall delete rule name="SnapSign v10 8000" >nul 2>&1
netsh advfirewall firewall add rule name="SnapSign v10 8000" dir=in action=allow protocol=TCP localport=8000
echo.
echo [OK] Port 8000 dibuka untuk koneksi masuk.
pause
