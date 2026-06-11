@echo off
title SnapSign Flask API - v10 RGB (MobileNetV2+TSM+CBAM)
cd /d "%~dp0"
echo ================================================
echo   SnapSign Flask API - v10 RGB
echo ================================================
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ada di PATH.
    pause & exit /b 1
)
echo [INFO] Boot server (torch + mediapipe ~10-20 detik)...
echo [INFO] Biarkan jendela ini TERBUKA. Ctrl+C untuk stop.
echo.
REM --light clahe+gray bisa ditambah kalau cahaya HP beda jauh
python server_v10.py --checkpoint best_stage2_v10.pth --port 8000
echo.
echo [INFO] Server berhenti.
pause
