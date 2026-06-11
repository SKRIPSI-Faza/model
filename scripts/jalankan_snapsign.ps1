# ============================================================
#  SnapSign v10 — Jalankan Manual Step-by-Step (PowerShell)
#  Jalankan: pwsh -ExecutionPolicy Bypass -File jalankan_snapsign.ps1
#  atau klik kanan → "Run with PowerShell"
# ============================================================

$ROOT    = Split-Path -Parent $MyInvocation.MyCommand.Path
$BACKEND = (Resolve-Path (Join-Path $ROOT "..\backend")).Path
$FLUTTER = (Resolve-Path (Join-Path $ROOT "..\flutter_application_isyarat")).Path
Set-Location $ROOT

# ── Warna helper ────────────────────────────────────────────
function Show-Step($num, $total, $msg) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  LANGKAH $num/$total : $msg" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
}
function OK($msg)   { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function INFO($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Yellow }
function ERR($msg)  { Write-Host "  [ERR]  $msg" -ForegroundColor Red }
function Pause-Step {
    Write-Host ""
    Write-Host "  Tekan ENTER untuk lanjut ke langkah berikutnya..." -ForegroundColor DarkGray
    Read-Host | Out-Null
}

# ── Header ──────────────────────────────────────────────────
Clear-Host
Write-Host @"

  ███████╗███╗   ██╗ █████╗ ██████╗ ███████╗██╗ ██████╗ ███╗   ██╗
  ██╔════╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║
  ███████╗██╔██╗ ██║███████║██████╔╝███████╗██║██║  ███╗██╔██╗ ██║
  ╚════██║██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██║██║   ██║██║╚██╗██║
  ███████║██║ ╚████║██║  ██║██║     ███████║██║╚██████╔╝██║ ╚████║
  ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                 v10 — MobileNetV2 + TSM + CBAM  (24 Kelas BISINDO)
"@ -ForegroundColor Magenta

Write-Host "  Pilih mode:" -ForegroundColor White
Write-Host "  [1] Jalankan Server + Flutter (Demo/Pengujian)" -ForegroundColor White
Write-Host "  [2] Training Model (train_v10.py)" -ForegroundColor White
Write-Host "  [3] Evaluasi Model (eval_topk.py)" -ForegroundColor White
Write-Host "  [4] Hanya Server Flask (tanpa Flutter)" -ForegroundColor White
Write-Host ""
$mode = Read-Host "  Pilihan (1/2/3/4)"

# ═══════════════════════════════════════════════════════════════════
#  MODE 1 — Server + Flutter
# ═══════════════════════════════════════════════════════════════════
if ($mode -eq "1") {

    # ── LANGKAH 1: Cek Python ──────────────────────────────────────
    Show-Step 1 6 "Cek Python"
    try {
        $pyVer = python --version 2>&1
        OK $pyVer
    } catch {
        ERR "Python tidak ditemukan. Pastikan Python ada di PATH."
        exit 1
    }
    Pause-Step

    # ── LANGKAH 2: Cek dependencies ───────────────────────────────
    Show-Step 2 6 "Cek Library Python"
    $deps = @("torch", "flask", "cv2", "mediapipe", "numpy")
    $missing = @()
    foreach ($pkg in $deps) {
        $check = python -c "import $pkg; print('ok')" 2>&1
        if ($check -match "ok") {
            OK "$pkg tersedia"
        } else {
            ERR "$pkg TIDAK ditemukan"
            $missing += $pkg
        }
    }
    if ($missing.Count -gt 0) {
        INFO "Install yang kurang: pip install $($missing -join ' ')"
        $install = Read-Host "  Install sekarang? (y/n)"
        if ($install -eq "y") {
            $pipMap = @{ "cv2" = "opencv-python"; "mediapipe" = "mediapipe" }
            foreach ($m in $missing) {
                $pkgName = if ($pipMap[$m]) { $pipMap[$m] } else { $m }
                INFO "pip install $pkgName ..."
                pip install $pkgName
            }
        }
    }
    Pause-Step

    # ── LANGKAH 3: Deteksi IP WiFi ────────────────────────────────
    Show-Step 3 6 "Deteksi IP WiFi"
    $wifiIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi" -ErrorAction SilentlyContinue).IPAddress
    if (-not $wifiIP) {
        # fallback cari semua IP non-loopback
        $wifiIP = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" } |
            Select-Object -First 1).IPAddress
    }
    if ($wifiIP) {
        OK "IP ditemukan: $wifiIP"
    } else {
        ERR "IP WiFi tidak terdeteksi. Jalankan ipconfig dan isi manual."
        $wifiIP = Read-Host "  Masukkan IP WiFi laptop kamu"
    }
    INFO "Server akan berjalan di: http://${wifiIP}:8000"
    Pause-Step

    # ── LANGKAH 4: Update IP di Flutter ──────────────────────────
    Show-Step 4 6 "Update IP di Flutter app_constants.dart"
    $dartFile = Join-Path $FLUTTER "lib\core\app_constants.dart"
    if (Test-Path $dartFile) {
        $content = Get-Content $dartFile -Raw
        $newContent = $content -replace "http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+", "http://${wifiIP}:8000"
        $newContent | Set-Content $dartFile -Encoding utf8 -NoNewline
        OK "app_constants.dart diupdate → http://${wifiIP}:8000"
        INFO "Isi saat ini:"
        Get-Content $dartFile | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkCyan }
    } else {
        ERR "app_constants.dart tidak ditemukan di: $dartFile"
    }
    Pause-Step

    # ── LANGKAH 5: Buka Firewall + Start Server ───────────────────
    Show-Step 5 6 "Buka Firewall Port 8000 & Jalankan Server Flask"
    INFO "Membuka firewall port 8000..."
    try {
        $rule = Get-NetFirewallRule -DisplayName "SnapSign-8000" -ErrorAction SilentlyContinue
        if (-not $rule) {
            New-NetFirewallRule -DisplayName "SnapSign-8000" -Direction Inbound `
                -Protocol TCP -LocalPort 8000 -Action Allow -ErrorAction Stop | Out-Null
            OK "Firewall rule dibuat untuk port 8000"
        } else {
            OK "Firewall rule sudah ada"
        }
    } catch {
        INFO "Gagal buat firewall rule (butuh Admin). Coba jalankan open_firewall_8000.bat sebagai Admin."
    }

    INFO "Memulai server Flask di terminal baru..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "Set-Location '$BACKEND'; Write-Host 'Server SnapSign v10' -ForegroundColor Green; python server_v10.py --checkpoint best_stage2_v10.pth --port 8000"

    INFO "Menunggu server siap (15 detik)..."
    $wait = 15
    for ($i = $wait; $i -gt 0; $i--) {
        Write-Host "`r  Menunggu... $i detik  " -NoNewline
        Start-Sleep 1
    }
    Write-Host ""

    # Cek health
    try {
        $resp = Invoke-WebRequest -Uri "http://${wifiIP}:8000/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        OK "Server merespons — HTTP $($resp.StatusCode)"
    } catch {
        INFO "Server belum merespons di /health, coba /predict saja nanti"
    }
    Pause-Step

    # ── LANGKAH 6: Jalankan Flutter ───────────────────────────────
    Show-Step 6 6 "Jalankan Aplikasi Flutter"
    INFO "Pastikan HP tersambung USB + USB Debugging aktif"
    $flutterDir = $FLUTTER

    $adb = Get-Command adb -ErrorAction SilentlyContinue
    if ($adb) {
        $devices = adb devices 2>&1 | Select-String "device$"
        if ($devices) {
            OK "Perangkat Android terdeteksi:"
            $devices | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGreen }
        } else {
            INFO "Tidak ada perangkat. Sambungkan HP dan aktifkan USB Debugging."
        }
    }

    Write-Host ""
    INFO "Menjalankan: flutter run"
    INFO "Tekan Ctrl+C di terminal Flutter untuk stop."
    Write-Host ""
    $run = Read-Host "  Jalankan Flutter sekarang? (y/n)"
    if ($run -eq "y") {
        Set-Location $flutterDir
        flutter run
        Set-Location $ROOT
    }

    Write-Host ""
    Write-Host "  SnapSign aktif di http://${wifiIP}:8000" -ForegroundColor Green
    Write-Host "  Preview frame: http://${wifiIP}:8000/preview" -ForegroundColor DarkCyan
}

# ═══════════════════════════════════════════════════════════════════
#  MODE 2 — Training
# ═══════════════════════════════════════════════════════════════════
elseif ($mode -eq "2") {

    Show-Step 1 3 "Cek GPU / CUDA"
    $cudaCheck = python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1
    OK $cudaCheck
    if ($cudaCheck -match "False") {
        INFO "CUDA tidak tersedia — training akan jalan di CPU (SANGAT lambat)"
        $cont = Read-Host "  Lanjut? (y/n)"
        if ($cont -ne "y") { exit 0 }
    }
    Pause-Step

    Show-Step 2 3 "Cek Dataset Path di train_v10.py"
    $datasetLine = Select-String -Path "$BACKEND\train_v10.py" -Pattern "DATASET_PATH" | Select-Object -First 1
    INFO "Dataset path di script: $($datasetLine.Line.Trim())"
    INFO "Pastikan path ini benar sebelum lanjut!"
    Pause-Step

    Show-Step 3 3 "Jalankan Training"
    INFO "Command: python train_v10.py"
    INFO "Output: best_stage1_v10.pth, best_stage2_v10.pth, confusion_matrix_test_v10.png"
    Write-Host ""
    $run = Read-Host "  Mulai training? (y/n)"
    if ($run -eq "y") {
        python "$BACKEND\train_v10.py"
    }
}

# ═══════════════════════════════════════════════════════════════════
#  MODE 3 — Evaluasi
# ═══════════════════════════════════════════════════════════════════
elseif ($mode -eq "3") {

    Show-Step 1 2 "Pilih Checkpoint untuk Evaluasi"
    $pths = Get-ChildItem "$BACKEND\*.pth" | Select-Object -ExpandProperty Name
    $pths | ForEach-Object -Begin { $i = 1 } -Process {
        Write-Host "  [$i] $_"; $i++
    }
    $pick = Read-Host "  Pilih nomor checkpoint"
    $chosen = $pths[$([int]$pick - 1)]
    OK "Dipilih: $chosen"
    Pause-Step

    Show-Step 2 2 "Jalankan Evaluasi Top-K"
    INFO "Command: python eval_topk.py --checkpoint $chosen"
    $run = Read-Host "  Jalankan? (y/n)"
    if ($run -eq "y") {
        python "$BACKEND\eval_topk.py" --checkpoint "$BACKEND\$chosen"
    }
}

# ═══════════════════════════════════════════════════════════════════
#  MODE 4 — Hanya Server
# ═══════════════════════════════════════════════════════════════════
elseif ($mode -eq "4") {

    Show-Step 1 2 "Deteksi IP"
    $wifiIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi" -ErrorAction SilentlyContinue).IPAddress
    if (-not $wifiIP) {
        $wifiIP = Read-Host "  Masukkan IP laptop"
    }
    OK "IP: $wifiIP"
    Pause-Step

    Show-Step 2 2 "Jalankan Server Flask"
    INFO "Server: http://${wifiIP}:8000"
    INFO "Health: http://${wifiIP}:8000/health"
    INFO "Preview: http://${wifiIP}:8000/preview"
    Write-Host ""
    Set-Location $BACKEND
    python "$BACKEND\server_v10.py" --checkpoint best_stage2_v10.pth --port 8000
}

else {
    ERR "Pilihan tidak valid."
}

Write-Host ""
Write-Host "  Selesai." -ForegroundColor Green
