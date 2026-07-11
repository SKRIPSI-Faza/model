Write-Host "=== SnapSign: Model Fine-Tune LAMA (5 kata) ===" -ForegroundColor Cyan
Write-Host "Checkpoint: best_finetune_v10.pth" -ForegroundColor Yellow
Write-Host "Buka http://localhost:8000 untuk uji upload video" -ForegroundColor Green
Write-Host ""
& C:\Users\ASUS\AppData\Local\Programs\Python\Python39\python.exe d:\v10_lighting\backend\server_v10.py --checkpoint d:\v10_lighting\backend\best_finetune_v10.pth --port 8000
