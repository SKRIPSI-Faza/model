Write-Host "=== SnapSign: Model Fine-Tune v2 STRATIFIED (24 kata) ===" -ForegroundColor Cyan
Write-Host "Checkpoint: best_finetune_v2_stratified.pth" -ForegroundColor Yellow
Write-Host "Buka app HP, sambungkan ke IP laptop ini port 8000" -ForegroundColor Green
Write-Host ""
& C:\Users\ASUS\AppData\Local\Programs\Python\Python39\python.exe d:\v10_lighting\backend\server_v10.py --checkpoint d:\v10_lighting\backend\best_finetune_v2_stratified.pth --port 8000
