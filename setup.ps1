# Setup script for distillation-testing
# Run from project root: .\setup.ps1

if (Test-Path .venv) {
    Write-Host "Removing existing .venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

Write-Host "Creating Python 3.12 venv..." -ForegroundColor Cyan
py -3.12 -m venv .venv

Write-Host "Activating venv..." -ForegroundColor Cyan
& .venv\Scripts\Activate.ps1

Write-Host "Installing dependencies (PyTorch CUDA 12.4 + ML stack)..." -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cu124 transformers peft datasets accelerate numpy scikit-learn matplotlib

Write-Host ""
Write-Host "Done! Run the benchmark with:" -ForegroundColor Green
Write-Host "  python -m distillation_benchmark.run_code_benchmark --quick" -ForegroundColor White
