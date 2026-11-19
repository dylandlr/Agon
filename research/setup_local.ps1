# Set up Agon: Eos on native Windows (PowerShell). Native Windows JAX runs on the CPU only;
# for an NVIDIA GPU, use WSL2 (Ubuntu) and run setup_local.sh there instead.
# Usage:  powershell -ExecutionPolicy Bypass -File setup_local.ps1
#   then: .\.venv\Scripts\Activate.ps1 ; python scripts\run_local.py --quick
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -q -U pip
python -m pip install -q -U jax
python -m pip install -q -U "flax>=0.12.10" "optax>=0.2.8" pytest matplotlib
python -m pip install -q --no-deps -e .
python scripts\run_local.py --steps check
Write-Host ""
Write-Host "Ready (CPU). Next:  .\.venv\Scripts\Activate.ps1 ; python scripts\run_local.py --quick"
Write-Host "For your NVIDIA GPU, install WSL2 and run setup_local.sh inside it."
