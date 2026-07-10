# CWD-PHFT one-command local training (Windows, NVIDIA GPU, CUDA 11.8)
#
#   cd D:\Projects\Hyper_context   (or wherever this repo is cloned)
#   .\run_local.ps1
#
# Does everything: venv, CUDA torch, deps, data download (GitHub mirrors,
# no Hugging Face account needed), then launches the full 10k-step run.
# Re-running auto-resumes from the newest checkpoint in checkpoints\.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# --- venv -------------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"

# --- dependencies -----------------------------------------------------
# A pre-existing venv may hold a CPU-only torch (plain `pip install torch`
# on Windows): detect that and replace it with the cu118 build.
& $py -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing CUDA 11.8 torch (replacing any CPU-only build)..."
    & $py -m pip install --upgrade pip
    & $py -m pip uninstall -y torch 2>$null
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu118
}
# Always ensure the rest of the deps (wandb included) are present -- this
# used to live inside the CPU-torch branch above, so a venv that already
# had a working CUDA torch would silently skip installing everything else.
& $py -m pip install -r requirements.txt -r requirements-dev.txt
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available - check nvidia-smi and that torch is a +cu118 build'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "GPU check failed - aborting before training." }

# --- wandb --------------------------------------------------------------
# Logs everything (loss, energy, grad norm, per-layer memory/velocity/
# curvature_scale diagnostics) to Weights & Biases. Online logging needs
# WANDB_API_KEY set once (from wandb.ai/authorize) -- e.g.
#   $env:WANDB_API_KEY = "..."
# before running this script. Without it, fall back to offline mode so the
# run never hangs waiting on an interactive login prompt; `wandb sync
# wandb\offline-run-...` uploads it later once you do have a key.
$wandbArgs = @("--wandb")
if (-not $env:WANDB_API_KEY) {
    Write-Host "wandb: WANDB_API_KEY not set -- running in offline mode (set it and re-run, or 'wandb sync' the offline run later)"
    $env:WANDB_MODE = "offline"
}

# --- console log backstop ------------------------------------------------
# Belt-and-suspenders: keep a plain timestamped log file regardless of
# wandb, since scrolled-away console output is otherwise unrecoverable.
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$logFile = "logs\train_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# --- data (GitHub mirrors, cached after first download) ----------------
New-Item -ItemType Directory -Force -Path "data\wikitext-2", "data\gpt2-tokenizer" | Out-Null
$files = @{
    "data\wikitext-2\train.txt"        = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    "data\wikitext-2\valid.txt"        = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
    "data\gpt2-tokenizer\vocab.json"   = "https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/master/GPT2/encoder.json"
    "data\gpt2-tokenizer\merges.txt"   = "https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/master/GPT2/vocab.bpe"
}
foreach ($dest in $files.Keys) {
    if (-not (Test-Path $dest)) {
        Write-Host "Downloading $dest..."
        Invoke-WebRequest -Uri $files[$dest] -OutFile $dest
    }
}
$env:WIKITEXT2_DIR      = Join-Path $PSScriptRoot "data\wikitext-2"
$env:GPT2_TOKENIZER_DIR = Join-Path $PSScriptRoot "data\gpt2-tokenizer"

# --- auto-resume from newest checkpoint --------------------------------
$resume = @()
$latest = Get-ChildItem "checkpoints\cwd_phft_step*.pt" -ErrorAction SilentlyContinue |
    Sort-Object { [int]($_.BaseName -replace '\D', '') } | Select-Object -Last 1
if ($latest) {
    Write-Host "Resuming from $($latest.FullName)"
    $resume = @("--resume", $latest.FullName)
}

# --- train -------------------------------------------------------------
Write-Host "Logging console output to $logFile"
& $py python\training\train.py `
    --dataset wikitext --steps 10000 `
    --log-every 10 --eval-every 500 --eval-batches 10 `
    --ckpt-every 200 --keep-ckpts 3 --ckpt-dir checkpoints `
    @wandbArgs @resume @args 2>&1 | Tee-Object -FilePath $logFile
