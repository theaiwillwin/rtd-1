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
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing CUDA 11.8 torch (plain pip would give CPU-only)..."
    & $py -m pip install --upgrade pip
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu118
    & $py -m pip install -r requirements.txt -r requirements-dev.txt
}
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available - check nvidia-smi and that torch is a +cu118 build'; print('GPU:', torch.cuda.get_device_name(0))"

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
& $py python\training\train.py `
    --dataset wikitext --steps 10000 `
    --log-every 10 --eval-every 500 --eval-batches 10 `
    --ckpt-every 200 --keep-ckpts 3 --ckpt-dir checkpoints `
    @resume @args
