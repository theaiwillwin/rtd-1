# CWD-PHFT — Local Setup (Windows, CUDA 11.8)

Instructions for setting up this workspace on a Windows 10/11 machine with an
NVIDIA GPU and CUDA 11.8 drivers. The repo was bootstrapped and verified in a
CPU-only cloud session; nothing in the model pins tensors to CPU, so it uses
CUDA automatically once torch can see a GPU.

## 1. Clone

```powershell
cd D:\Projects
git clone https://github.com/theaiwillwin/rtd-1.git Hyper_context
cd Hyper_context
git checkout claude/cwd-phft-workspace-setup-2ws2rq
```

## 2. Virtual environment

Always use `.venv`. Never install to system Python.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -c "import sys; print(sys.prefix != sys.base_prefix)"   # must print True
```

## 3. Install torch (CUDA 11.8 build FIRST)

A plain `pip install torch` on Windows gives a CPU-only wheel. Install the
cu118 build first, then the rest of the requirements — the pre-installed
torch already satisfies `torch>=2.1.0`, so pip leaves it alone:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Alternatively, reuse an existing venv that already has a CUDA 11.8 torch:
activate it and run only the two `pip install -r ...` lines.

## 4. Verify GPU

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect something like `2.x.x+cu118 True`. If it prints `False`, check
`nvidia-smi` works and that the venv's torch is a `+cu118` build, not `+cpu`.

## 5. Run verification

```powershell
pytest python\tests\ -v

python python\model\lorentz.py     # round-trip error < 1e-3
python python\model\octonion.py    # compression ~7.9x
python python\model\hamming.py     # syndrome magnitude prints cleanly
```

All tests should pass. Note for `test_lorentz.py`: the expmap0/logmap0
round-trip only holds for inputs with norm <= MAX_TANGENT_NORM (2.0) —
larger inputs are clamped by design, not a bug.

## 6. Training

Full run with the established hyperparameters (WikiText-2 + GPT-2
tokenizer download from Hugging Face happens on first use):

```powershell
python python\training\train.py                # 10k steps, curvature 0.1 -> 2.0
python python\training\train.py --wandb        # same, logged to W&B project cwd-phft
```

Offline smoke test (synthetic data, no downloads):

```powershell
python python\training\train.py --dataset synthetic --steps 20 `
  --dim 64 --num-layers 2 --num-heads 4 --mem-size 16 `
  --vocab-size 1000 --batch-size 2 --seq-len 32
```

Checkpoints land in `checkpoints\` (gitignored). Note: the WikiText-2
path could not be network-verified in the cloud bootstrap session
(huggingface.co is blocked by its proxy policy); the tokenization and
batching logic is covered offline by `python\tests\test_data.py`.
