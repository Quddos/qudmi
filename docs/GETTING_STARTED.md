# Getting started

## Prerequisites

- Python 3.10+ (developed against 3.14)
- A CUDA-capable GPU is strongly recommended for training (developed against an RTX 3060,
  12GB VRAM) but not required for exploring the code.

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
```

Install PyTorch with CUDA support explicitly — the plain `pip install torch` from PyPI gives you
a **CPU-only** build with no warning, so check your GPU driver's CUDA version first
(`nvidia-smi`, top-right corner) and match it to a wheel from
https://download.pytorch.org/whl/torch/. For a CUDA 13.x driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Verify CUDA is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Next steps

1. Read [SPEC.md](SPEC.md) for the exact problem definition.
2. Follow [DATA.md](DATA.md) to get a small AMASS subset downloaded.
3. Check [ROADMAP.md](ROADMAP.md) for current status.
