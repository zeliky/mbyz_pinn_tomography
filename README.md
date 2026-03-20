# PINN Tomography

PINN tomography project with a unified `State -> Operator -> (optional) Initializer/Policy` flow under `src/tomo/`.

This guide documents how to set up and run the project in a **local virtual environment only** (no global Python package changes).

## Requirements

- WSL/Linux shell
- Python `3.12` (project supports `>=3.11`)
- NVIDIA GPU + WSL GPU driver (for CUDA runs)
- `uv` available in your shell (`uv --version`)

Optional (only if you need to rebuild native FMM library):
- `gcc`
- system C toolchain (`build-essential` on Ubuntu/Debian)

## 0. Install `uv` (if missing)

If `uv` is not installed, install it for your user:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `uv` is still not found in the current terminal, add it to `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

This installs the `uv` binary only. Project Python dependencies still go into `.venv`.

## 1. Verify CUDA in WSL

Run in terminal:

```bash
nvidia-smi
```

Expected:
- GPU is listed
- Driver and CUDA version are shown

## 2. Create and Activate a Local Virtual Environment

From repo root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Notes:
- This creates `.venv/` inside this project only.
- Do not run global `pip install ...` outside the venv.

## 3. Install Project Dependencies

From repo root (with venv active):

```bash
uv sync
```

This installs dependencies declared in `pyproject.toml` (including `torch`, `torch-geometric`, `hydra-core`, etc.) into `.venv`.

## 4. Validate the Environment

### 4.1 Check PyTorch CUDA visibility

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'torch_cuda', torch.version.cuda)"
```

You want `cuda_available` to be `True` for GPU execution.

### 4.2 Check core imports

```bash
python -c "import torch_geometric; import tomo; print('pyg', torch_geometric.__version__)"
```

## 5. Run From This Editor Terminal

From repo root with `.venv` active:

```bash
python -m scripts.train
```

Expected output includes:
- `Train boot OK`

Note:
- This command is used as a smoke/boot check in this repo and can run without full dataset availability.

## 6. Run Tests

Basic test run:

```bash
pytest -q
```

Current status note:
- Environment setup is valid, but one test currently fails in-repo:
  - `tests/unit/test_operator.py::test_fmm_propagation_min_style`
- Treat this as a project/test logic issue, not an installation failure.

Useful targeted runs:

```bash
pytest -q tests/unit/test_operator.py
pytest -q tests/smoke/test_train_boot.py
```

## 7. Native FMM (`py2mat.msfm2d`) Notes

The repo uses a native shared library at:
- `src/py2mat/c/libmsfm2d.so`

If the library already exists, no action is needed.

If missing, rebuild from repo root:

```bash
cd src/py2mat/c
gcc -c -fPIC common.c -o common.o
gcc -c -fPIC msfm2d.c -o msfm2d.o
gcc -shared -o libmsfm2d.so msfm2d.o common.o -lm
cd ../../..
```

Then verify FMM-dependent unit test:

```bash
pytest -q tests/unit/test_forward_tof.py
```

## 8. Daily Usage

Each new terminal session:

```bash
cd /home/yaronz/pinn_tomography
export PATH="$HOME/.local/bin:$PATH"  # needed if uv was installed via install.sh
source .venv/bin/activate
```

Then run your commands (`python -m scripts.train`, `pytest -q`, etc.).

