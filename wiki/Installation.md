# Installation & Environment Setup

## 1. System Requirements

- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+), macOS, or Windows WSL2
- **Python**: Version $\ge 3.8$
- **GPU (Recommended)**: NVIDIA GPU with CUDA 11.8 or 12.x support (e.g., RTX 3090, 4090, A100, V100). PKDP can also execute on CPU by passing `--device cpu`.

---

## 2. Conda Environment Setup (Recommended)

Using Conda ensures isolated dependency management:

```bash
# 1. Create a dedicated conda environment
conda create -n PKDP_env python=3.8 -y

# 2. Activate the environment
conda activate PKDP_env

# 3. Clone the PKDP repository
git clone https://github.com/yunmika/PKDP.git
cd PKDP

# 4. Make execution script executable (Linux / macOS)
chmod +x ./PKDP.py

# 5. Install Python dependencies
pip install -r requirements.txt
```

---

## 3. PyTorch CUDA Verification

Ensure your PyTorch installation detects your NVIDIA GPU correctly:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If PyTorch does not detect your GPU, install the official PyTorch build matching your local CUDA toolkit version:

```bash
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Dependencies List

The core dependencies specified in `requirements.txt` are:
- `torch >= 1.10.0`
- `numpy >= 1.20.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `scipy >= 1.7.0`
- `optuna >= 2.10.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`
