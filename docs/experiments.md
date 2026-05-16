# Experiments

This document records the command-line entry points used by the repository.

## 1. CSS, WPP, and Full Sliced Wasserstein Baseline

Script:

```bash
python src/css_wpp_fullsw_baseline_final.py --help
```

Main options:

| Option | Meaning |
| --- | --- |
| `--gpu` | Use CuPy/GPU instead of NumPy/CPU. |
| `--gpu-device` | GPU index. Usually `0` on a single-card AutoDL instance. |
| `--dtype` | Numerical precision, either `float32` or `float64`. |
| `--n` | Number of samples from each distribution. |
| `--n-trials` | Number of independent Monte Carlo trials. |
| `--k-max` | Maximum subspace dimension evaluated. |
| `--select-k-plateau` | Enable plateau-based dimension selection. |
| `--k-min` | Minimum candidate dimension for plateau selection. |
| `--plateau-rel-tol` | Relative WPP gain threshold used by the plateau rule. |
| `--sw-L` | Number of residual sliced-Wasserstein directions. |
| `--full-sw-L` | Number of full-dimensional sliced-Wasserstein directions. |
| `--sinkhorn-inner` | Sinkhorn scaling steps per RBCD outer iteration. |
| `--out` | Output PDF path. Matching `.npz` and `.csv` files are also saved. |

Small test run:

```bash
python src/css_wpp_fullsw_baseline_final.py \
  --gpu --gpu-device 0 --dtype float64 \
  --n 200 --n-trials 1 --k-max 30 \
  --out results/css_n200_test.pdf
```

Large GPU run:

```bash
python src/css_wpp_fullsw_baseline_final.py \
  --gpu --gpu-device 0 --dtype float64 \
  --n 50000 --n-trials 1 --k-max 30 \
  --sinkhorn-inner 1 \
  --out results/css_n50000_trial1.pdf
```

## 2. Finite-Sample Trade-Off

Script:

```bash
python src/finite_sample_trade_off_gpu_fast.py --help
```

Example:

```bash
python src/finite_sample_trade_off_gpu_fast.py \
  --gpu --gpu-device 0 --dtype float64 \
  --n 2000 --n-trials 20 --k-max 30 \
  --out results/finite_sample_tradeoff_n2000.pdf
```

## 3. AutoDL CUDA Notes

Check GPU visibility:

```bash
nvidia-smi
python -c "import cupy as cp; x=cp.ones((10,)); print(cp.cuda.runtime.getDeviceCount()); print((x+1).sum())"
```

If CuPy cannot find `libnvrtc`, make sure the CUDA library path matches the
actual CUDA version installed on the machine. For CUDA 11.8:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

