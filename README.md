# Causal OT CSS Experiments

This repository contains GPU/CPU Python code for the synthetic experiments used
to study causal subspace selection (CSS), Wasserstein projection pursuit (WPP),
and sliced Wasserstein baselines in the optimal-transport causal PI project.

The code is organized as a small, reproducible research repository: core solver
code is placed in `src/`, example generated figures are placed in `results/`,
and command-line usage is documented in `docs/`.

## Repository Layout

```text
Causal-OT-CSS-Code/
├── src/
│   ├── RiemannianBCD_gpu_fast_fixed.py
│   ├── css_wpp_fullsw_baseline_final.py
│   └── finite_sample_trade_off_gpu_fast.py
├── results/
│   ├── css_n200.pdf
│   └── css_n20000.pdf
├── docs/
│   └── experiments.md
├── environment.yml
├── requirements.txt
└── README.md
```

## Main Scripts

- `src/RiemannianBCD_gpu_fast_fixed.py` implements the NumPy/CuPy backend for
  Riemannian block coordinate descent on the Stiefel manifold.
- `src/css_wpp_fullsw_baseline_final.py` compares CSS, WPP, full-dimensional
  scaled sliced Wasserstein, and the analytic Gaussian ground truth.
- `src/finite_sample_trade_off_gpu_fast.py` decomposes the synthetic experiment
  error into population bias, finite-sample error, and total error.

## Installation

Create an isolated Python environment first.

```bash
conda create -n causal_ot python=3.10 -y
conda activate causal_ot
pip install -r requirements.txt
```

For GPU experiments, install exactly one CuPy build matching the CUDA runtime on
your machine. For example, on CUDA 11.x:

```bash
pip install cupy-cuda11x==13.3.0
```

On CUDA 12.x:

```bash
pip install cupy-cuda12x==13.3.0
```

## Quick Start

Run the CSS/WPP/full sliced Wasserstein comparison on GPU:

```bash
cd Causal-OT-CSS-Code
python src/css_wpp_fullsw_baseline_final.py \
  --gpu \
  --gpu-device 0 \
  --dtype float64 \
  --n 2000 \
  --n-trials 20 \
  --k-max 30 \
  --out results/css_n2000.pdf
```

Enable plateau-based dimension selection and plot annotation:

```bash
python src/css_wpp_fullsw_baseline_final.py \
  --gpu \
  --gpu-device 0 \
  --dtype float64 \
  --n 2000 \
  --n-trials 20 \
  --k-max 30 \
  --k-min 5 \
  --plateau-rel-tol 0.03 \
  --select-k-plateau \
  --out results/css_n2000_selected_k.pdf
```

Run the finite-sample trade-off experiment:

```bash
python src/finite_sample_trade_off_gpu_fast.py \
  --gpu \
  --gpu-device 0 \
  --dtype float64 \
  --n 2000 \
  --n-trials 20 \
  --k-max 30 \
  --out results/finite_sample_tradeoff_n2000.pdf
```

## Outputs

The experiment scripts save:

- `.pdf` figures for paper/rebuttal visualization.
- `.npz` files containing raw arrays for later analysis.
- `.csv` summaries for the CSS/WPP/full sliced baseline script.

## Notes

- Large experiments such as `n=50000` should be run on a GPU server.
- Use `float64` when numerical stability matters more than speed.
- The `--select-k-plateau` flag is optional. Without it, all dimensions up to
  `--k-max` are still evaluated but no selected dimension is reported.

## Citation

If this repository is made public, add the final paper citation here.

