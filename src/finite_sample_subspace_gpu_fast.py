# -*- coding: utf-8 -*-
"""Finite-sample subspace recovery experiment.

The experiment uses the same synthetic Gaussian setup and RBCD solver defaults
as `css_wpp_fullsw_baseline_final.py`, but evaluates only subspace recovery.
The target dimension is fixed at k_star=5.  The output PDF overlays alpha=0.0,
1.0, and 1.5 on a shared axis to make alpha-induced differences visible.  It
can also replot existing NPZ outputs without rerunning the expensive solver.

Outputs:
- A PDF containing subspace error vs sample size for each alpha.
- A compressed NPZ file containing raw subspace-error trial arrays.
- A CSV file containing mean / std subspace-error summaries by alpha and n.
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from RiemannianBCD_gpu_fast_fixed import RiemannianBlockCoordinateDescent


# -------------------------
# 1) DGP helpers
# -------------------------

def build_tau_signal_exp_decay(r: int, E_sig: float, eta_sig: float) -> np.ndarray:
    powers = eta_sig ** np.arange(r, dtype=float)
    C_sig = E_sig / float(np.sum(powers))
    return C_sig * powers


def build_tau_residual_powerlaw(k_res: int, E_res: float, alpha: float) -> np.ndarray:
    if k_res <= 0:
        return np.zeros(0, dtype=float)
    j = np.arange(1, k_res + 1, dtype=float)
    w = j ** (-alpha)
    w /= float(np.sum(w))
    return E_res * w


def tau_to_cov_eigs(tau: np.ndarray) -> np.ndarray:
    # If tau_i = (sqrt(lambda_i) - 1)^2, then sqrt(lambda_i) = 1 + sqrt(tau_i).
    return (1.0 + np.sqrt(np.maximum(tau, 0.0))) ** 2


def fixed_orthogonal_basis(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_sigmas_from_taus(taus: dict[float, np.ndarray], Q_basis: np.ndarray) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for key, tau in taus.items():
        lam = tau_to_cov_eigs(tau)
        Sigma = Q_basis @ np.diag(lam) @ Q_basis.T
        Sigma = (Sigma + Sigma.T) / 2.0
        out[key] = Sigma
    return out


def sample_gaussian_target(Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(Sigma.shape[0]))
    return rng.standard_normal((n, Sigma.shape[0])) @ L.T


# -------------------------
# 2) Subspace helpers
# -------------------------

def safe_initial_stiefel(rbcd: RiemannianBlockCoordinateDescent, d: int, k: int, seed: int) -> np.ndarray:
    try:
        return rbcd.InitialStiefel(d, k, seed=seed)
    except TypeError:
        return rbcd.InitialStiefel(d, k)


def projection_fro_error(U_hat: np.ndarray, U_true: np.ndarray) -> float:
    P_hat = U_hat @ U_hat.T
    P_true = U_true @ U_true.T
    return float(np.linalg.norm(P_true - P_hat, ord="fro"))


def run_single_trial_subspace(
    X,
    Y,
    k_star: int,
    rbcd: RiemannianBlockCoordinateDescent,
    seed: int,
    solver: str = "rbcd",
) -> np.ndarray:
    n, d = X.shape
    a = rbcd.asarray(np.full(n, 1.0 / n, dtype=rbcd.dtype_np))
    b = rbcd.asarray(np.full(n, 1.0 / n, dtype=rbcd.dtype_np))

    solve_fn = rbcd.run_RABCD if solver.lower() == "rabcd" else rbcd.run_RBCD
    U0 = safe_initial_stiefel(rbcd, d, k_star, seed=seed + 31 * k_star)
    _, U_hat, _, _, _, _ = solve_fn(
        a, b, X, Y, k_star, U0, warm_state=None, return_state=True
    )
    return U_hat.astype(rbcd.dtype_np, copy=False)


# -------------------------
# 3) Plotting / saving
# -------------------------

def summarize_bands(values: np.ndarray):
    return {
        "median": np.median(values, axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
        "q25": np.quantile(values, 0.25, axis=0),
        "q75": np.quantile(values, 0.75, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0, ddof=0),
    }


def plot_with_bands(ax, x, bands, label: str, color: str):
    ax.plot(x, bands["median"], linestyle="-", marker="o", linewidth=2.0, color=color, label=label)
    ax.fill_between(x, bands["q10"], bands["q90"], color=color, alpha=0.12)
    ax.fill_between(x, bands["q25"], bands["q75"], color=color, alpha=0.22)


def plot_subspace_graph(results, n_list: np.ndarray, save_path: str, rho: float):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    alphas = [0.0, 1.0, 1.5]
    alpha_labels = {0.0: "0", 1.0: "1.0", 1.5: "1.5"}
    alpha_colors = {0.0: "tab:blue", 1.0: "tab:green", 1.5: "tab:red"}
    x_pos = np.arange(len(n_list), dtype=float)

    y_low = min(float(np.min(results[alpha]["subspace_bands"]["q10"])) for alpha in alphas)
    y_high = max(float(np.max(results[alpha]["subspace_bands"]["q90"])) for alpha in alphas)
    y_pad = 0.08 * max(y_high - y_low, 1e-12)
    y_min = max(0.0, y_low - y_pad)
    y_max = y_high + y_pad

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))

    for alpha in alphas:
        data = results[alpha]
        plot_with_bands(
            ax,
            x_pos,
            data["subspace_bands"],
            label=rf"$\alpha={alpha_labels[alpha]}$",
            color=alpha_colors[alpha],
        )

    ax.set_title(rf"Subspace error at fixed $k^\star=5$ ($\rho={rho:g}$)")
    ax.set_xlabel("Number of points n")
    ax.set_ylabel(r"$\| \Omega^* - \widehat{\Omega} \|_F$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(n)) for n in n_list], rotation=30, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Residual decay")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {save_path}")


def summarize_contrast_bands(raw_subspace: dict[float, np.ndarray], reference_alpha: float = 0.0):
    reference = raw_subspace[reference_alpha]
    out = {}
    for alpha, values in raw_subspace.items():
        if alpha == reference_alpha:
            continue
        out[alpha] = summarize_bands((values - reference).T)
    return out


def plot_alpha_contrast_graph(
    contrast_bands: dict[float, dict[str, np.ndarray]],
    n_list: np.ndarray,
    save_path: str,
    rho: float,
    reference_alpha: float = 0.0,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    alpha_labels = {0.0: "0", 1.0: "1.0", 1.5: "1.5"}
    alpha_colors = {1.0: "tab:green", 1.5: "tab:red"}
    x_pos = np.arange(len(n_list), dtype=float)

    y_low = min(float(np.min(bands["q10"])) for bands in contrast_bands.values())
    y_high = max(float(np.max(bands["q90"])) for bands in contrast_bands.values())
    span = max(y_high - y_low, abs(y_high), abs(y_low), 1e-12)
    y_pad = 0.12 * span

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
    ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--", alpha=0.7)

    for alpha in [1.0, 1.5]:
        bands = contrast_bands[alpha]
        plot_with_bands(
            ax,
            x_pos,
            bands,
            label=rf"$\alpha={alpha_labels[alpha]}$ minus $\alpha={alpha_labels[reference_alpha]}$",
            color=alpha_colors[alpha],
        )

    ax.set_title(rf"Subspace error contrast at fixed $k^\star=5$ ($\rho={rho:g}$)")
    ax.set_xlabel("Number of points n")
    ax.set_ylabel(r"$\Delta\|\Omega^*-\widehat{\Omega}\|_F$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(n)) for n in n_list], rotation=30, ha="right")
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Contrast")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {save_path}")


def save_summary_csv(results, n_list: np.ndarray, csv_path: str):
    fieldnames = [
        "alpha",
        "n",
        "k_star",
        "rho",
        "subspace_err_mean",
        "subspace_err_std",
        "subspace_err_median",
        "subspace_err_q10",
        "subspace_err_q90",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in [0.0, 1.0, 1.5]:
            data = results[alpha]
            bands = data["subspace_bands"]
            for idx, n in enumerate(n_list):
                writer.writerow(
                    {
                        "alpha": alpha,
                        "n": int(n),
                        "k_star": int(data["k_star"]),
                        "rho": float(data["rho"]),
                        "subspace_err_mean": float(bands["mean"][idx]),
                        "subspace_err_std": float(bands["std"][idx]),
                        "subspace_err_median": float(bands["median"][idx]),
                        "subspace_err_q10": float(bands["q10"][idx]),
                        "subspace_err_q90": float(bands["q90"][idx]),
                    }
                )
    print(f"Saved: {csv_path}")


def save_contrast_csv(
    contrast_bands: dict[float, dict[str, np.ndarray]],
    n_list: np.ndarray,
    csv_path: str,
    k_star: int,
    rho: float,
    reference_alpha: float = 0.0,
):
    fieldnames = [
        "alpha",
        "reference_alpha",
        "n",
        "k_star",
        "rho",
        "contrast_mean",
        "contrast_std",
        "contrast_median",
        "contrast_q10",
        "contrast_q90",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in [1.0, 1.5]:
            bands = contrast_bands[alpha]
            for idx, n in enumerate(n_list):
                writer.writerow(
                    {
                        "alpha": alpha,
                        "reference_alpha": reference_alpha,
                        "n": int(n),
                        "k_star": int(k_star),
                        "rho": float(rho),
                        "contrast_mean": float(bands["mean"][idx]),
                        "contrast_std": float(bands["std"][idx]),
                        "contrast_median": float(bands["median"][idx]),
                        "contrast_q10": float(bands["q10"][idx]),
                        "contrast_q90": float(bands["q90"][idx]),
                    }
                )
    print(f"Saved: {csv_path}")


def print_one_line_summary(results, n_list: np.ndarray):
    print("\n==== Subspace summary at largest n ====")
    idx = len(n_list) - 1
    n = int(n_list[idx])
    for alpha in [0.0, 1.0, 1.5]:
        data = results[alpha]
        bands = data["subspace_bands"]
        print(
            f"alpha={alpha:<3} | n={n} | "
            f"subspace mean={bands['mean'][idx]:.4f} | "
            f"median={bands['median'][idx]:.4f} | std={bands['std'][idx]:.4f}"
        )


def print_contrast_summary(contrast_bands: dict[float, dict[str, np.ndarray]], n_list: np.ndarray):
    print("\n==== Alpha contrast summary at largest n ====")
    idx = len(n_list) - 1
    n = int(n_list[idx])
    for alpha in [1.0, 1.5]:
        bands = contrast_bands[alpha]
        print(
            f"alpha={alpha:<3} - alpha=0 | n={n} | "
            f"contrast mean={bands['mean'][idx]:.4f} | "
            f"median={bands['median'][idx]:.4f} | std={bands['std'][idx]:.4f}"
        )


def load_raw_subspace_npz(npz_path: str):
    data = np.load(npz_path)
    n_list = np.asarray(data["n_list"], dtype=int)
    k_star = int(np.asarray(data["k_star"]).ravel()[0])
    rho = float(np.asarray(data["rho"]).ravel()[0]) if "rho" in data.files else 0.10

    raw_subspace = {}
    for alpha in [0.0, 1.0, 1.5]:
        key = f"alpha_{alpha}_subspace_err"
        if key not in data.files:
            raise KeyError(f"Missing {key} in {npz_path}")
        raw_subspace[alpha] = np.asarray(data[key], dtype=float)
    return n_list, k_star, rho, raw_subspace


def build_results(raw_subspace: dict[float, np.ndarray], k_star: int, rho: float):
    return {
        alpha: {
            "k_star": k_star,
            "rho": rho,
            "subspace_bands": summarize_bands(raw_subspace[alpha].T),
        }
        for alpha in [0.0, 1.0, 1.5]
    }


def write_outputs(
    raw_subspace: dict[float, np.ndarray],
    n_list: np.ndarray,
    k_star: int,
    rho: float,
    out_path: str,
    plot_mode: str,
    save_npz: bool = True,
):
    results = build_results(raw_subspace, k_star=k_star, rho=rho)
    out_base = Path(out_path)

    if plot_mode in ("absolute", "both"):
        absolute_path = str(out_base if plot_mode == "absolute" else out_base.with_name(out_base.stem + "_absolute" + out_base.suffix))
        plot_subspace_graph(results, n_list, save_path=absolute_path, rho=rho)

    contrast_bands = summarize_contrast_bands(raw_subspace, reference_alpha=0.0)
    if plot_mode in ("contrast", "both"):
        contrast_path = str(out_base if plot_mode == "contrast" else out_base.with_name(out_base.stem + "_contrast" + out_base.suffix))
        plot_alpha_contrast_graph(contrast_bands, n_list, save_path=contrast_path, rho=rho)

    out_csv = str(out_base.with_suffix('.csv'))
    save_summary_csv(results, n_list, out_csv)
    save_contrast_csv(
        contrast_bands,
        n_list,
        str(out_base.with_name(out_base.stem + "_contrast").with_suffix('.csv')),
        k_star=k_star,
        rho=rho,
    )

    if save_npz:
        out_npz = str(out_base.with_suffix('.npz'))
        flat = {
            "n_list": n_list.astype(int),
            "k_star": np.array([k_star], dtype=int),
            "rho": np.array([rho], dtype=float),
        }
        for alpha in [0.0, 1.0, 1.5]:
            flat[f"alpha_{alpha}_subspace_err"] = raw_subspace[alpha]
        np.savez_compressed(out_npz, **flat)
        print(f"Saved: {out_npz}")

    print_one_line_summary(results, n_list)
    print_contrast_summary(contrast_bands, n_list)


# -------------------------
# 4) CLI / main
# -------------------------

def parse_n_list(raw: str) -> np.ndarray:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--n-list must contain at least one integer.")
    if any(n <= 0 for n in values):
        raise argparse.ArgumentTypeError("All values in --n-list must be positive.")
    return np.array(values, dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finite-sample fixed-k subspace recovery experiment"
    )
    parser.add_argument("--gpu", action="store_true", help="Use CuPy/GPU backend")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--n-list", type=parse_n_list, default=parse_n_list("25,50,100,250,500,1000"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--from-npz", type=str, default=None,
                        help="Replot an existing NPZ result without rerunning RBCD")
    parser.add_argument("--plot-mode", choices=["contrast", "absolute", "both"], default="contrast",
                        help="contrast plots alpha-wise differences relative to alpha=0")
    parser.add_argument("--rho", type=float, default=0.10,
                        help="Residual energy fraction; increase to emphasize alpha-induced subspace differences")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--sinkhorn-inner", type=int, default=3,
                        help="Set to 1 to match Huang et al. RBCD exactly")
    parser.add_argument("--sinkhorn-max-iter", type=int, default=400)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-7)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--solver", choices=["rbcd", "rabcd"], default="rbcd")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True,
                        help="Print fixed-k progress during each trial")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable fixed-k progress printing")
    parser.add_argument("--out", type=str, default="finite_sample_subspace.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    # Match the current baseline comparison setup.
    d = 30
    r = 5
    k_star = 5
    E_total = 120.0
    rho = float(args.rho)
    eta_sig = 1.0
    alphas = [0.0, 1.0, 1.5]

    n_list = args.n_list
    n_trials = args.n_trials
    if args.from_npz is not None:
        n_list_loaded, k_star_loaded, rho_loaded, raw_subspace_loaded = load_raw_subspace_npz(args.from_npz)
        write_outputs(
            raw_subspace=raw_subspace_loaded,
            n_list=n_list_loaded,
            k_star=k_star_loaded,
            rho=rho_loaded,
            out_path=args.out,
            plot_mode=args.plot_mode,
            save_npz=False,
        )
        return

    if rho < 0.0 or rho >= 1.0:
        raise ValueError("--rho must be in [0, 1).")

    rbcd = RiemannianBlockCoordinateDescent(
        eta=2.0,
        tau=0.05,
        max_iter=args.max_iter,
        threshold=args.threshold,
        verbose=False,
        use_gpu=args.gpu,
        gpu_device=args.gpu_device,
        sinkhorn_inner=args.sinkhorn_inner,
        sinkhorn_max_iter=args.sinkhorn_max_iter,
        sinkhorn_tol=args.sinkhorn_tol,
        dtype=dtype,
        store_pi=False,
    )
    print(
        f"Backend: {rbcd.backend_name}, dtype={rbcd.dtype_np}, fixed k*={k_star}, "
        f"rho={rho:g}, trials={n_trials}, solver={args.solver}"
    )

    E_sig = (1.0 - rho) * E_total
    E_res = rho * E_total
    tau_sig = build_tau_signal_exp_decay(r, E_sig, eta_sig)

    taus: dict[float, np.ndarray] = {}
    for alpha in alphas:
        tau_res = build_tau_residual_powerlaw(d - r, E_res, alpha)
        taus[alpha] = np.concatenate([tau_sig, tau_res])

    Q_basis = fixed_orthogonal_basis(d, seed=12345)
    sigmas = make_sigmas_from_taus(taus, Q_basis)
    U_true = Q_basis[:, :k_star]

    raw_subspace = {
        alpha: np.zeros((len(n_list), n_trials), dtype=float)
        for alpha in alphas
    }

    print(f"Running {n_trials} trials for each n and alpha ...")
    overall_tic = time.perf_counter()
    for i, n in enumerate(n_list):
        n_int = int(n)
        print(f"[n={n_int}] start", flush=True)
        for t in range(n_trials):
            rngX = np.random.default_rng(10000 + 1009 * i + t)
            X = rngX.standard_normal((n_int, d)).astype(dtype, copy=False)
            Xb = rbcd.asarray(X)

            for alpha in alphas:
                trial_tic = time.perf_counter()
                Sigma = sigmas[alpha]
                rngY = np.random.default_rng(20000 + 1009 * i + 97 * t + int(100 * alpha))
                Y = sample_gaussian_target(Sigma, n_int, rngY).astype(dtype, copy=False)
                Yb = rbcd.asarray(Y)

                U_hat = run_single_trial_subspace(
                    X=Xb,
                    Y=Yb,
                    k_star=k_star,
                    rbcd=rbcd,
                    seed=30000 + 1009 * i + 999 * t + int(100 * alpha),
                    solver=args.solver,
                )
                err_sub = projection_fro_error(U_hat, U_true)
                raw_subspace[alpha][i, t] = err_sub

                if args.progress:
                    print(
                        f"[n={n_int} | trial {t + 1}/{n_trials} | alpha={alpha}] "
                        f"subspace error={err_sub:.4f} | "
                        f"elapsed={time.perf_counter() - trial_tic:.1f}s",
                        flush=True,
                    )

                gc.collect()
                rbcd.free_backend_memory()
        print(f"[n={n_int}] done", flush=True)
    print(f"All trials completed in {time.perf_counter() - overall_tic:.1f}s", flush=True)

    write_outputs(
        raw_subspace=raw_subspace,
        n_list=n_list,
        k_star=k_star,
        rho=rho,
        out_path=args.out,
        plot_mode=args.plot_mode,
        save_npz=True,
    )


if __name__ == "__main__":
    main()
