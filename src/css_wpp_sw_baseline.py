# -*- coding: utf-8 -*-
"""Baseline comparison experiment for CSS vs WPP vs full d*SW.

This script mirrors the synthetic Gaussian setup used in
`finite_sample_trade_off_gpu_fast_fixed.py`, but instead of decomposing errors,
it directly reports estimator values:

1. CSS estimator:
       WPP_k + (d-k) * SW2 on the orthogonal complement of the learned subspace.
2. WPP estimator:
       projected OT value on the learned k-dimensional subspace.
3. Full sliced baseline:
       d * SW2 computed in the full ambient dimension.
4. True W2^2 value:
       analytic Gaussian ground truth used as a horizontal reference line.

Outputs:
- A PDF plot with one panel per alpha.
- A compressed NPZ file containing raw trial arrays.
- A CSV file containing the mean / std summary by alpha and k.
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from RiemannianBCD import RiemannianBlockCoordinateDescent


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


def make_sigmas_from_taus(d: int, taus: dict[float, np.ndarray], Q_basis: np.ndarray) -> dict[float, np.ndarray]:
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
# 2) Geometry / SW helpers
# -------------------------

def safe_initial_stiefel(rbcd: RiemannianBlockCoordinateDescent, d: int, k: int, seed: int) -> np.ndarray:
    try:
        return rbcd.InitialStiefel(d, k, seed=seed)
    except TypeError:
        return rbcd.InitialStiefel(d, k)


def extend_U_warmstart(U_prev: np.ndarray, seed: int) -> np.ndarray:
    d, k_prev = U_prev.shape
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((d, 1))
    U = np.hstack([U_prev, v])
    Q, _ = np.linalg.qr(U)
    return Q[:, : k_prev + 1]


def orthogonal_complement(U: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(U, mode="complete")
    return Q[:, U.shape[1]:]


def sliced_w2_empirical_batched(
    X,
    Y,
    L: int,
    seed: int,
    batch_size: int = 64,
    rbcd: RiemannianBlockCoordinateDescent | None = None,
) -> float:
    """Monte Carlo estimate of empirical sliced W2^2 between equally weighted samples."""
    if rbcd is None or not rbcd.use_gpu:
        rng = np.random.default_rng(seed)
        X = np.asarray(X)
        Y = np.asarray(Y)
        _, m = X.shape
        if m == 0:
            return 0.0
        total = 0.0
        done = 0
        while done < L:
            b = min(batch_size, L - done)
            Theta = rng.standard_normal((m, b))
            Theta /= np.linalg.norm(Theta, axis=0, keepdims=True) + 1e-12
            PX = X @ Theta
            PY = Y @ Theta
            PX.sort(axis=0)
            PY.sort(axis=0)
            total += float(np.sum(np.mean((PX - PY) ** 2, axis=0)))
            done += b
        return total / float(L)

    xp = rbcd.xp
    rs = xp.random.RandomState(seed)
    X = rbcd.asarray(X)
    Y = rbcd.asarray(Y)
    _, m = X.shape
    if m == 0:
        return 0.0

    total = 0.0
    done = 0
    while done < L:
        b = min(batch_size, L - done)
        Theta = rs.standard_normal((m, b)).astype(X.dtype, copy=False)
        Theta /= xp.linalg.norm(Theta, axis=0, keepdims=True) + 1e-12
        PX = X @ Theta
        PY = Y @ Theta
        PX = xp.sort(PX, axis=0)
        PY = xp.sort(PY, axis=0)
        total += rbcd.to_float(xp.sum(xp.mean((PX - PY) ** 2, axis=0)))
        done += b
    return total / float(L)


# -------------------------
# 3) Single-trial estimator computation
# -------------------------

def run_single_trial_estimators(
    X,
    Y,
    k_values: np.ndarray,
    rbcd: RiemannianBlockCoordinateDescent,
    sw_L: int,
    full_sw_L: int,
    seed: int,
    sw_batch: int = 64,
    full_sw_batch: int = 128,
    solver: str = "rbcd",
    carry_dual_across_k: bool = True,
    progress_label: str | None = None,
    progress: bool = True,
):
    n, d = X.shape
    a = rbcd.asarray(np.full(n, 1.0 / n, dtype=rbcd.dtype_np))
    b = rbcd.asarray(np.full(n, 1.0 / n, dtype=rbcd.dtype_np))

    solve_fn: Callable = rbcd.run_RABCD if solver.lower() == "rabcd" else rbcd.run_RBCD

    # Full-dimensional d * SW baseline is independent of k.
    full_sw2 = sliced_w2_empirical_batched(
        X, Y, L=full_sw_L, seed=seed + 13579, batch_size=full_sw_batch, rbcd=rbcd
    )
    full_dsw = float(d * full_sw2)

    wpp_hat = np.zeros(len(k_values), dtype=float)
    css_hat = np.zeros(len(k_values), dtype=float)

    U = None
    state = None
    for idx, k in enumerate(k_values):
        tic_k = time.perf_counter()
        kk = int(k)
        if U is None:
            U = safe_initial_stiefel(rbcd, d, kk, seed=seed + 31 * kk)
        else:
            U = extend_U_warmstart(U, seed=seed + 17 * kk).astype(rbcd.dtype_np, copy=False)

        warm_state = state if carry_dual_across_k else None
        _, U, _, _, _, state_k = solve_fn(
            a, b, X, Y, k, U, warm_state=warm_state, return_state=True
        )

        # Polish the projected OT value on the final learned subspace.
        _, _, _, f_val, _, polish_state = rbcd.run_fixed_projection_ot(
            a, b, X, Y, U=U, warm_state=state_k, return_state=True
        )
        wpp_hat[idx] = float(f_val)

        if kk < d:
            U_perp = orthogonal_complement(U).astype(rbcd.dtype_np, copy=False)
            U_perp_b = rbcd.asarray(U_perp)
            Xp = X @ U_perp_b
            Yp = Y @ U_perp_b
            sw2_res = sliced_w2_empirical_batched(
                Xp,
                Yp,
                L=sw_L,
                seed=seed + 2024 + 101 * kk,
                batch_size=sw_batch,
                rbcd=rbcd,
            )
            css_hat[idx] = wpp_hat[idx] + (d - kk) * sw2_res
        else:
            css_hat[idx] = wpp_hat[idx]

        state = polish_state if carry_dual_across_k else None
        if progress:
            prefix = f"{progress_label} | " if progress_label else ""
            print(
                f"{prefix}k={kk:>2}/{int(k_values[-1])} "
                f"({idx + 1}/{len(k_values)}) | "
                f"WPP={wpp_hat[idx]:.4f} | CSS={css_hat[idx]:.4f} | "
                f"elapsed={time.perf_counter() - tic_k:.1f}s",
                flush=True,
            )

    return wpp_hat, css_hat, full_dsw


# -------------------------
# 4) Plotting / saving
# -------------------------

def select_plateau_k(
    k_values: np.ndarray,
    wpp_values: np.ndarray,
    k_min: int,
    plateau_rel_tol: float,
) -> tuple[int, int, np.ndarray]:
    """Select the first dimension before the WPP path enters a relative plateau."""
    if k_min > int(k_values[-1]):
        raise ValueError(f"k_min={k_min} cannot exceed k_max={int(k_values[-1])}.")

    gains = np.full_like(wpp_values, fill_value=np.nan, dtype=float)
    for idx in range(1, len(k_values)):
        prev = max(abs(float(wpp_values[idx - 1])), 1e-12)
        gains[idx] = (float(wpp_values[idx]) - float(wpp_values[idx - 1])) / prev

    for idx in range(1, len(k_values)):
        if int(k_values[idx]) <= k_min:
            continue
        if gains[idx] <= plateau_rel_tol:
            return int(k_values[idx - 1]), idx - 1, gains

    return int(k_values[-1]), len(k_values) - 1, gains


def plot_estimator_graph(results, k_values: np.ndarray, save_path: str, show_selected_k: bool = False):
    alphas = sorted(list(results.keys()))
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.2 * len(alphas), 5.0), squeeze=False)

    for i, alpha in enumerate(alphas):
        ax = axes[0, i]
        data = results[alpha]

        ax.plot(k_values, data["wpp_mean"], linestyle=":", linewidth=2.4,
                color="tab:blue", label="WPP (RBCD)")
        ax.plot(k_values, data["css_mean"], linestyle="-", linewidth=2.4,
                color="tab:green", label="CSS (RBCD)")
        ax.axhline(data["full_dsw_mean"], linestyle="-.", linewidth=2.2,
                   color="tab:orange", label=r"$d\times \widehat{SW}_2^2$ (full)")
        ax.axhline(data["true_w2"], linestyle="--", linewidth=2.0,
                   color="black", label=r"True $W_2^2$")
        if show_selected_k:
            selected_k = int(data["selected_k"])
            selected_idx = int(np.where(k_values == selected_k)[0][0])
            selected_wpp = float(data["wpp_mean"][selected_idx])
            selected_css = float(data["css_mean"][selected_idx])
            selected_full = float(data["full_dsw_mean"])
            ax.axvline(selected_k, linestyle="--", linewidth=1.8,
                       color="tab:red", alpha=0.85, label=r"Selected $k^\star$")
            ax.scatter([selected_k, selected_k], [selected_wpp, selected_css],
                       color=["tab:blue", "tab:green"], s=36, zorder=5)
            ax.text(
                selected_k,
                0.03,
                rf"$k^\star={selected_k}$",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="tab:red",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="tab:red", alpha=0.85, boxstyle="round,pad=0.25"),
            )
            ax.text(
                0.98,
                0.05,
                f"k*={selected_k}\nWPP={selected_wpp:.2f}\nCSS={selected_css:.2f}\nfull d*SW={selected_full:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9, boxstyle="round,pad=0.3"),
            )

        ax.set_title(f"Estimator comparison ($\\alpha={alpha}$)")
        ax.set_xlabel(r"Subspace dimension $k^\star$")
        ax.set_ylabel(r"Estimated value")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(k_values[0], k_values[-1])
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {save_path}")


def save_summary_csv(results, k_values: np.ndarray, csv_path: str, include_selected_k: bool = False):
    fieldnames = [
        "alpha",
        "k",
        "wpp_mean",
        "wpp_std",
        "css_mean",
        "css_std",
        "full_dsw_mean",
        "full_dsw_std",
        "true_w2",
    ]
    if include_selected_k:
        fieldnames.extend(["selected_k", "wpp_rel_gain"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in sorted(results.keys()):
            data = results[alpha]
            for idx, k in enumerate(k_values):
                row = {
                    "alpha": alpha,
                    "k": int(k),
                    "wpp_mean": float(data["wpp_mean"][idx]),
                    "wpp_std": float(data["wpp_std"][idx]),
                    "css_mean": float(data["css_mean"][idx]),
                    "css_std": float(data["css_std"][idx]),
                    "full_dsw_mean": float(data["full_dsw_mean"]),
                    "full_dsw_std": float(data["full_dsw_std"]),
                    "true_w2": float(data["true_w2"]),
                }
                if include_selected_k:
                    row["selected_k"] = int(data["selected_k"])
                    row["wpp_rel_gain"] = (
                        float(data["wpp_rel_gains"][idx])
                        if np.isfinite(data["wpp_rel_gains"][idx])
                        else ""
                    )
                writer.writerow(row)
    print(f"Saved: {csv_path}")


def print_one_line_summary(results, k_values: np.ndarray, include_selected_k: bool = False):
    print("\n==== Mean estimator summary (best on grid) ====")
    for alpha in sorted(results.keys()):
        data = results[alpha]
        idx_css = int(np.argmax(data["css_mean"]))
        idx_wpp = int(np.argmax(data["wpp_mean"]))
        msg = (
            f"alpha={alpha:<3} | true W2^2={data['true_w2']:.4f} | "
            f"best CSS={data['css_mean'][idx_css]:.4f} @ k={int(k_values[idx_css])} | "
            f"best WPP={data['wpp_mean'][idx_wpp]:.4f} @ k={int(k_values[idx_wpp])}"
        )
        if include_selected_k:
            idx_sel = int(np.where(k_values == int(data["selected_k"]))[0][0])
            msg += (
                f" | selected k={int(data['selected_k'])} "
                f"(WPP={data['wpp_mean'][idx_sel]:.4f}, CSS={data['css_mean'][idx_sel]:.4f})"
            )
        msg += f" | full d*SW={data['full_dsw_mean']:.4f}"
        print(msg)


# -------------------------
# 5) CLI / main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline comparison: CSS vs WPP vs full d*SW under the synthetic Gaussian setup"
    )
    parser.add_argument("--gpu", action="store_true", help="Use CuPy/GPU backend")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--select-k-plateau", action="store_true",
                        help="Enable WPP plateau-based dimension selection and plot annotation")
    parser.add_argument("--k-min", type=int, default=5,
                        help="Minimum dimension allowed by the WPP plateau selection rule")
    parser.add_argument("--plateau-rel-tol", type=float, default=0.03,
                        help="Relative WPP gain threshold for selecting the first plateau dimension")
    parser.add_argument("--sw-L", type=int, default=256,
                        help="Number of random directions for the residual SW term in CSS")
    parser.add_argument("--sw-batch", type=int, default=64)
    parser.add_argument("--full-sw-L", type=int, default=1024,
                        help="Number of random directions for the full d*SW baseline")
    parser.add_argument("--full-sw-batch", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--sinkhorn-inner", type=int, default=3,
                        help="Set to 1 to match Huang et al. RBCD exactly")
    parser.add_argument("--sinkhorn-max-iter", type=int, default=400)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-7)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--solver", choices=["rbcd", "rabcd"], default="rbcd")
    parser.add_argument("--reset-dual-across-k", action="store_true",
                        help="If set, do not reuse Sinkhorn dual state across different k")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True,
                        help="Print per-dimension progress during each trial")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable per-dimension progress printing")
    parser.add_argument("--out", type=str, default="css_wpp_fullsw_baseline.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    # Match the current trade-off experiment setup.
    d = 30
    r = 5
    E_total = 120.0
    rho = 0.10
    eta_sig = 1.0
    alphas = [0.0, 1.0, 1.5]

    n = args.n
    n_trials = args.n_trials
    k_values = np.arange(1, args.k_max + 1)
    if args.k_min < 1 or args.k_min > args.k_max:
        raise ValueError("--k-min must be between 1 and --k-max.")

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
        f"Backend: {rbcd.backend_name}, dtype={rbcd.dtype_np}, n={n}, trials={n_trials}, "
        f"solver={args.solver}, carry_dual_across_k={not args.reset_dual_across_k}"
    )

    E_sig = (1.0 - rho) * E_total
    E_res = rho * E_total
    tau_sig = build_tau_signal_exp_decay(r, E_sig, eta_sig)

    taus: dict[float, np.ndarray] = {}
    true_w2: dict[float, float] = {}
    for alpha in alphas:
        tau_res = build_tau_residual_powerlaw(d - r, E_res, alpha)
        tau = np.concatenate([tau_sig, tau_res])
        taus[alpha] = tau
        true_w2[alpha] = float(np.sum(tau))

    Q_basis = fixed_orthogonal_basis(d, seed=12345)
    sigmas = make_sigmas_from_taus(d, taus, Q_basis)

    raw = {alpha: {"wpp": [], "css": [], "full_dsw": []} for alpha in alphas}

    print(f"Running {n_trials} trials ...")
    overall_tic = time.perf_counter()
    for t in range(n_trials):
        rngX = np.random.default_rng(10000 + t)
        X = rngX.standard_normal((n, d)).astype(dtype, copy=False)
        Xb = rbcd.asarray(X)

        for alpha in alphas:
            trial_tic = time.perf_counter()
            print(f"[trial {t + 1}/{n_trials} | alpha={alpha}] start", flush=True)
            Sigma = sigmas[alpha]
            rngY = np.random.default_rng(20000 + 97 * t + int(100 * alpha))
            Y = sample_gaussian_target(Sigma, n, rngY).astype(dtype, copy=False)
            Yb = rbcd.asarray(Y)

            wpp_hat, css_hat, full_dsw = run_single_trial_estimators(
                X=Xb,
                Y=Yb,
                k_values=k_values,
                rbcd=rbcd,
                sw_L=args.sw_L,
                full_sw_L=args.full_sw_L,
                seed=30000 + 999 * t + int(100 * alpha),
                sw_batch=args.sw_batch,
                full_sw_batch=args.full_sw_batch,
                solver=args.solver,
                carry_dual_across_k=not args.reset_dual_across_k,
                progress_label=f"[trial {t + 1}/{n_trials} | alpha={alpha}]",
                progress=args.progress,
            )
            raw[alpha]["wpp"].append(wpp_hat)
            raw[alpha]["css"].append(css_hat)
            raw[alpha]["full_dsw"].append(float(full_dsw))
            print(
                f"[trial {t + 1}/{n_trials} | alpha={alpha}] done in "
                f"{time.perf_counter() - trial_tic:.1f}s",
                flush=True,
            )

            gc.collect()
            rbcd.free_backend_memory()
    print(f"All trials completed in {time.perf_counter() - overall_tic:.1f}s", flush=True)

    results = {}
    for alpha in alphas:
        W_emp_wpp = np.array(raw[alpha]["wpp"], dtype=float)
        W_emp_css = np.array(raw[alpha]["css"], dtype=float)
        W_emp_full = np.array(raw[alpha]["full_dsw"], dtype=float)
        wpp_mean = np.mean(W_emp_wpp, axis=0)

        results[alpha] = {
            "wpp_mean": wpp_mean,
            "wpp_std": np.std(W_emp_wpp, axis=0, ddof=0),
            "css_mean": np.mean(W_emp_css, axis=0),
            "css_std": np.std(W_emp_css, axis=0, ddof=0),
            "full_dsw_mean": float(np.mean(W_emp_full)),
            "full_dsw_std": float(np.std(W_emp_full, ddof=0)),
            "true_w2": true_w2[alpha],
        }
        if args.select_k_plateau:
            selected_k, _, wpp_rel_gains = select_plateau_k(
                k_values=k_values,
                wpp_values=wpp_mean,
                k_min=args.k_min,
                plateau_rel_tol=args.plateau_rel_tol,
            )
            results[alpha]["selected_k"] = selected_k
            results[alpha]["wpp_rel_gains"] = wpp_rel_gains

    plot_estimator_graph(
        results,
        k_values,
        save_path=args.out,
        show_selected_k=args.select_k_plateau,
    )

    out_npz = str(Path(args.out).with_suffix('.npz'))
    out_csv = str(Path(args.out).with_suffix('.csv'))

    flat = {"k_values": k_values.astype(int)}
    for alpha in alphas:
        flat[f"alpha_{alpha}_true_w2"] = np.array([true_w2[alpha]], dtype=float)
        flat[f"alpha_{alpha}_wpp_emp"] = np.array(raw[alpha]["wpp"], dtype=float)
        flat[f"alpha_{alpha}_css_emp"] = np.array(raw[alpha]["css"], dtype=float)
        flat[f"alpha_{alpha}_full_dsw_emp"] = np.array(raw[alpha]["full_dsw"], dtype=float)
        if args.select_k_plateau:
            flat[f"alpha_{alpha}_selected_k"] = np.array([results[alpha]["selected_k"]], dtype=int)
            flat[f"alpha_{alpha}_wpp_rel_gains"] = np.array(results[alpha]["wpp_rel_gains"], dtype=float)
    np.savez_compressed(out_npz, **flat)
    print(f"Saved: {out_npz}")

    save_summary_csv(results, k_values, out_csv, include_selected_k=args.select_k_plateau)
    print_one_line_summary(results, k_values, include_selected_k=args.select_k_plateau)


if __name__ == "__main__":
    main()
