import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from RiemannianBCD import RiemannianBlockCoordinateDescent


# -------------------------
# 1) Tau & model design
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
    out = {}
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
# 2) Population calculations
# -------------------------

def population_wpp_from_tau(tau: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    tau_sorted = np.sort(tau)[::-1]
    csum = np.cumsum(tau_sorted)
    out = np.empty_like(k_values, dtype=float)
    for idx, k in enumerate(k_values):
        kk = int(k)
        if kk <= 0:
            out[idx] = 0.0
        else:
            out[idx] = csum[min(kk, len(csum)) - 1]
    return out


def sw2_pop_diag(s_diag: np.ndarray, n_mc: int, rng: np.random.Generator) -> float:
    """Monte Carlo sliced W2^2 for N(0, I) vs N(0, diag(s_diag^2))."""
    m = len(s_diag)
    if m == 0:
        return 0.0
    # If theta is uniform on S^{m-1}, then theta_i^2 has Dirichlet(1/2,...,1/2).
    g = rng.gamma(shape=0.5, scale=1.0, size=(n_mc, m))
    w = g / np.sum(g, axis=1, keepdims=True)
    val = np.sum(w * (s_diag ** 2), axis=1)
    return float(np.mean((np.sqrt(val) - 1.0) ** 2))


def population_lb_from_tau(tau: np.ndarray, d: int, k_values: np.ndarray, n_mc_sw: int, seed: int) -> np.ndarray:
    tau_sorted = np.sort(tau)[::-1]
    wpp = population_wpp_from_tau(tau, k_values)
    lb = np.zeros_like(wpp)

    base_rng = np.random.default_rng(seed)
    for idx, k in enumerate(k_values):
        kk = int(k)
        if kk >= d:
            lb[idx] = wpp[idx]
            continue
        tau_res = tau_sorted[kk:]
        s_res = 1.0 + np.sqrt(np.maximum(tau_res, 0.0))
        rng_k = np.random.default_rng(base_rng.integers(0, 2**32 - 1) + kk)
        sw2 = sw2_pop_diag(s_res, n_mc=n_mc_sw, rng=rng_k)
        lb[idx] = wpp[idx] + (d - kk) * sw2
    return lb


# -------------------------
# 3) Finite-sample helpers
# -------------------------

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


def sliced_w2_residual_batched(
    X_perp,
    Y_perp,
    L: int,
    seed: int,
    batch_size: int = 64,
    rbcd: RiemannianBlockCoordinateDescent | None = None,
) -> float:
    if rbcd is None or not rbcd.use_gpu:
        rng = np.random.default_rng(seed)
        X_perp = np.asarray(X_perp)
        Y_perp = np.asarray(Y_perp)
        _, m = X_perp.shape
        if m == 0:
            return 0.0
        total = 0.0
        done = 0
        while done < L:
            b = min(batch_size, L - done)
            Theta = rng.standard_normal((m, b))
            Theta /= np.linalg.norm(Theta, axis=0, keepdims=True) + 1e-12
            PX = X_perp @ Theta
            PY = Y_perp @ Theta
            PX.sort(axis=0)
            PY.sort(axis=0)
            total += float(np.sum(np.mean((PX - PY) ** 2, axis=0)))
            done += b
        return total / float(L)

    xp = rbcd.xp
    rs = xp.random.RandomState(seed)
    X_perp = rbcd.asarray(X_perp)
    Y_perp = rbcd.asarray(Y_perp)
    _, m = X_perp.shape
    if m == 0:
        return 0.0

    total = 0.0
    done = 0
    while done < L:
        b = min(batch_size, L - done)
        Theta = rs.standard_normal((m, b)).astype(X_perp.dtype, copy=False)
        Theta /= xp.linalg.norm(Theta, axis=0, keepdims=True) + 1e-12
        PX = X_perp @ Theta
        PY = Y_perp @ Theta
        PX = xp.sort(PX, axis=0)
        PY = xp.sort(PY, axis=0)
        total += rbcd.to_float(xp.sum(xp.mean((PX - PY) ** 2, axis=0)))
        done += b
    return total / float(L)



def run_single_trial_curve(
    X,
    Y,
    k_values: np.ndarray,
    rbcd: RiemannianBlockCoordinateDescent,
    sw_L: int,
    seed: int,
    sw_batch: int = 64,
):
    n, d = X.shape
    a = rbcd.asarray(np.full(n, 1.0 / n, dtype=rbcd.dtype_np))
    b = rbcd.asarray(np.full(n, 1.0 / n, dtype=rbcd.dtype_np))

    wpp_hat = np.zeros(len(k_values), dtype=float)
    lb_hat = np.zeros(len(k_values), dtype=float)

#    U = None
#    state = None
#    for idx, k in enumerate(k_values):
#        kk = int(k)
#       if U is None:
#           U = rbcd.InitialStiefel(d, kk)
#       else:
#            U = extend_U_warmstart(U, seed=seed + 17 * kk).astype(rbcd.dtype_np, copy=False)
#
#        _, U, _, _, _, state = rbcd.run_RBCD(
#            a, b, X, Y, k, U, warm_state=state, return_state=True
#        )
#
#        _, _, _, f_val, _, state = rbcd.run_fixed_projection_ot(
#            a, b, X, Y, U=U, warm_state=state, return_state=True
#       )
#       wpp_hat[idx] = float(f_val)

    U = None
    for idx, k in enumerate(k_values):
        kk = int(k)
        if U is None:
            U = rbcd.InitialStiefel(d, kk, seed=seed + 31 * kk)
        else:
            U = extend_U_warmstart(U, seed=seed + 17 * kk).astype(rbcd.dtype_np, copy=False)

        # 关键：每个新的 k 都重置 dual state
        _, U, _, _, _, state_k = rbcd.run_RBCD(
            a, b, X, Y, k, U, warm_state=None, return_state=True
        )

        # 只在同一个 k 内，用 state_k 做 fixed-U polished rescoring
        _, _, _, f_val, _, _ = rbcd.run_fixed_projection_ot(
            a, b, X, Y, U=U, warm_state=state_k, return_state=True
        )

        wpp_hat[idx] = float(f_val)


        if kk < d:
            U_perp = orthogonal_complement(U).astype(rbcd.dtype_np, copy=False)
            U_perp_b = rbcd.asarray(U_perp)
            Xp = X @ U_perp_b
            Yp = Y @ U_perp_b
            sw2 = sliced_w2_residual_batched(
                Xp, Yp, L=sw_L, seed=seed + 2024 + 101 * kk, batch_size=sw_batch, rbcd=rbcd
            )
            lb_hat[idx] = wpp_hat[idx] + (d - kk) * sw2
        else:
            lb_hat[idx] = wpp_hat[idx]

    return wpp_hat, lb_hat


# -------------------------
# 4) Plotting
# -------------------------

def plot_trade_off_graph(results, k_values, save_path: str):
    alphas = sorted(list(results.keys()))
    fig, axes = plt.subplots(2, len(alphas), figsize=(6 * len(alphas), 9), squeeze=False)

    color_pop = "tab:blue"
    color_fs = "tab:orange"
    color_total = "tab:green"

    for i, alpha in enumerate(alphas):
        data = results[alpha]

        ax_wpp = axes[0, i]
        ax_wpp.plot(k_values, data["wpp"]["pop_err"], linestyle=":", linewidth=2.2,
                    color=color_pop, label="Population Error (Bias)")
        ax_wpp.plot(k_values, data["wpp"]["fs_err"], linestyle="--", linewidth=2.2,
                    color=color_fs, label="Finite Sample Error")
        ax_wpp.plot(k_values, data["wpp"]["total_err"], linestyle="-", linewidth=2.2,
                    color=color_total, label="Total Error")
        ax_wpp.set_title(f"WPP Trade-off ($\\alpha={alpha}$)")
        ax_wpp.set_xlabel("Subspace Dimension $k^\\star$")
        ax_wpp.set_ylabel("Absolute Error")
        ax_wpp.grid(True, alpha=0.3)
        ax_wpp.set_ylim(bottom=0)
        if i == 0:
            ax_wpp.legend()

        ax_lb = axes[1, i]
        ax_lb.plot(k_values, data["lb"]["pop_err"], linestyle=":", linewidth=2.2,
                   color=color_pop, label="Population Error (Bias)")
        ax_lb.plot(k_values, data["lb"]["fs_err"], linestyle="--", linewidth=2.2,
                   color=color_fs, label="Finite Sample Error")
        ax_lb.plot(k_values, data["lb"]["total_err"], linestyle="-", linewidth=2.2,
                   color=color_total, label="Total Error")
        ax_lb.set_title(f"LB Trade-off ($\\alpha={alpha}$)")
        ax_lb.set_xlabel("Subspace Dimension $k^\\star$")
        ax_lb.set_ylabel("Absolute Error")
        ax_lb.grid(True, alpha=0.3)
        ax_lb.set_ylim(bottom=0)
        if i == 0:
            ax_lb.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {save_path}")


# -------------------------
# 5) Main execution
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Finite-sample trade-off experiment (GPU + corrected version)")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy/GPU backend")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--sw-L", type=int, default=256)
    parser.add_argument("--sw-batch", type=int, default=64)
    parser.add_argument("--pop-sw-mc", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--sinkhorn-inner", type=int, default=2,
                        help="Set to 1 to match Huang et al. RBCD.")
    parser.add_argument("--sinkhorn-max-iter", type=int, default=400)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-7)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--out", type=str, default="finite_sample_tradeoff_gpu_fast2_fixed.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    d = 60
    r = 10
    E_total = 120.0
    rho = 0.10  # paper setting: residual energy is 10% of total
    eta_sig = 1.0
    alphas = [0.0, 1.0, 1.5]

    n = args.n
    n_trials = args.n_trials
    k_values = np.arange(1, args.k_max + 1)
    sw_L = args.sw_L
    sw_batch = args.sw_batch
    pop_sw_mc = args.pop_sw_mc

    rbcd = RiemannianBlockCoordinateDescent(
        eta=2.0,
        tau=0.07,
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
    print(f"Backend: {rbcd.backend_name}, dtype={rbcd.dtype_np}, n={n}, trials={n_trials}")

    E_sig = (1.0 - rho) * E_total
    E_res = rho * E_total
    tau_sig = build_tau_signal_exp_decay(r, E_sig, eta_sig)

    taus = {}
    pop_curves = {}
    print("Computing population curves...")
    for alpha in alphas:
        tau_res = build_tau_residual_powerlaw(d - r, E_res, alpha)
        tau = np.concatenate([tau_sig, tau_res])
        taus[alpha] = tau

        wpp_pop = population_wpp_from_tau(tau, k_values)
        lb_pop = population_lb_from_tau(
            tau, d, k_values, n_mc_sw=pop_sw_mc, seed=777 + int(100 * alpha)
        )
        pop_curves[alpha] = {
            "wpp": wpp_pop,
            "lb": lb_pop,
            "true_w2": float(np.sum(tau)),
        }

    Q_basis = fixed_orthogonal_basis(d, seed=12345)
    sigmas = make_sigmas_from_taus(d, taus, Q_basis)

    raw_emp = {alpha: {"wpp": [], "lb": []} for alpha in alphas}

    print(f"Running {n_trials} trials ...")
    for t in range(n_trials):
        rngX = np.random.default_rng(10000 + t)
        X = rngX.standard_normal((n, d)).astype(dtype, copy=False)
        Xb = rbcd.asarray(X)

        for alpha in alphas:
            Sigma = sigmas[alpha]
            rngY = np.random.default_rng(20000 + 97 * t + int(100 * alpha))
            Y = sample_gaussian_target(Sigma, n, rngY).astype(dtype, copy=False)
            Yb = rbcd.asarray(Y)

            wpp_hat, lb_hat = run_single_trial_curve(
                X=Xb,
                Y=Yb,
                k_values=k_values,
                rbcd=rbcd,
                sw_L=sw_L,
                seed=30000 + 999 * t + int(100 * alpha),
                sw_batch=sw_batch,
            )
            raw_emp[alpha]["wpp"].append(wpp_hat)
            raw_emp[alpha]["lb"].append(lb_hat)

            gc.collect()
            rbcd.free_backend_memory()

    results = {}
    for alpha in alphas:
        W2_true = pop_curves[alpha]["true_w2"]
        W_pop_wpp = pop_curves[alpha]["wpp"]
        W_pop_lb = pop_curves[alpha]["lb"]

        W_emp_wpp = np.array(raw_emp[alpha]["wpp"])
        W_emp_lb = np.array(raw_emp[alpha]["lb"])

        pop_err_wpp = np.abs(W_pop_wpp - W2_true)
        pop_err_lb = np.abs(W_pop_lb - W2_true)

        fs_err_wpp = np.mean(np.abs(W_emp_wpp - W_pop_wpp[None, :]), axis=0)
        fs_err_lb = np.mean(np.abs(W_emp_lb - W_pop_lb[None, :]), axis=0)

        total_err_wpp = np.mean(np.abs(W_emp_wpp - W2_true), axis=0)
        total_err_lb = np.mean(np.abs(W_emp_lb - W2_true), axis=0)

        results[alpha] = {
            "wpp": {
                "pop_err": pop_err_wpp,
                "fs_err": fs_err_wpp,
                "total_err": total_err_wpp,
            },
            "lb": {
                "pop_err": pop_err_lb,
                "fs_err": fs_err_lb,
                "total_err": total_err_lb,
            },
        }

    plot_trade_off_graph(results, k_values, save_path=args.out)

    # Also save the raw arrays for later manuscript integration / reproducibility.
    out_npz = str(Path(args.out).with_suffix('.npz'))
    flat = {"k_values": k_values.astype(int)}
    for alpha in alphas:
        flat[f"alpha_{alpha}_true_w2"] = np.array([pop_curves[alpha]["true_w2"]], dtype=float)
        flat[f"alpha_{alpha}_wpp_pop"] = pop_curves[alpha]["wpp"]
        flat[f"alpha_{alpha}_lb_pop"] = pop_curves[alpha]["lb"]
        flat[f"alpha_{alpha}_wpp_emp"] = np.array(raw_emp[alpha]["wpp"])
        flat[f"alpha_{alpha}_lb_emp"] = np.array(raw_emp[alpha]["lb"])
    np.savez_compressed(out_npz, **flat)
    print(f"Saved: {out_npz}")


if __name__ == "__main__":
    main()
