# -*- coding: utf-8 -*-
"""Faithful GPU/CPU implementation of Huang et al. (2021) RBCD for PRW.

This version keeps the algebraic speedups of the previous GPU file but fixes two
practical issues in the original fast draft:

1. The returned (U, f_val, state) are now *self-consistent*: they are evaluated
   at the same final subspace U. The previous draft could return a subspace after
   one last retraction step together with a transport plan / objective computed at
   the pre-update subspace.
2. The default RBCD behavior matches Huang et al. (2021): one Sinkhorn-style
   scaling update per outer RBCD iteration (`sinkhorn_inner=1`).

The core formulas remain the same as in the author code; they are simply written
in projected coordinates XU, YU to avoid forming U U^T explicitly and to support
NumPy/CuPy backends.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


class RiemannianBlockCoordinateDescent:
    def __init__(
        self,
        eta,
        tau,
        max_iter,
        threshold,
        verbose: bool = False,
        use_gpu: bool = False,
        gpu_device: int = 0,
        sinkhorn_max_iter: int = 400,
        sinkhorn_tol: float = 1e-7,
        sinkhorn_inner: int = 1,
        dtype=np.float32,
        store_pi: bool = False,
        max_store_plan: int = 2_000_000,
        strict_gpu: bool = True,
    ):
        assert eta >= 0
        if tau is not None:
            assert tau > 0
        assert isinstance(max_iter, int) and max_iter > 0
        assert threshold > 0
        assert isinstance(verbose, bool)
        assert sinkhorn_max_iter > 0
        assert sinkhorn_tol > 0
        assert sinkhorn_inner > 0

        self.eta = float(eta)
        self.tau = float(tau) if tau is not None else None
        self.max_iter = int(max_iter)
        self.threshold = float(threshold)
        self.verbose = bool(verbose)
        self.requested_gpu = bool(use_gpu)
        self.gpu_device = int(gpu_device)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.sinkhorn_inner = int(sinkhorn_inner)
        self.dtype_np = np.dtype(dtype)
        self.store_pi = bool(store_pi)
        self.max_store_plan = int(max_store_plan)
        self.strict_gpu = bool(strict_gpu)
        self._eps = 1e-20 if self.dtype_np == np.float32 else 1e-30

        if self.requested_gpu:
            if cp is None:
                msg = (
                    "use_gpu=True but CuPy is not installed. Install a CuPy build matching "
                    "your CUDA version, or set use_gpu=False."
                )
                if self.strict_gpu:
                    raise ImportError(msg)
                print(f"[RBCD] Warning: {msg} Falling back to NumPy/CPU.")
                self.use_gpu = False
                self.xp = np
            else:
                cp.cuda.Device(self.gpu_device).use()
                self.use_gpu = True
                self.xp = cp
        else:
            self.use_gpu = False
            self.xp = np

        if self.use_gpu:
            self.dtype = cp.float32 if self.dtype_np == np.float32 else cp.float64
            self.high_dtype = cp.float64
        else:
            self.dtype = self.dtype_np
            self.high_dtype = np.float64

    @property
    def backend_name(self) -> str:
        return "cupy" if self.use_gpu else "numpy"

    def synchronize(self) -> None:
        if self.use_gpu:
            cp.cuda.Device(self.gpu_device).synchronize()

    def free_backend_memory(self) -> None:
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def asarray(self, arr):
        if self.use_gpu:
            if isinstance(arr, cp.ndarray):
                return arr.astype(self.dtype, copy=False)
            return cp.asarray(arr, dtype=self.dtype)
        return np.asarray(arr, dtype=self.dtype_np)

    def to_numpy(self, arr):
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def to_float(self, scalar) -> float:
        if self.use_gpu and isinstance(scalar, cp.ndarray):
            return float(cp.asnumpy(scalar))
        return float(scalar)

    def InitialStiefel(self, d: int, k: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((d, k))
        q, _ = np.linalg.qr(U)
        return q.astype(self.dtype_np, copy=False)

    def StiefelRetraction(self, U, G):
        q, _ = self.xp.linalg.qr(U + G)
        return q.astype(self.dtype, copy=False)

    def StiefelGradientProj(self, G, U):
        temp = G.T @ U
        return G - U @ ((temp + temp.T) / 2.0)

    def _kernel_from_projected(self, XU, YU):
        xp = self.xp
        XU = self.asarray(XU)
        YU = self.asarray(YU)
        x2 = xp.sum(XU * XU, axis=1, dtype=self.high_dtype).astype(self.dtype, copy=False)
        y2 = xp.sum(YU * YU, axis=1, dtype=self.high_dtype).astype(self.dtype, copy=False)
        K = x2[:, None] + y2[None, :] - 2.0 * (XU @ YU.T)
        xp.maximum(K, 0.0, out=K)
        K *= (-1.0 / self.eta)
        xp.exp(K, out=K)
        return K

    def _sinkhorn_update(self, K, a, b, u, v, num_iter: int):
        xp = self.xp
        for _ in range(int(num_iter)):
            Kv = K @ v
            u = a / xp.maximum(Kv, self._eps)
            KTu = K.T @ u
            v = b / xp.maximum(KTu, self._eps)
        return u, v

    def _sinkhorn_to_convergence(self, K, a, b, u=None, v=None, max_iter=None, tol=None, check_every=10):
        xp = self.xp
        a = self.asarray(a)
        b = self.asarray(b)
        n = int(a.shape[0])
        m = int(b.shape[0])
        if u is None or u.shape[0] != n:
            u = xp.full(n, 1.0 / max(n, 1), dtype=self.dtype)
        else:
            u = self.asarray(u).copy()
        if v is None or v.shape[0] != m:
            v = xp.full(m, 1.0 / max(m, 1), dtype=self.dtype)
        else:
            v = self.asarray(v).copy()

        max_iter = self.sinkhorn_max_iter if max_iter is None else int(max_iter)
        tol = self.sinkhorn_tol if tol is None else float(tol)
        err = np.inf
        it = 0
        while it < max_iter:
            inner = min(check_every, max_iter - it)
            u, v = self._sinkhorn_update(K, a, b, u, v, inner)
            it += inner
            row = u * (K @ v)
            col = v * (K.T @ u)
            err_row = self.to_float(xp.max(xp.abs(row - a)))
            err_col = self.to_float(xp.max(xp.abs(col - b)))
            err = max(err_row, err_col)
            if err < tol:
                break
        return u, v, it, err

    def _maybe_plan(self, K, u, v):
        if self.store_pi and K.size <= self.max_store_plan:
            pi = u[:, None] * K * v[None, :]
            return self.to_numpy(pi)
        return None

    def _transport_cost_from_projected(self, XU, YU, a, colsum, T1) -> float:
        xp = self.xp
        x2 = xp.sum(XU * XU, axis=1, dtype=self.high_dtype)
        y2 = xp.sum(YU * YU, axis=1, dtype=self.high_dtype)
        val = (
            xp.sum(a.astype(self.high_dtype, copy=False) * x2)
            + xp.sum(colsum.astype(self.high_dtype, copy=False) * y2)
            - 2.0 * xp.sum(XU.astype(self.high_dtype, copy=False) * T1.astype(self.high_dtype, copy=False))
        )
        return self.to_float(val)

    def _evaluate_current_subspace(self, a, b, X, Y, XT, YT, U, u=None, v=None):
        xp = self.xp
        n = int(X.shape[0])
        m = int(Y.shape[0])
        if u is None or u.shape[0] != n:
            u = xp.full(n, 1.0 / max(n, 1), dtype=self.dtype)
        else:
            u = self.asarray(u)
        if v is None or v.shape[0] != m:
            v = xp.full(m, 1.0 / max(m, 1), dtype=self.dtype)
        else:
            v = self.asarray(v)

        XU = X @ U
        YU = Y @ U
        K = self._kernel_from_projected(XU, YU)
        u, v = self._sinkhorn_update(K, a, b, u, v, self.sinkhorn_inner)

        colsum = v * (K.T @ u)
        T1 = u[:, None] * (K @ (v[:, None] * YU))
        T2 = v[:, None] * (K.T @ (u[:, None] * XU))

        term_x = XT @ (a[:, None] * XU)
        term_y = YT @ (colsum[:, None] * YU)
        cross1 = XT @ T1
        cross2 = YT @ T2
        VU = term_x + term_y - cross1 - cross2

        G = (2.0 / self.eta) * VU
        xi = self.StiefelGradientProj(G, U)
        grad_norm = self.to_float(xp.linalg.norm(xi))
        f_val = self._transport_cost_from_projected(XU, YU, a, colsum, T1)

        return {
            "K": K,
            "u": u,
            "v": v,
            "colsum": colsum,
            "VU": VU,
            "xi": xi,
            "grad_norm": grad_norm,
            "f_val": f_val,
        }

    def run_fixed_projection_ot(self, a, b, X, Y, U=None, warm_state=None, return_state=False):
        self.synchronize()
        tic = time.perf_counter()
        X = self.asarray(X)
        Y = self.asarray(Y)
        a = self.asarray(a)
        b = self.asarray(b)
        n, d = X.shape
        _, d2 = Y.shape
        assert d == d2

        if U is None:
            XU = X
            YU = Y
            U_backend = self.asarray(np.eye(d, dtype=self.dtype_np))
        else:
            U_backend = self.asarray(U)
            XU = X @ U_backend
            YU = Y @ U_backend

        K = self._kernel_from_projected(XU, YU)
        u0 = None if warm_state is None else warm_state.get("u")
        v0 = None if warm_state is None else warm_state.get("v")
        u, v, it, err = self._sinkhorn_to_convergence(K, a, b, u=u0, v=v0)

        colsum = v * (K.T @ u)
        T1 = u[:, None] * (K @ (v[:, None] * YU))
        f_val = self._transport_cost_from_projected(XU, YU, a, colsum, T1)
        pi = self._maybe_plan(K, u, v)

        self.synchronize()
        toc = time.perf_counter()
        if self.verbose:
            print(
                f"Fixed-U Sinkhorn[{self.backend_name}]: it={it}, err={err:.3e}, "
                f"time={toc - tic:.3f}s, fval={f_val:.6f}"
            )
        state = {"u": u, "v": v, "err": err, "iters": it}
        U_cpu = self.to_numpy(U_backend)
        if return_state:
            return pi, U_cpu, toc - tic, f_val, it, state
        return pi, U_cpu, toc - tic, f_val, it

    def run_RBCD(self, a, b, X, Y, k, U, warm_state=None, return_state=False):
        xp = self.xp
        self.synchronize()
        X = self.asarray(X)
        Y = self.asarray(Y)
        a = self.asarray(a)
        b = self.asarray(b)
        U_backend = self.asarray(U)

        n, d = X.shape
        m, d2 = Y.shape
        assert d == d2
        assert U_backend.shape == (d, k)

        XT = X.T
        YT = Y.T
        step_size = self.tau
        elapsed = 0.0
        num_updates = 0

        u = None if warm_state is None else warm_state.get("u")
        v = None if warm_state is None else warm_state.get("v")
        if u is not None and u.shape[0] != n:
            u = None
        if v is not None and v.shape[0] != m:
            v = None

        last_eval = None
        while True:
            self.synchronize()
            tic = time.perf_counter()
            last_eval = self._evaluate_current_subspace(a, b, X, Y, XT, YT, U_backend, u=u, v=v)
            u, v = last_eval["u"], last_eval["v"]
            grad_norm = last_eval["grad_norm"]

            stop = (self.eta * grad_norm <= self.threshold) or (num_updates >= self.max_iter)
            if not stop:
                U_backend = self.StiefelRetraction(U_backend, step_size * last_eval["xi"])
                num_updates += 1

            self.synchronize()
            toc = time.perf_counter()
            elapsed += toc - tic
            if stop:
                break

        assert last_eval is not None
        pi = self._maybe_plan(last_eval["K"], last_eval["u"], last_eval["v"])
        U_cpu = self.to_numpy(U_backend)
        if self.verbose:
            print(
                f"RBCD[{self.backend_name}]: iter={num_updates}, eta*grad={self.eta * grad_norm:.3e}, "
                f"time={elapsed:.3f}s, fval={last_eval['f_val']:.6f}"
            )
        state = {"u": last_eval["u"], "v": last_eval["v"], "grad_norm": grad_norm, "colsum": last_eval["colsum"]}
        if return_state:
            return pi, U_cpu, elapsed, last_eval["f_val"], num_updates, state
        return pi, U_cpu, elapsed, last_eval["f_val"], num_updates

    def run_RABCD(self, a, b, X, Y, k, U, warm_state=None, return_state=False):
        xp = self.xp
        self.synchronize()
        X = self.asarray(X)
        Y = self.asarray(Y)
        a = self.asarray(a)
        b = self.asarray(b)
        U_backend = self.asarray(U)

        n, d = X.shape
        m, d2 = Y.shape
        assert d == d2
        assert U_backend.shape == (d, k)

        XT = X.T
        YT = Y.T
        alpha = 1e-6
        beta = 0.8
        step_size = self.tau
        elapsed = 0.0
        num_updates = 0

        # Same initialization strategy as the author implementation.
        x2 = xp.sum(X * X, axis=1, dtype=self.high_dtype).astype(self.dtype, copy=False)
        y2 = xp.sum(Y * Y, axis=1, dtype=self.high_dtype).astype(self.dtype, copy=False)
        C = x2[:, None] + y2[None, :] - 2.0 * (X @ Y.T)
        cmax = self.to_float(xp.max(xp.abs(C)))
        p = xp.zeros(d, dtype=self.dtype)
        q = xp.zeros(k, dtype=self.dtype)
        p_hat = (alpha * (cmax ** 2)) * xp.ones(d, dtype=self.dtype)
        q_hat = (alpha * (cmax ** 2)) * xp.ones(k, dtype=self.dtype)

        u = None if warm_state is None else warm_state.get("u")
        v = None if warm_state is None else warm_state.get("v")
        if u is not None and u.shape[0] != n:
            u = None
        if v is not None and v.shape[0] != m:
            v = None

        last_eval = None
        while True:
            self.synchronize()
            tic = time.perf_counter()
            last_eval = self._evaluate_current_subspace(a, b, X, Y, XT, YT, U_backend, u=u, v=v)
            u, v = last_eval["u"], last_eval["v"]
            grad_norm = last_eval["grad_norm"]

            stop = (self.eta * grad_norm <= self.threshold) or (num_updates >= self.max_iter)
            if not stop:
                G_t = self.StiefelGradientProj(-2.0 * last_eval["VU"], U_backend)
                p = beta * p + (1.0 - beta) * xp.diag(G_t @ G_t.T) / k
                p_hat = xp.maximum(p_hat, p)
                q = beta * q + (1.0 - beta) * xp.diag(G_t.T @ G_t) / d
                q_hat = xp.maximum(q_hat, q)
                scaled = xp.diag(xp.power(p_hat, -0.25)) @ G_t @ xp.diag(xp.power(q_hat, -0.25))
                xi = self.StiefelGradientProj(scaled, U_backend)
                U_backend = self.StiefelRetraction(U_backend, -(step_size / self.eta) * xi)
                num_updates += 1

            self.synchronize()
            toc = time.perf_counter()
            elapsed += toc - tic
            if stop:
                break

        assert last_eval is not None
        pi = self._maybe_plan(last_eval["K"], last_eval["u"], last_eval["v"])
        U_cpu = self.to_numpy(U_backend)
        if self.verbose:
            print(
                f"RABCD[{self.backend_name}]: iter={num_updates}, eta*grad={self.eta * grad_norm:.3e}, "
                f"time={elapsed:.3f}s, fval={last_eval['f_val']:.6f}"
            )
        state = {"u": last_eval["u"], "v": last_eval["v"], "grad_norm": grad_norm, "colsum": last_eval["colsum"]}
        if return_state:
            return pi, U_cpu, elapsed, last_eval["f_val"], num_updates, state
        return pi, U_cpu, elapsed, last_eval["f_val"], num_updates
