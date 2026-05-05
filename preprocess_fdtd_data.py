"""
Preprocess the FDTD reference into normalized (u, q, g) training data.

Run ONCE before main_transcranial.py:
    python preprocess_fdtd_data.py

The first-order system variables are:
    q = du/dtau  =  du/dt * t_ref          [same units as u = metres]
    g = kappa_hat(xhat) * du/dxhat         [same units as u = metres]

These are computed from FDTD u at FULL time resolution (no subsampling first),
then subsampled. This avoids noise amplification from differencing coarse data.

Outputs pinn_training_full_normalized.npz:
    tau, xhat            float32 dimensionless coordinates
    u, q, g              float32 state variables, each normalised to O(1) RMS
    u_scale, q_scale, g_scale   float32 scale factors (multiply to recover metres)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def preprocess(
    fdtd_npz: str | Path = "transcranial_fdtd1d_reference_full.npz",
    out_dir: str | Path = ".",
    subsample_t: int = 10,  # every 10th step: 50 ns → 40 samples/carrier-period (safe)
    subsample_x: int = 4,   # every 4th node:  ~133 µm → 23 samples/wavelength in brain
    phys=None,
) -> tuple[float, float, float]:

    if phys is None:
        from lib_transcranial.physics import TranscranialHybridPhysics
        phys = TranscranialHybridPhysics()

    print(f"Loading {fdtd_npz} …")
    data  = np.load(fdtd_npz)
    x_mm  = data["x_mm"].astype(np.float64)
    t_us  = data["t_us"].astype(np.float64)
    u_raw = data["u"].astype(np.float64)

    t_s = t_us * 1e-6
    x_m = x_mm * 1e-3

    # Dimensionless coordinates
    tau  = phys.s_to_tau(t_s)
    xhat = phys.mm_to_xhat(x_mm)

    Nt, Nx1 = u_raw.shape
    dt_s = t_s[1] - t_s[0]
    dx_m = x_m[1] - x_m[0]
    print(f"  FDTD grid: Nt={Nt}, Nx+1={Nx1}, "
          f"dt={dt_s*1e9:.1f} ns, dx={dx_m*1e6:.1f} µm")
    print(f"  u: peak={np.max(np.abs(u_raw)):.3e} m, "
          f"rms={np.sqrt(np.mean(u_raw**2)):.3e} m")

    # q = du/dtau = du/dt * t_ref
    # np.gradient with the actual t_s array handles non-uniform spacing
    print("  Computing q = du/dtau …")
    dudt  = np.gradient(u_raw, t_s, axis=0)   # m/s
    q_raw = dudt * phys.t_ref                 # metres  (tau is dimensionless)
    # ------------------------------------------------------------------
    # g = kappa_hat(xhat) * du/dxhat
    # du/dxhat = du/dx * L
    # ------------------------------------------------------------------
    print("  Computing g = kappa_hat * du/dxhat …")
    dudx     = np.gradient(u_raw, x_m, axis=1)
    du_dxhat = dudx * phys.L_m
    kappa_hat = phys.kappa_hat_np(xhat)
    g_raw    = kappa_hat[None, :] * du_dxhat

    u_rms = float(np.sqrt(np.mean(u_raw ** 2)))
    q_rms = float(np.sqrt(np.mean(q_raw ** 2)))
    g_rms = float(np.sqrt(np.mean(g_raw ** 2)))
    print(f"  q: rms={q_rms:.3e} m  (ratio q/u = {q_rms/u_rms:.1f}, expected ~149)")
    print(f"  g: rms={g_rms:.3e} m  (ratio g/u = {g_rms/u_rms:.1f}, expected ~149)")
    # ------------------------------------------------------------------
    # Subsample (after derivative computation — not before)
    # ------------------------------------------------------------------
    tau_s  = tau [::subsample_t].astype(np.float32)
    xhat_s = xhat[::subsample_x].astype(np.float32)
    u_s = u_raw[::subsample_t, ::subsample_x].astype(np.float32)
    q_s = q_raw[::subsample_t, ::subsample_x].astype(np.float32)
    g_s = g_raw[::subsample_t, ::subsample_x].astype(np.float32)
    print(f"  After subsampling (t×{subsample_t}, x×{subsample_x}): "
          f"Nt={tau_s.size}, Nx={xhat_s.size}, "
          f"total={tau_s.size * xhat_s.size:,} points")
    # ------------------------------------------------------------------
    # RMS-based scale factors (robust, positive, finite)
    # ------------------------------------------------------------------
    def rms_scale(arr: np.ndarray) -> float:
        v = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        if not (np.isfinite(v) and v > 0.0):
            v = float(np.max(np.abs(arr)))
        return max(v, 1e-30)

    u_scale = rms_scale(u_s)
    q_scale = rms_scale(q_s)
    g_scale = rms_scale(g_s)

    print(f"\nScale factors:")
    print(f"  u_scale = {u_scale:.6e} m")
    print(f"  q_scale = {q_scale:.6e} m  (q/u = {q_scale/u_scale:.1f}x)")
    print(f"  g_scale = {g_scale:.6e} m  (g/u = {g_scale/u_scale:.1f}x)")
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "pinn_training_full_normalized.npz"
    np.savez(
        out_path,
        tau     = tau_s,
        xhat    = xhat_s,
        u       = (u_s / u_scale).astype(np.float32),
        q       = (q_s / q_scale).astype(np.float32),
        g       = (g_s / g_scale).astype(np.float32),
        u_scale = np.float32(u_scale),
        q_scale = np.float32(q_scale),
        g_scale = np.float32(g_scale),
    )
    print(f"\nSaved: {out_path}")
    return u_scale, q_scale, g_scale

if __name__ == "__main__":
    preprocess()