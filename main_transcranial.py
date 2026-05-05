"""
Main training script for the transcranial hybrid PINN.

Workflow:
  1. Run FDTD solver (auto if NPZ missing):  generates transcranial_fdtd1d_reference_full.npz
  2. Run preprocessor (once):              python preprocess_fdtd_data.py
  3. Run this script:                      python main_transcranial.py

Module map (top-down):
  * set_seed(...)                       deterministic numpy / torch / cuda seeds
  * build_traces(...)                   builds the 5 normalised probe time-series
  * build_stages_normal(phys)           production curriculum: 3 Adam stages + LBFGS
  * build_stages_early_time(...)        time-marching curriculum (alternative)
  * main()                              parse CLI -> physics -> network -> trainer
                                        -> save checkpoint -> render comparison plot

CLI:
  --frequency      carrier f0 in kHz                       (default 500)
  --mode           normal | early_time                     (default normal)
  --snapshot-spacing-us / --points-per-snapshot / --snapshot-window-us
                   early_time curriculum knobs
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from lib_transcranial.physics import TranscranialHybridPhysics
from lib_transcranial.network_transcranial import FourierMLP
from lib_transcranial.pinn_transcranial import FirstOrderHybridPINN
from lib_transcranial.optimiser_transcranial import HybridTrainer, StageConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reference_npz(path: Path):
    d = np.load(path)
    return (
        d["t_us"].astype(np.float32),
        d["x_mm"].astype(np.float32),
        d["u"].astype(np.float32),
    )


def build_traces(
    phys: TranscranialHybridPhysics,
    reference_npz: Path,
    u_scale: float,
    probe_mm: tuple[float, ...] = (1.0, 3.2, 13.2, 40.0, 70.0),
) -> dict:
    """
    Build probe-point traces in NORMALIZED u space.

    traces["u_norm"] is divided by u_scale so that it is O(1) — the same
    scale as model.evaluate_u_normalized().  The old code kept physical units
    (~1e-8), making the trace loss gradient ~1e16x smaller than the state loss.
    """
    t_us, x_mm, u_phys = load_reference_npz(reference_npz)
    tau = phys.s_to_tau(t_us * 1e-6).astype(np.float32)

    probe_idx = [int(np.argmin(np.abs(x_mm - p))) for p in probe_mm]
    return dict(
        tau   = tau,
        xhat  = phys.mm_to_xhat(x_mm[probe_idx]).astype(np.float32),
        u_norm= (u_phys[:, probe_idx] / u_scale).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_wavefield_plot(
    model: FirstOrderHybridPINN,
    phys: TranscranialHybridPhysics,
    device: torch.device,
    nt: int = 600,
    nx: int = 400,
    out_path: str = "wavefield_transcranial.png",
) -> None:
    tau = torch.linspace(0.0, phys.tau_max, nt, device=device)
    x   = torch.linspace(0.0, 1.0,         nx,  device=device)
    u   = model.evaluate_u_on_grid(tau, x, batch_points=40000).cpu().numpy()

    scale = max(float(np.quantile(np.abs(u), 0.995)), 1e-20)
    u_plot = np.clip(u / scale, -1.0, 1.0)

    t_us = phys.tau_to_us(tau.cpu().numpy())
    x_mm = phys.xhat_to_mm(x.cpu().numpy())

    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(
        u_plot.T,
        extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
        aspect="auto", origin="lower", cmap="seismic",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0),
    )
    for mm in phys.x_interfaces_mm:
        ax.axhline(mm, color="gray", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel("t (µs)");  ax.set_ylabel("x (mm)")
    ax.set_title("First-order PINN wavefield")
    plt.colorbar(im, ax=ax).set_label("u (normalized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200);  print(f"Saved: {out_path}")
    plt.close(fig)


@torch.no_grad()
def make_comparison_plot(
    model: FirstOrderHybridPINN,
    phys: TranscranialHybridPhysics,
    reference_npz: Path,
    device: torch.device,
    out_path: str = "comparison_transcranial.png",
    nt: int = 500,
    nx: int = 300,
) -> None:
    """Three-panel plot: FDTD reference | PINN prediction | error."""
    # FDTD reference (subsample for display)
    t_us_fdtd, x_mm_fdtd, u_fdtd = load_reference_npz(reference_npz)
    t_idx = np.linspace(0, len(t_us_fdtd) - 1, nt, dtype=int)
    x_idx = np.linspace(0, len(x_mm_fdtd) - 1, nx, dtype=int)
    t_plot = t_us_fdtd[t_idx]
    x_plot = x_mm_fdtd[x_idx]
    u_ref  = u_fdtd[np.ix_(t_idx, x_idx)]

    # PINN prediction on same grid
    tau_t = torch.tensor(phys.s_to_tau(t_plot * 1e-6), dtype=torch.float32, device=device)
    x_t   = torch.tensor(phys.mm_to_xhat(x_plot),      dtype=torch.float32, device=device)
    u_pred = model.evaluate_u_on_grid(tau_t, x_t, batch_points=40000).cpu().numpy()

    vmax = float(np.quantile(np.abs(u_ref), 0.995))
    vmax = max(vmax, 1e-20)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, sharey=True)
    kw = dict(
        extent=[t_plot.min(), t_plot.max(), x_plot.min(), x_plot.max()],
        aspect="auto", origin="lower", cmap="seismic",
    )
    for ax, data, title, scale in zip(
        axes,
        [u_ref, u_pred, u_pred - u_ref],
        ["FDTD reference", "Hybrid PINN prediction", "Prediction error"],
        [vmax, vmax, vmax],
    ):
        im = ax.imshow(data.T, vmin=-scale, vmax=scale, **kw)
        for mm in phys.x_interfaces_mm:
            ax.axhline(mm, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel("x (mm)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    axes[-1].set_xlabel("t (µs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180);  print(f"Saved: {out_path}")
    plt.close(fig)


def make_speed_of_sound_plot(
    phys: TranscranialHybridPhysics,
    nx: int = 1000,
    out_path: str = "speed_of_sound_profile.png",
) -> None:
    """Plot the piecewise speed-of-sound profile c(x) across all tissue layers."""
    xhat = np.linspace(0.0, 1.0, nx)
    x_mm = phys.xhat_to_mm(xhat)
    c    = phys.c_np(xhat)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_mm, c, color="teal", lw=2.0)
    ax.fill_between(x_mm, c, alpha=0.15, color="teal")
    for mm in phys.x_interfaces_mm:
        ax.axvline(mm, color="gray", ls="--", lw=1.0, alpha=0.7)

    # Label each layer at its midpoint
    for lyr in phys.layers:
        mid_m = 0.5 * (lyr.x_start_m + lyr.x_end_m)
        mid_mm = mid_m * 1e3
        ax.annotate(
            f"{lyr.name}\nc={lyr.c:.0f} m/s",
            xy=(mid_mm, lyr.c), xytext=(mid_mm, lyr.c + 80),
            ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Speed of sound (m/s)")
    ax.set_title("Speed of sound profile across tissue layers")
    ax.set_xlim(x_mm.min(), x_mm.max())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200);  print(f"Saved: {out_path}")
    plt.close(fig)


def make_amplitude_attenuation_plot(
    phys: TranscranialHybridPhysics,
    nx: int = 1000,
    out_path: str = "amplitude_attenuation_profile.png",
) -> None:
    """
    Plot the cumulative amplitude attenuation profile A(x) / A(0).

    The amplitude decays as A(x) = A(0) * exp(-integral_0^x alpha(x') dx')
    where alpha(x) is the frequency-dependent attenuation in Np/m.
    This shows how much signal remains at each depth.
    """
    xhat = np.linspace(0.0, 1.0, nx)
    x_mm = phys.xhat_to_mm(xhat)
    x_m  = phys.xhat_to_m(xhat)
    alpha_npm = phys.alpha_material_np(xhat)   # Np/m at the operating frequency

    # Cumulative attenuation via trapezoidal integration
    dx_m = np.diff(x_m)
    alpha_mid = 0.5 * (alpha_npm[:-1] + alpha_npm[1:])
    cum_atten = np.concatenate([[0.0], np.cumsum(alpha_mid * dx_m)])
    amplitude_ratio = np.exp(-cum_atten)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top panel: attenuation coefficient
    ax1.plot(x_mm, alpha_npm, color="crimson", lw=2.0)
    ax1.fill_between(x_mm, alpha_npm, alpha=0.15, color="crimson")
    for mm in phys.x_interfaces_mm:
        ax1.axvline(mm, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax1.set_ylabel("α (Np/m)")
    ax1.set_title(f"Attenuation coefficient at f = {phys.f0_hz/1e3:.0f} kHz")
    ax1.grid(True, alpha=0.3)

    # Label layers
    for lyr in phys.layers:
        mid_mm = 0.5 * (lyr.x_start_m + lyr.x_end_m) * 1e3
        a_val  = lyr.alpha_npm(phys.f0_hz)
        ax1.annotate(
            f"{lyr.name}\n{lyr.alpha_dBcmMHz:.1f} dB/cm/MHz",
            xy=(mid_mm, a_val), xytext=(mid_mm, a_val + 20),
            ha="center", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
        )

    # Bottom panel: cumulative amplitude ratio
    ax2.plot(x_mm, amplitude_ratio * 100, color="navy", lw=2.0)
    ax2.fill_between(x_mm, amplitude_ratio * 100, alpha=0.12, color="navy")
    for mm in phys.x_interfaces_mm:
        ax2.axvline(mm, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("Amplitude remaining (%)")
    ax2.set_title("Cumulative amplitude attenuation A(x)/A(0)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # Annotate amplitude at each interface
    for mm_val in phys.x_interfaces_mm:
        idx = int(np.argmin(np.abs(x_mm - mm_val)))
        pct = amplitude_ratio[idx] * 100
        ax2.annotate(
            f"{pct:.1f}%",
            xy=(mm_val, pct), xytext=(mm_val + 1.5, pct + 5),
            fontsize=8, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy", lw=0.7),
        )
    # Annotate final amplitude
    pct_final = amplitude_ratio[-1] * 100
    ax2.annotate(
        f"Exit: {pct_final:.1f}%",
        xy=(x_mm[-1], pct_final), xytext=(x_mm[-1] - 8, pct_final + 8),
        fontsize=9, fontweight="bold", color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200);  print(f"Saved: {out_path}")
    plt.close(fig)


def make_frequency_plot(
    phys: TranscranialHybridPhysics,
    nx: int = 1000,
    out_path: str = "frequency_response_profile.png",
) -> None:
    """
    Plot how the ultrasound frequency content changes across layers.

    Shows:
    - Top panel:  wavelength λ(x) = c(x) / f0 across layers (spatial resolution).
    - Bottom panel: frequency-dependent attenuation at several harmonics to
      illustrate low-pass filtering by tissue (skull especially attenuates
      higher harmonics much more than the fundamental).
    """
    xhat = np.linspace(0.0, 1.0, nx)
    x_mm = phys.xhat_to_mm(xhat)
    x_m  = phys.xhat_to_m(xhat)
    c    = phys.c_np(xhat)

    f0 = phys.f0_hz
    wavelength_mm = (c / f0) * 1e3

    # Frequency-dependent attenuation: compute cumulative A(x)/A(0) at f0, 2*f0, 3*f0
    harmonics = [0.5, 1.0, 2.0, 3.0]
    harmonic_labels = ["f₀/2", "f₀", "2f₀", "3f₀"]
    harmonic_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top panel: wavelength profile
    ax1.plot(x_mm, wavelength_mm, color="purple", lw=2.0)
    ax1.fill_between(x_mm, wavelength_mm, alpha=0.12, color="purple")
    for mm in phys.x_interfaces_mm:
        ax1.axvline(mm, color="gray", ls="--", lw=1.0, alpha=0.7)
    for lyr in phys.layers:
        mid_mm = 0.5 * (lyr.x_start_m + lyr.x_end_m) * 1e3
        lam = (lyr.c / f0) * 1e3
        ax1.annotate(
            f"{lyr.name}\nλ={lam:.2f} mm",
            xy=(mid_mm, lam), xytext=(mid_mm, lam + 0.5),
            ha="center", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
        )
    ax1.set_ylabel("Wavelength λ (mm)")
    ax1.set_title(f"Wavelength profile at f₀ = {f0/1e3:.0f} kHz")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: cumulative attenuation at multiple frequencies
    dx_m = np.diff(x_m)
    for mult, label, color in zip(harmonics, harmonic_labels, harmonic_colors):
        f_hz = f0 * mult
        # Recompute alpha at this frequency for each layer
        alpha_at_f = np.zeros_like(xhat)
        for lyr in phys.layers:
            if lyr == phys.layers[-1]:
                mask = (x_m >= lyr.x_start_m) & (x_m <= lyr.x_end_m)
            else:
                mask = (x_m >= lyr.x_start_m) & (x_m < lyr.x_end_m)
            alpha_at_f[mask] = lyr.alpha_npm(f_hz)

        alpha_mid = 0.5 * (alpha_at_f[:-1] + alpha_at_f[1:])
        cum = np.concatenate([[0.0], np.cumsum(alpha_mid * dx_m)])
        amp = np.exp(-cum) * 100.0
        ax2.plot(x_mm, amp, color=color, lw=2.0,
                 label=f"{label} ({f_hz/1e3:.0f} kHz)")

    for mm in phys.x_interfaces_mm:
        ax2.axvline(mm, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("Amplitude remaining (%)")
    ax2.set_title("Frequency-dependent attenuation: tissue acts as a low-pass filter")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200);  print(f"Saved: {out_path}")
    plt.close(fig)

def plot_loss(history: dict, out_path: str = "loss_transcranial_hybrid.png") -> None:
    keys = ["total", "state", "u", "q", "g", "pde", "r1", "r2", "r3", "trace", "ic"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 18), sharex=True)
    for ax, k in zip(axes, keys):
        vals = np.asarray(history.get(k, []), dtype=np.float64)
        if len(vals) == 0:
            continue
        ax.semilogy(np.arange(1, len(vals) + 1), np.maximum(vals, 1e-30), lw=1.5)
        ax.set_ylabel(k);  ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("logged step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160);  print(f"Saved: {out_path}")
    plt.close(fig)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transcranial hybrid PINN training")
    parser.add_argument("--scalp-length", type=float, default=3.2,
                        help="Entry-side scalp thickness in mm (default: 3.2)")
    parser.add_argument("--skull-length", type=float, default=10.0,
                        help="Entry-side skull thickness in mm (default: 10.0)")
    parser.add_argument("--brain-length", type=float, default=46.8,
                        help="Brain thickness in mm (default: 46.8)")
    parser.add_argument("--exit-skull-length", type=float, default=None,
                        help="Exit-side skull thickness in mm (default: mirrors entry)")
    parser.add_argument("--exit-scalp-length", type=float, default=None,
                        help="Exit-side scalp thickness in mm (default: mirrors entry)")
    parser.add_argument("--scalp-speed", type=float, default=1540.0,
                        help="Scalp wave speed in m/s (default: 1540.0)")
    parser.add_argument("--skull-speed", type=float, default=2800.0,
                        help="Skull wave speed in m/s (default: 2800.0)")
    parser.add_argument("--brain-speed", type=float, default=1560.0,
                        help="Brain wave speed in m/s (default: 1560.0)")
    parser.add_argument("--frequency", type=float, default=500.0,
                        help="Ultrasound frequency in kHz (default: 500.0)")
    parser.add_argument("--mode", choices=["normal", "early_time"],
                        default="normal",
                        help="Training mode: "
                             "'normal' = full-window FDTD supervision "
                             "(8 tiles x 200 x 128 = 204,800 supervised points/step, "
                             "production baseline with Adam + LBFGS); "
                             "'early_time' = time-marching curriculum with dense "
                             "windowed snapshot strips.")
    parser.add_argument("--snapshot-spacing-us", type=float, default=3.0,
                        help="(early_time mode only) Spacing between snapshot centres in µs.")
    parser.add_argument("--points-per-snapshot", type=int, default=80,
                        help="(early_time mode only) Random spatial points per snapshot.")
    parser.add_argument("--snapshot-window-us", type=float, default=1.0,
                        help="(early_time mode only) Half-width of snapshot window in µs.")
    return parser.parse_args()


# ------------------------------------------------------------------
#  Stage builders  (production = normal,  alternative = early_time)
# ------------------------------------------------------------------
def build_stages_normal(phys):
    """Normal training stages: PINN supervised on the full FDTD wavefield."""
    return [
        StageConfig(
            name="supervised_full",
            steps=10_000, lr=5e-4,
            n_tiles=8, tile_nt=200, tile_nx=128, n_pde=0,
            state_weight=1.0, pde_weight=0.0,
            trace_weight=0.5, ic_weight=0.2,
            tau_max=phys.tau_max,
            pde_source_enabled=True, log_every=200,
        ),
        StageConfig(
            name="pde_refinement",
            steps=5_000, lr=1e-4,
            n_tiles=8, tile_nt=200, tile_nx=128, n_pde=2_000,
            state_weight=1.0, pde_weight=0.005,
            trace_weight=0.5, ic_weight=0.1,
            tau_max=phys.tau_max,
            pde_source_enabled=True, log_every=100,
        ),
        StageConfig(
            name="cooldown",
            steps=2_000, lr=2e-5,
            n_tiles=8, tile_nt=200, tile_nx=128, n_pde=0,
            state_weight=1.0, pde_weight=0.0,
            trace_weight=0.5, ic_weight=0.05,
            tau_max=phys.tau_max,
            pde_source_enabled=True, log_every=100,
        ),
    ]


def build_stages_early_time(
    phys,
    snapshot_spacing_us: float,
    points_per_snapshot: int,
    snapshot_window_us: float,
):
    """
    EARLY-TIME PINN training
    """
    import numpy as _np

    us_to_tau = lambda us: float(phys.s_to_tau(us * 1e-6))
    snapshot_window_tau = us_to_tau(snapshot_window_us)

    # ------------------------------------------------------------------
    # Time-marching sub-horizons.
    # Hardcoded to give a ~15/30/50/full curriculum. The full-window cap
    # is phys.tau_max (= ~1.262 for the default 60 µs total).
    # ------------------------------------------------------------------
    t_total_us = float(phys.t_total_s * 1e6)
    march_us   = [
        min(15.0, t_total_us),
        min(30.0, t_total_us),
        min(50.0, t_total_us),
        t_total_us,
    ]
    march_tau  = [us_to_tau(t) for t in march_us[:-1]] + [phys.tau_max]

    def snapshots_up_to(t_hi_us: float) -> tuple[float, ...]:
        """Return snapshot CENTRES (in tau) at spacing, 2*spacing, ... ≤ t_hi_us."""
        n = int(_np.floor(t_hi_us / snapshot_spacing_us))
        ts_us = [(k + 1) * snapshot_spacing_us for k in range(n)]
        # Drop any that would put the window past the horizon
        ts_us = [t for t in ts_us if t + snapshot_window_us <= t_hi_us + 1e-9]
        return tuple(us_to_tau(t) for t in ts_us)

    stage_snapshots = [snapshots_up_to(t_hi) for t_hi in march_us]

    tau_indices_per_window_est = max(1, int(round((2.0 * snapshot_window_us) / 0.05)))

    print(f"\n[early_time mode v4] TIME-MARCHING curriculum:")
    print(f"  snapshot spacing   = {snapshot_spacing_us:.2f} µs")
    print(f"  snapshot window    = ±{snapshot_window_us:.2f} µs "
          f"(~{tau_indices_per_window_est} tau indices)")
    print(f"  points per snapshot = {points_per_snapshot}")
    print(f"  causal_eps         = 0 (no causal weighting — dense snapshots replace it)")
    for i, (t_hi_us, t_tau, snaps) in enumerate(zip(march_us, march_tau, stage_snapshots)):
        anchors = len(snaps) * tau_indices_per_window_est * points_per_snapshot
        snap_us_str = ", ".join(f"{float(phys.tau_to_s(s)) * 1e6:.1f}" for s in snaps)
        print(f"  stage {i+1}: tau_max = {t_tau:.4f}  ({t_hi_us:.1f} µs)  "
              f"{len(snaps):2d} snapshots  ≈{anchors:,} anchors/step")
        print(f"           snapshots (µs): [{snap_us_str}]")
    print()

    common = dict(
        n_tiles=0, tile_nt=0, tile_nx=0,                  # unused in snapshot mode
        n_points_per_snapshot=points_per_snapshot,
        snapshot_window_tau=snapshot_window_tau,
        pde_source_enabled=True,
    )

    return [
        # ----- Stage 1: march to 15 µs (LR 5e-4) -----
        StageConfig(
            name="march_0_15us",
            steps=6_000, lr=5e-4,
            snapshot_times_tau=stage_snapshots[0],
            tau_max=march_tau[0],
            n_pde=15_000,
            state_weight=1.0, pde_weight=1.0,
            trace_weight=1.0, ic_weight=1.0,
            causal_eps=0.0,
            log_every=100,
            **common,
        ),
        # ----- Stage 2: march to 30 µs (LR 3e-4) -----
        StageConfig(
            name="march_0_30us",
            steps=8_000, lr=3e-4,
            snapshot_times_tau=stage_snapshots[1],
            tau_max=march_tau[1],
            n_pde=20_000,
            state_weight=1.0, pde_weight=1.0,
            trace_weight=1.0, ic_weight=0.8,
            causal_eps=0.0,
            log_every=100,
            **common,
        ),
        # ----- Stage 3: march to 50 µs (LR 2e-4) -----
        StageConfig(
            name="march_0_50us",
            steps=10_000, lr=2e-4,
            snapshot_times_tau=stage_snapshots[2],
            tau_max=march_tau[2],
            n_pde=25_000,
            state_weight=1.0, pde_weight=1.0,
            trace_weight=1.0, ic_weight=0.5,
            causal_eps=0.0,
            log_every=100,
            **common,
        ),
        # ----- Stage 4: polish full horizon (LR 1e-4) -----
        StageConfig(
            name="polish_full",
            steps=10_000, lr=1e-4,
            snapshot_times_tau=stage_snapshots[3],
            tau_max=march_tau[3],
            n_pde=30_000,
            state_weight=1.0, pde_weight=1.0,
            trace_weight=1.0, ic_weight=0.3,
            causal_eps=0.0,
            log_every=100,
            **common,
        ),
    ]


def main():
    args = parse_args()
    set_seed(42)
    workdir = Path(".")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phys = TranscranialHybridPhysics(
        f0_hz=args.frequency * 1e3,
        scalp_thickness_m=args.scalp_length * 1e-3,
        skull_thickness_m=args.skull_length * 1e-3,
        brain_thickness_m=args.brain_length * 1e-3,
        exit_skull_thickness_m=(args.exit_skull_length * 1e-3
                                if args.exit_skull_length is not None else None),
        exit_scalp_thickness_m=(args.exit_scalp_length * 1e-3
                                if args.exit_scalp_length is not None else None),
        scalp_c=args.scalp_speed,
        skull_c=args.skull_speed,
        brain_c=args.brain_speed,
    )
    phys.print_summary()

    # ---- 1. Generate FDTD reference if needed ----
    phys.maybe_generate_reference(workdir)

    # ---- 2. Load preprocessed training data ----
    norm_npz = workdir / "pinn_training_full_normalized.npz"
    if not norm_npz.exists():
        from preprocess_fdtd_data import preprocess
        print("Preprocessed data not found, running preprocessing...")
        preprocess(phys=phys)

    data    = np.load(norm_npz)
    u_scale = float(data["u_scale"])
    q_scale = float(data["q_scale"])
    g_scale = float(data["g_scale"])
    print(f"\nScale factors loaded:")
    print(f"  u_scale = {u_scale:.4e} m")
    print(f"  q_scale = {q_scale:.4e} m  (q/u = {q_scale/u_scale:.1f}x)")
    print(f"  g_scale = {g_scale:.4e} m  (g/u = {g_scale/u_scale:.1f}x)")

    dataset = {
        "tau" : data["tau"].astype(np.float32),
        "xhat": data["xhat"].astype(np.float32),
        "u"   : data["u"].astype(np.float32),    # already normalized, O(1)
        "q"   : data["q"].astype(np.float32),    # already normalized, O(1)
        "g"   : data["g"].astype(np.float32),    # already normalized, O(1)
    }
    print(f"\nDataset: Nt={dataset['tau'].size}, Nx={dataset['xhat'].size}, "
          f"total={dataset['tau'].size * dataset['xhat'].size:,} points")

    # ---- 3. Build normalized probe traces ----
    reference_npz = workdir / "transcranial_fdtd1d_reference_full.npz"
    traces = build_traces(phys, reference_npz, u_scale)

    # ---- 4. Build model ----
    net = FourierMLP(
        n_freqs=64,
        hidden_features=256,
        hidden_layers=5,
        out_features=3,
        omega_low=1.0,  # covers [1, 300] rad/tau, omega_dim = 149
        omega_high=300.0,
    )

    model = FirstOrderHybridPINN(
        network=net,
        phys=phys,
        u_scale=u_scale,
        q_scale=q_scale,
        g_scale=g_scale,
        pde_source_enabled=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNetwork: {type(net).__name__}, {n_params:,} trainable parameters")

    # ---- 5. Train ----
    trainer = HybridTrainer(
        model=model,
        phys=phys,
        device=device,
        seed=42,
        max_batch_points=32000,
    )
    # ------------------------------------------------------------------
    # Tile sizing (5-layer, L = 73.2 mm):
    # ------------------------------------------------------------------
    if args.mode == "early_time":
        stages = build_stages_early_time(
            phys,
            snapshot_spacing_us=args.snapshot_spacing_us,
            points_per_snapshot=args.points_per_snapshot,
            snapshot_window_us=args.snapshot_window_us,
        )
    else:  # normal
        stages = build_stages_normal(phys)

    trainer.train(stages=stages, dataset=dataset, traces=traces)

    # LBFGS refinement: only in normal mode (full FDTD supervision).
    # early_time: LBFGS would re-anchor against the full FDTD wavefield and
    # undo the early-time curriculum.
    if args.mode == "normal":
        trainer.lbfgs_refine(dataset, max_iter=300, tau_max=phys.tau_max)

    # ---- 6. Save ----
    # Build physics config from all layers
    physics_config = {
        "f0_hz": phys.f0_hz,
        "t_total_us": phys.t_total_s * 1e6,
    }
    for lyr in phys.layers:
        prefix = lyr.name
        thickness_m = lyr.x_end_m - lyr.x_start_m
        physics_config[f"{prefix}_thickness_m"] = thickness_m
        physics_config[f"{prefix}_c"] = lyr.c
        physics_config[f"{prefix}_rho"] = lyr.rho
        physics_config[f"{prefix}_alpha"] = lyr.alpha_dBcmMHz

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": trainer.history,
            "scales": {"u_scale": u_scale, "q_scale": q_scale, "g_scale": g_scale},
            "network_type": type(net).__name__,
            "physics_config": physics_config,
            "network_config": {
                "n_freqs": 64,
                "hidden_features": 256,
                "hidden_layers": 5,
                "out_features": 3,
                "omega_low": 1.0,
                "omega_high": 300.0,
            },
            "training_mode": args.mode,
            "early_time_config": (
                {
                    "snapshot_spacing_us": float(args.snapshot_spacing_us),
                    "snapshot_window_us":  float(args.snapshot_window_us),
                    "points_per_snapshot": int(args.points_per_snapshot),
                    "schedule":            "v4 time-marching curriculum (15/30/50/full µs)",
                }
                if args.mode == "early_time" else None
            ),
        },
        f"pinn_transcranial_hybrid_{args.mode}.pt",
    )
    print(f"Saved: pinn_transcranial_hybrid_{args.mode}.pt")

    # ---- 7. Plots ----
    plot_loss(trainer.history)
    make_wavefield_plot(model, phys, device)
    make_comparison_plot(model, phys, reference_npz, device)
    make_speed_of_sound_plot(phys)
    make_amplitude_attenuation_plot(phys)
    make_frequency_plot(phys)


if __name__ == "__main__":
    main()
