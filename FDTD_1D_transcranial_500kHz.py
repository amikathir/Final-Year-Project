"""
1D transcranial ultrasound FDTD reference solver.

Discretises the heterogeneous damped acoustic equation
    u_tt + 2*alpha(x)*u_t = (1/rho(x)) * d/dx[ kappa(x) * du/dx ] + s(x,t)/rho(x)
on a 5-layer head model (scalp - skull - brain - skull - scalp) with a
Gaussian-modulated 500 kHz tone-burst source and PML-style sponge boundaries.
The conservative leapfrog uses face-averaged stiffness flux (k_plus, k_minus)
so interfaces are stable, and homogeneous Neumann boundaries behind the sponge.

Run directly to generate two NPZ files:
  * transcranial_fdtd1d_reference_full.npz   full-resolution wavefield
  * transcranial_fdtd1d_reference.npz        coarsely subsampled for plots

KNOWN ISSUE: the damping coefficient on line ~376 currently reads
    damp = 2.0 * alpha_total * _dt
which is missing a factor of c (the local sound speed). The physically
complete form is
    damp = 2.0 * alpha_total * c_local * _dt
The current implementation under-attenuates by ~c (1500-2800x weaker).
See README "Known issues" for the fix.
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

SCALP_THICKNESS_M = 3.2e-3
SKULL_THICKNESS_M = 10.0e-3
BRAIN_THICKNESS_M = 66.8e-3
L = SCALP_THICKNESS_M + SKULL_THICKNESS_M + BRAIN_THICKNESS_M

F0_HZ = 500.0e3
T_TOTAL_S = 60.0e-6

# Spatial layout of interfaces
X_SCALP_SKULL = SCALP_THICKNESS_M
X_SKULL_BRAIN = SCALP_THICKNESS_M + SKULL_THICKNESS_M

# Snapshot times for profile exports / diagnostic plots
PROFILE_TIMES_US = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
EARLY_ANCHOR_MAX_US = 10.0

# ----------------------------------------------------------------------
# Tissue properties
# ----------------------------------------------------------------------
TISSUES = {
    "scalp": {
        "rho": 1050.0,              # kg/m^3
        "c": 1540.0,                # m/s
        "alpha_dBcmMHz": 0.60,      # dB / cm / MHz
        "x_start": 0.0,
        "x_end": X_SCALP_SKULL,
    },
    "skull": {
        "rho": 1900.0,
        "c": 2800.0,
        "alpha_dBcmMHz": 18.0,
        "x_start": X_SCALP_SKULL,
        "x_end": X_SKULL_BRAIN,
    },
    "brain": {
        "rho": 1040.0,
        "c": 1560.0,
        "alpha_dBcmMHz": 0.60,
        "x_start": X_SKULL_BRAIN,
        "x_end": L,
    },
}

# ----------------------------------------------------------------------
# Numerical configuration
# ----------------------------------------------------------------------
NX = 2400                        # 80 mm / 2400 = 33.3 um spacing
DX = L / NX

# CFL-limited time step for explicit scheme
C_MAX = max(t["c"] for t in TISSUES.values())
DT_CFL = 0.45 * DX / C_MAX
DT_TARGET = 5.0e-9               # 5 ns target for a stable, well-resolved simulation
DT = min(DT_TARGET, DT_CFL)
N_STEPS = int(np.ceil(T_TOTAL_S / DT))

# Source configuration: Gaussian-windowed tone burst injected in scalp
SOURCE_X_M = 1.0e-3              # source centre 1 mm from left boundary, inside scalp
SOURCE_FWHM_M = 0.6e-3           # spatial compactness
SOURCE_SIGMA_M = SOURCE_FWHM_M / 2.355
SOURCE_CYCLES = 3.0
SOURCE_T0_S = 2.5 / F0_HZ        # start delay so the pulse is cleanly launched
SOURCE_SIGMA_T_S = SOURCE_CYCLES / (2.355 * F0_HZ)
SOURCE_AMPLITUDE = 5.0e8         # tuned so the wavefield is visually clear without instability

# Boundary sponge to suppress reflections from the outer numerical edges
SPONGE_WIDTH_M = 2.0e-3
SPONGE_STRENGTH = 3.0e6          # 1/s


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def alpha_dbcmmhz_to_npm(alpha_dbcmmhz: float, freq_hz: float) -> float:
    """Convert attenuation from dB/cm/MHz to Nepers/m at frequency freq_hz."""
    alpha_dBcm = alpha_dbcmmhz * (freq_hz / 1e6)
    alpha_dBm = alpha_dBcm * 100.0
    return alpha_dBm / 8.685889638

def tissue_property(x_phys: np.ndarray, prop: str) -> np.ndarray:
    x_phys = np.asarray(x_phys, dtype=np.float64)
    out = np.zeros_like(x_phys, dtype=np.float64)

    for t in TISSUES.values():
        x0 = t["x_start"]
        x1 = t["x_end"]
        if math.isclose(x1, L):
            mask = (x_phys >= x0) & (x_phys <= x1)
        else:
            mask = (x_phys >= x0) & (x_phys < x1)

        if prop == "rho":
            out[mask] = t["rho"]
        elif prop == "c":
            out[mask] = t["c"]
        elif prop == "alpha_Npm":
            out[mask] = alpha_dbcmmhz_to_npm(t["alpha_dBcmMHz"], F0_HZ)
        elif prop == "kappa":
            out[mask] = t["rho"] * t["c"] ** 2
        else:
            raise ValueError(f"Unsupported property: {prop}")

    return out

def build_sponge_alpha(x_phys: np.ndarray) -> np.ndarray:
    x_phys = np.asarray(x_phys, dtype=np.float64)
    sponge = np.zeros_like(x_phys)

    left_mask = x_phys < SPONGE_WIDTH_M
    if np.any(left_mask):
        xi = (SPONGE_WIDTH_M - x_phys[left_mask]) / SPONGE_WIDTH_M
        sponge[left_mask] = SPONGE_STRENGTH * xi**2

    right_mask = x_phys > (L - SPONGE_WIDTH_M)
    if np.any(right_mask):
        xi = (x_phys[right_mask] - (L - SPONGE_WIDTH_M)) / SPONGE_WIDTH_M
        sponge[right_mask] = np.maximum(sponge[right_mask], SPONGE_STRENGTH * xi**2)

    return sponge

def source_space(x_phys: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((x_phys - SOURCE_X_M) / SOURCE_SIGMA_M) ** 2)

def source_time(t_s: float) -> float:
    tau = t_s - SOURCE_T0_S
    envelope = np.exp(-0.5 * (tau / SOURCE_SIGMA_T_S) ** 2)
    carrier = np.sin(2.0 * np.pi * F0_HZ * tau)
    return SOURCE_AMPLITUDE * envelope * carrier

def save_reference_npzs(x_phys: np.ndarray, t_s: np.ndarray, u_hist: np.ndarray) -> None:
    x_mm = 1e3 * np.asarray(x_phys, dtype=np.float64)
    t_us = 1e6 * np.asarray(t_s, dtype=np.float64)
    u_hist = np.asarray(u_hist, dtype=np.float64)

    np.savez(
        "transcranial_fdtd1d_reference_full.npz",
        x_mm=x_mm,
        t_us=t_us,
        u=u_hist,
    )
    print("Saved: transcranial_fdtd1d_reference_full.npz")

    early_mask = t_us <= EARLY_ANCHOR_MAX_US
    np.savez(
        "transcranial_fdtd1d_early_anchors.npz",
        x_mm=x_mm,
        t_us=t_us[early_mask],
        u=u_hist[early_mask, :],
    )
    print("Saved: transcranial_fdtd1d_early_anchors.npz")

def plot_reference_wavefield_xt(x_phys: np.ndarray, t_s: np.ndarray, u_hist: np.ndarray) -> None:
    x_mm = 1e3 * np.asarray(x_phys)
    t_us = 1e6 * np.asarray(t_s)
    u = np.asarray(u_hist).T

    vmax = np.percentile(np.abs(u), 99.5)
    vmax = max(vmax, 1e-16)

    plt.figure(figsize=(13, 4.8))
    im = plt.imshow(
        u,
        extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
        origin="lower",
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.axhline(X_SCALP_SKULL * 1e3, ls="--", c="k", lw=1.5)
    plt.axhline(X_SKULL_BRAIN * 1e3, ls="--", c="k", lw=1.5)
    plt.text(1.2, X_SCALP_SKULL * 1e3 + 0.35, "Scalp|Skull", color="k", fontsize=11)
    plt.text(1.2, X_SKULL_BRAIN * 1e3 + 0.35, "Skull|Brain", color="k", fontsize=11)
    plt.xlabel("t (us)")
    plt.ylabel("x (mm)")
    plt.title("1D transcranial ultrasound reference wavefield (500 kHz, 60 us)")
    cbar = plt.colorbar(im)
    cbar.set_label("u (arb. units)")
    plt.tight_layout()
    plt.savefig("transcranial_fdtd1d_reference_wavefield_xt.png", dpi=220)
    plt.close()
    print("Saved: transcranial_fdtd1d_reference_wavefield_xt.png")

def plot_profiles(x_phys: np.ndarray, t_s: np.ndarray, u_hist: np.ndarray) -> None:
    x_mm = 1e3 * np.asarray(x_phys)
    t_us = 1e6 * np.asarray(t_s)

    plt.figure(figsize=(12, 7))
    for target_us in PROFILE_TIMES_US:
        idx = int(np.argmin(np.abs(t_us - target_us)))
        plt.plot(x_mm, u_hist[idx], lw=1.1, label=f"{t_us[idx]:.1f} us")

    plt.axvline(X_SCALP_SKULL * 1e3, ls="--", c="k", lw=1.3)
    plt.axvline(X_SKULL_BRAIN * 1e3, ls="--", c="k", lw=1.3)
    plt.xlabel("x (mm)")
    plt.ylabel("u (arb. units)")
    plt.title("Spatial wavefield profiles through scalp / skull / brain")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig("transcranial_fdtd1d_profiles.png", dpi=220)
    plt.close()
    print("Saved: transcranial_fdtd1d_profiles.png")
# ----------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------
def run_fdtd_1d(phys=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run 1D FDTD solver. If phys is provided, uses its tissue/source config."""
    # Configuration from physics object or module-level defaults
    if phys is not None:
        _L = phys.L_m
        _f0 = phys.f0_hz
        _t_total = phys.t_total_s
        _layers = phys.layers
        _src_x_m = phys.source_x_m
        _src_sigma_m = phys.source_sigma_m
        _src_t0 = phys.source_t0_s
        _src_sigma_t = phys.source_sigma_t_s
        _src_amp = phys.source_amplitude
        _sponge_w = phys.sponge_width_m
        _sponge_s = phys.sponge_strength
    else:
        _L = L
        _f0 = F0_HZ
        _t_total = T_TOTAL_S
        _layers = None
        _src_x_m = SOURCE_X_M
        _src_sigma_m = SOURCE_SIGMA_M
        _src_t0 = SOURCE_T0_S
        _src_sigma_t = SOURCE_SIGMA_T_S
        _src_amp = SOURCE_AMPLITUDE
        _sponge_w = SPONGE_WIDTH_M
        _sponge_s = SPONGE_STRENGTH

    # Grid resolution: maintain ~33 µm spacing
    _dx_target = L / NX  # default spacing from module constants
    _nx = max(NX, int(round(_L / _dx_target)))
    _dx = _L / _nx

    _c_max = max(lyr.c for lyr in _layers) if _layers else C_MAX
    _dt_cfl = 0.45 * _dx / _c_max
    _dt = min(DT_TARGET, _dt_cfl)
    _n_steps = int(np.ceil(_t_total / _dt))

    print("Running 1D transcranial ultrasound reference solver...")
    print(f"Total thickness: {_L*1e3:.1f} mm")
    if _layers:
        for lyr in _layers:
            thick = (lyr.x_end_m - lyr.x_start_m) * 1e3
            print(f"  {lyr.name}: {thick:.1f} mm, c={lyr.c:.1f} m/s, rho={lyr.rho:.1f} kg/m^3")
    else:
        print(f"Layers: scalp={SCALP_THICKNESS_M*1e3:.1f} mm, skull={SKULL_THICKNESS_M*1e3:.1f} mm, brain={BRAIN_THICKNESS_M*1e3:.1f} mm")
    print(f"Frequency: {_f0/1e3:.1f} kHz")
    print(f"Time window: {_t_total*1e6:.1f} us")
    print(f"dx = {_dx*1e6:.2f} um, dt = {_dt*1e9:.2f} ns, steps = {_n_steps}")

    x = np.linspace(0.0, _L, _nx + 1)

    # Tissue properties
    if _layers:
        def _get_prop(x_phys, prop):
            out = np.zeros_like(x_phys, dtype=np.float64)
            for j, lyr in enumerate(_layers):
                if j == len(_layers) - 1:
                    mask = (x_phys >= lyr.x_start_m) & (x_phys <= lyr.x_end_m)
                else:
                    mask = (x_phys >= lyr.x_start_m) & (x_phys < lyr.x_end_m)
                if prop == "rho":
                    out[mask] = lyr.rho
                elif prop == "alpha_Npm":
                    out[mask] = alpha_dbcmmhz_to_npm(lyr.alpha_dBcmMHz, _f0)
                elif prop == "kappa":
                    out[mask] = lyr.rho * lyr.c ** 2
            return out
        rho = _get_prop(x, "rho")
        alpha_material = _get_prop(x, "alpha_Npm")
        kappa = _get_prop(x, "kappa")
    else:
        rho = tissue_property(x, "rho")
        alpha_material = tissue_property(x, "alpha_Npm")
        kappa = tissue_property(x, "kappa")

    # Sponge layer
    sponge = np.zeros_like(x)
    left_mask = x < _sponge_w
    if np.any(left_mask):
        xi = (_sponge_w - x[left_mask]) / _sponge_w
        sponge[left_mask] = _sponge_s * xi**2
    right_mask = x > (_L - _sponge_w)
    if np.any(right_mask):
        xi = (x[right_mask] - (_L - _sponge_w)) / _sponge_w
        sponge[right_mask] = np.maximum(sponge[right_mask], _sponge_s * xi**2)
    alpha_total = alpha_material + sponge

    # Source
    src_x = np.exp(-0.5 * ((x - _src_x_m) / _src_sigma_m) ** 2)

    def _source_time_fn(t_s):
        dt = t_s - _src_t0
        env = np.exp(-0.5 * (dt / _src_sigma_t) ** 2)
        carrier = np.sin(2.0 * np.pi * _f0 * dt)
        return _src_amp * env * carrier

    # Initial rest state
    u_prev = np.zeros_like(x)
    u_cur = np.zeros_like(x)

    t_hist = []
    u_hist = []

    report_every = max(1, _n_steps // 10)

    for step in range(_n_steps + 1):
        t_now = step * _dt
        t_hist.append(t_now)
        u_hist.append(u_cur.copy())

        if step == _n_steps:
            break

        # Conservative heterogeneous stiffness flux
        flux = np.zeros_like(u_cur)
        k_plus = 0.5 * (kappa[2:] + kappa[1:-1])
        k_minus = 0.5 * (kappa[1:-1] + kappa[:-2])
        flux[1:-1] = (
            k_plus * (u_cur[2:] - u_cur[1:-1])
            - k_minus * (u_cur[1:-1] - u_cur[:-2])
        ) / _dx**2

        s_now = _source_time_fn(t_now) * src_x

        # c_local = np.sqrt(kappa / rho)
        # damp = 2.0 * alpha_total * c_local * _dt
        damp = 2.0 * alpha_total * _dt
        u_next = (
            (2.0 - damp) * u_cur
            - (1.0 - damp) * u_prev
            + (_dt**2 / rho) * (flux + s_now)
        )

        # Homogeneous Neumann boundary condition
        u_next[0] = u_next[1]
        u_next[-1] = u_next[-2]

        u_prev = u_cur
        u_cur = u_next

        if step % report_every == 0:
            print(f"  progress: {100.0 * step / _n_steps:5.1f}%")

    u_hist = np.asarray(u_hist, dtype=np.float64)
    t_hist = np.asarray(t_hist, dtype=np.float64)
    return x, t_hist, u_hist


if __name__ == "__main__":
    x, t_s, u_hist = run_fdtd_1d()
    save_reference_npzs(x, t_s, u_hist)
    plot_reference_wavefield_xt(x, t_s, u_hist)
    plot_profiles(x, t_s, u_hist)
    print("Done.")
