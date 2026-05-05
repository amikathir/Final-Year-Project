"""
Transcranial physics: geometry, tissue properties, dimensionless coordinates.

`TranscranialHybridPhysics` is the single source of truth for the 5-layer head
model (scalp - skull - brain - skull - scalp). It exposes:

  * Layer geometry and per-layer (rho, c, alpha) tables (TissueLayer dataclass).
  * Coordinate conversions:  tau <-> seconds,  x_hat <-> metres / mm.
  * Piecewise material fields rho_np / kappa_np / c_np / alpha_material_np.
  * Dimensionless versions rho_hat_np / kappa_hat_np / alpha_hat_np that the
    PINN buffers internally (multiply / divide by t_ref, c_ref, rho_ref to
    transform between physical and dimensionless units).
  * Sponge profile (boundary-absorbing damping), in physical and hat form.
  * Source spatial / temporal profiles and the dimensionless source_hat_np.

This module is pure NumPy; no PyTorch operations live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import numpy as np


_DB_PER_NP = 8.685889638


# ─────────────────────────────────────────────────────────────────────
#  TissueLayer  -  one row of the tissue table
# ─────────────────────────────────────────────────────────────────────
@dataclass
class TissueLayer:
    name: str
    rho: float
    c: float
    alpha_dBcmMHz: float
    x_start_m: float
    x_end_m: float

    @property
    def kappa(self) -> float:
        return self.rho * self.c * self.c

    def alpha_npm(self, f_hz: float) -> float:
        alpha_db_cm = self.alpha_dBcmMHz * (f_hz / 1.0e6)
        alpha_db_m = alpha_db_cm * 100.0
        return alpha_db_m / _DB_PER_NP


# ─────────────────────────────────────────────────────────────────────
#  TranscranialHybridPhysics  -  5-layer geometry + dimensionless helpers
# ─────────────────────────────────────────────────────────────────────
class TranscranialHybridPhysics:
    """
    Physics helper aligned with the 1D 500 kHz transcranial benchmark.
    This class remains useful even for the simplified u-only surrogate because
    it provides coordinate conversions, layer interfaces, and reference helpers.
    """

    def __init__(
        self,
        f0_hz: float = 500.0e3,
        t_total_us: float = 60.0,
        scalp_thickness_m: float = 3.2e-3,
        skull_thickness_m: float = 10.0e-3,
        brain_thickness_m: float = 66.8e-3,
        scalp_c: float = 1540.0,
        skull_c: float = 2800.0,
        brain_c: float = 1560.0,
        scalp_rho: float = 1050.0,
        skull_rho: float = 1900.0,
        brain_rho: float = 1040.0,
        scalp_alpha: float = 0.60,
        skull_alpha: float = 18.0,
        brain_alpha: float = 0.60,
        # Exit-side layers (far side of head) — mirror entry by default
        exit_skull_thickness_m: float | None = None,
        exit_scalp_thickness_m: float | None = None,
        exit_skull_c: float | None = None,
        exit_scalp_c: float | None = None,
        exit_skull_rho: float | None = None,
        exit_scalp_rho: float | None = None,
        exit_skull_alpha: float | None = None,
        exit_scalp_alpha: float | None = None,
        source_x_m: float = 1.0e-3,
        source_fwhm_m: float = 0.6e-3,
        source_cycles: float = 3.0,
        source_delay_cycles: float = 2.5,
        source_amplitude: float = 5.0e8,
        sponge_width_m: float = 2.0e-3,
        sponge_strength: float = 3.0e6,
    ):
        # Exit-side defaults: mirror the entry-side parameters
        _exit_skull_thick = exit_skull_thickness_m if exit_skull_thickness_m is not None else skull_thickness_m
        _exit_scalp_thick = exit_scalp_thickness_m if exit_scalp_thickness_m is not None else scalp_thickness_m
        _exit_skull_c     = exit_skull_c     if exit_skull_c     is not None else skull_c
        _exit_scalp_c     = exit_scalp_c     if exit_scalp_c     is not None else scalp_c
        _exit_skull_rho   = exit_skull_rho   if exit_skull_rho   is not None else skull_rho
        _exit_scalp_rho   = exit_scalp_rho   if exit_scalp_rho   is not None else scalp_rho
        _exit_skull_alpha = exit_skull_alpha  if exit_skull_alpha  is not None else skull_alpha
        _exit_scalp_alpha = exit_scalp_alpha  if exit_scalp_alpha  is not None else scalp_alpha

        L_m = (scalp_thickness_m + skull_thickness_m + brain_thickness_m
               + _exit_skull_thick + _exit_scalp_thick)
        self.L_m = float(L_m)
        self.f0_hz = float(f0_hz)
        self.t_total_s = float(t_total_us) * 1.0e-6

        # 5-layer interfaces: scalp | skull | brain | skull_exit | scalp_exit
        x0 = 0.0
        x1 = scalp_thickness_m
        x2 = x1 + skull_thickness_m
        x3 = x2 + brain_thickness_m
        x4 = x3 + _exit_skull_thick
        x5 = x4 + _exit_scalp_thick   # == L_m

        self.layers = [
            TissueLayer("scalp",      scalp_rho,     scalp_c,     scalp_alpha,     x0, x1),
            TissueLayer("skull",      skull_rho,      skull_c,      skull_alpha,     x1, x2),
            TissueLayer("brain",      brain_rho,      brain_c,      brain_alpha,     x2, x3),
            TissueLayer("skull_exit", _exit_skull_rho, _exit_skull_c, _exit_skull_alpha, x3, x4),
            TissueLayer("scalp_exit", _exit_scalp_rho, _exit_scalp_c, _exit_scalp_alpha, x4, x5),
        ]

        self.rho_ref = self.layers[-1].rho
        self.kappa_ref = self.layers[-1].kappa
        self.c_ref = np.sqrt(self.kappa_ref / self.rho_ref)
        self.t_ref = self.L_m / self.c_ref
        self.tau_max = self.t_total_s / self.t_ref

        self.source_x_m = float(source_x_m)
        self.source_xhat = self.source_x_m / self.L_m
        self.source_fwhm_m = float(source_fwhm_m)
        self.source_sigma_m = self.source_fwhm_m / 2.355
        self.source_sigma_xhat = self.source_sigma_m / self.L_m
        self.source_cycles = float(source_cycles)
        self.source_t0_s = float(source_delay_cycles) / self.f0_hz
        self.source_sigma_t_s = self.source_cycles / (2.355 * self.f0_hz)
        self.source_amplitude = float(source_amplitude)

        self.sponge_width_m = float(sponge_width_m)
        self.sponge_width_xhat = self.sponge_width_m / self.L_m
        self.sponge_strength = float(sponge_strength)

        self.x_interfaces_xhat = np.array(
            [lyr.x_end_m / self.L_m for lyr in self.layers[:-1]], dtype=np.float32
        )
        self.x_interfaces_mm = self.xhat_to_mm(self.x_interfaces_xhat)

        self._rho_ref_safe = max(self.rho_ref, 1e-12)
        self._kappa_ref_safe = max(self.kappa_ref, 1e-12)

    # ─── Coordinate conversions  (tau <-> s,   x_hat <-> m / mm) ─────
    def xhat_to_m(self, xhat):
        return np.asarray(xhat) * self.L_m

    def xhat_to_mm(self, xhat):
        return 1.0e3 * self.xhat_to_m(xhat)

    def tau_to_s(self, tau):
        return np.asarray(tau) * self.t_ref

    def tau_to_us(self, tau):
        return 1.0e6 * self.tau_to_s(tau)

    def s_to_tau(self, t_s):
        return np.asarray(t_s) / self.t_ref

    def mm_to_xhat(self, x_mm):
        return (1.0e-3 * np.asarray(x_mm)) / self.L_m

    # ─── Piecewise tissue fields  (physical units) ───────────────────
    def _piecewise_from_layers(self, xhat: np.ndarray, attr: str) -> np.ndarray:
        x_m = self.xhat_to_m(xhat)
        out = np.zeros_like(x_m, dtype=np.float64)
        for i, lyr in enumerate(self.layers):
            if i == len(self.layers) - 1:
                mask = (x_m >= lyr.x_start_m) & (x_m <= lyr.x_end_m)
            else:
                mask = (x_m >= lyr.x_start_m) & (x_m < lyr.x_end_m)
            out[mask] = getattr(lyr, attr)
        return out

    def rho_np(self, xhat: np.ndarray) -> np.ndarray:
        return self._piecewise_from_layers(xhat, "rho")

    def kappa_np(self, xhat: np.ndarray) -> np.ndarray:
        return self._piecewise_from_layers(xhat, "kappa")

    def c_np(self, xhat: np.ndarray) -> np.ndarray:
        rho = self.rho_np(xhat)
        kappa = self.kappa_np(xhat)
        return np.sqrt(np.maximum(kappa / np.maximum(rho, 1e-12), 0.0))

    # ─── Dimensionless versions  (consumed by the PINN buffers) ──────
    def rho_hat_np(self, xhat: np.ndarray) -> np.ndarray:
        return self.rho_np(xhat) / self._rho_ref_safe

    def kappa_hat_np(self, xhat: np.ndarray) -> np.ndarray:
        return self.kappa_np(xhat) / self._kappa_ref_safe

    def alpha_material_np(self, xhat: np.ndarray) -> np.ndarray:
        x_m = self.xhat_to_m(xhat)
        out = np.zeros_like(x_m, dtype=np.float64)
        for i, lyr in enumerate(self.layers):
            if i == len(self.layers) - 1:
                mask = (x_m >= lyr.x_start_m) & (x_m <= lyr.x_end_m)
            else:
                mask = (x_m >= lyr.x_start_m) & (x_m < lyr.x_end_m)
            out[mask] = lyr.alpha_npm(self.f0_hz)
        return out

    # def alpha_hat_np(self, xhat):
    #     return self.alpha_material_np(xhat) * self.c_np(xhat) * self.t_ref
    def alpha_hat_np(self, xhat: np.ndarray) -> np.ndarray:
        return self.alpha_material_np(xhat) * self.t_ref

    # ─── Sponge boundary absorber  (numerical, not physical) ─────────
    def sponge_np(self, xhat: np.ndarray) -> np.ndarray:
        x_m = self.xhat_to_m(xhat)
        out = np.zeros_like(x_m, dtype=np.float64)
        left = x_m < self.sponge_width_m
        if np.any(left):
            xi = (self.sponge_width_m - x_m[left]) / self.sponge_width_m
            out[left] = self.sponge_strength * xi * xi
        right = x_m > (self.L_m - self.sponge_width_m)
        if np.any(right):
            xi = (x_m[right] - (self.L_m - self.sponge_width_m)) / self.sponge_width_m
            out[right] = np.maximum(out[right], self.sponge_strength * xi * xi)
        return out

    def sponge_hat_np(self, xhat: np.ndarray) -> np.ndarray:
        return self.sponge_np(xhat) * self.t_ref

    # ─── Source term  (Gaussian-modulated 500 kHz tone burst) ────────
    def source_space_np(self, xhat: np.ndarray) -> np.ndarray:
        x_m = self.xhat_to_m(xhat)
        return np.exp(-0.5 * ((x_m - self.source_x_m) / self.source_sigma_m) ** 2)

    def source_time_np(self, tau: np.ndarray) -> np.ndarray:
        t_s = self.tau_to_s(tau)
        dt = t_s - self.source_t0_s
        env = np.exp(-0.5 * (dt / self.source_sigma_t_s) ** 2)
        carrier = np.sin(2.0 * np.pi * self.f0_hz * dt)
        return self.source_amplitude * env * carrier

    def source_np(self, tau: np.ndarray, xhat: np.ndarray) -> np.ndarray:
        return self.source_time_np(tau) * self.source_space_np(xhat)

    def source_hat_np(self, tau: np.ndarray, xhat: np.ndarray) -> np.ndarray:
        source_phys = self.source_np(tau, xhat)
        return (self.t_ref**2 / self._rho_ref_safe) * source_phys

    def _geometry_fingerprint(self) -> dict:
        """Return a compact dict that uniquely identifies the current geometry."""
        return {
            "L_m": self.L_m,
            "f0_hz": self.f0_hz,
            "t_total_s": self.t_total_s,
            "n_layers": len(self.layers),
            "layers": [
                {
                    "name": lyr.name,
                    "x_start_m": lyr.x_start_m,
                    "x_end_m": lyr.x_end_m,
                    "rho": lyr.rho,
                    "c": lyr.c,
                    "alpha_dBcmMHz": lyr.alpha_dBcmMHz,
                }
                for lyr in self.layers
            ],
        }

    def _reference_is_stale(self, output_dir: Path) -> bool:
        """Check whether existing NPZ matches current geometry."""
        full = output_dir / "transcranial_fdtd1d_reference_full.npz"
        if not full.exists():
            return True
        try:
            d = np.load(full)
            x_mm = d["x_mm"]
            L_npz_mm = float(x_mm[-1])
            L_cur_mm = self.L_m * 1e3
            if abs(L_npz_mm - L_cur_mm) > 0.1:   # > 0.1 mm mismatch
                print(f"[geometry mismatch] NPZ L = {L_npz_mm:.1f} mm, "
                      f"current L = {L_cur_mm:.1f} mm → regenerating.")
                return True
        except Exception:
            return True
        return False

    def maybe_generate_reference(self, output_dir: str | Path = ".") -> None:
        output_dir = Path(output_dir)
        full = output_dir / "transcranial_fdtd1d_reference_full.npz"
        early = output_dir / "transcranial_fdtd1d_early_anchors.npz"
        norm = output_dir / "pinn_training_full_normalized.npz"

        stale = self._reference_is_stale(output_dir)
        if full.exists() and early.exists() and not stale:
            print("Reference NPZ files already exist and match current geometry.")
            return

        # Delete stale files so everything regenerates consistently
        if stale:
            for f in [full, early, norm]:
                if f.exists():
                    f.unlink()
                    print(f"  Deleted stale: {f.name}")

        solver_path = output_dir / "FDTD_1D_transcranial_500kHz.py"
        if not solver_path.exists():
            raise FileNotFoundError(
                "Reference npz files were not found and FDTD_1D_transcranial_500kHz.py is missing."
            )

        spec = importlib.util.spec_from_file_location("fdtd_ref", solver_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load reference solver module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cwd = Path.cwd()
        try:
            if cwd != output_dir:
                import os
                os.chdir(output_dir)

            x, t_s, u_hist = module.run_fdtd_1d(phys=self)
            module.save_reference_npzs(x, t_s, u_hist)

            if hasattr(module, "plot_reference_wavefield_xt"):
                module.plot_reference_wavefield_xt(x, t_s, u_hist)
            if hasattr(module, "plot_profiles"):
                module.plot_profiles(x, t_s, u_hist)
        finally:
            if cwd != output_dir:
                import os
                os.chdir(cwd)

    def print_summary(self) -> None:
        print("TranscranialHybridPhysics")
        print(f"  L            : {self.L_m * 1e3:.1f} mm")
        print(f"  f0           : {self.f0_hz / 1e3:.1f} kHz")
        print(f"  t_ref        : {self.t_ref * 1e6:.3f} us")
        print(f"  T_phys       : {self.t_total_s * 1e6:.3f} us")
        print(f"  tau_max      : {self.tau_max:.5f}")
        print(f"  source_x     : {self.source_x_m * 1e3:.3f} mm")
        print(f"  source_sigma : {self.source_sigma_m * 1e3:.3f} mm")
        print(f"  layers       : {len(self.layers)}")
        for lyr in self.layers:
            print(
                f"  {lyr.name:10s} x=[{lyr.x_start_m * 1e3:.2f}, {lyr.x_end_m * 1e3:.2f}] mm "
                f"rho={lyr.rho:.1f} c={lyr.c:.1f} alpha={lyr.alpha_dBcmMHz:.2f} dB/cm/MHz"
            )
