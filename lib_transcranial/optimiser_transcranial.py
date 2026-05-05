"""
Trainer for the hybrid PINN surrogate.

Pipeline (one Adam step):
  1. Sample supervised anchors  (random tiles  OR  windowed snapshots).
  2. Sample PDE collocation points (uniform random over (tau, x_hat)).
  3. Compute state_loss + pde_loss + trace_loss + ic_loss.
  4. Optionally apply Wang-Sifan 2022 causal weighting to the PDE residual.
  5. backward(); grad-clip @ max_norm=1.0; optim.step(); scheduler.step().

Outer driver:
  HybridTrainer.train(stages, dataset, traces)  runs three Adam stages
  (supervised_full, pde_refinement, cooldown), then
  HybridTrainer.lbfgs_refine(...)  does 5 outer L-BFGS Wolfe iterations.

Module organisation:
  - StageConfig         dataclass of every per-stage knob
  - HybridTrainer       owns samplers, loss eval, train loop, LBFGS polish

Key fix vs previous version:
  _trace_loss now calls model.evaluate_u_normalized() and compares against
  pre-normalized traces["u_norm"].  The old code compared physical-scale u
  (~1e-8 m) against a normalized state loss (~1), making the trace gradient
  effectively zero (1e-16 contribution) relative to the state loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class StageConfig:
    name: str
    steps: int
    lr: float
    n_tiles: int
    tile_nt: int
    tile_nx: int
    n_pde: int
    state_weight: float
    pde_weight: float
    trace_weight: float
    ic_weight: float
    tau_max: float = 1.0           # PDE collocation time horizon
    pde_source_enabled: bool = True
    log_every: int = 100
    super_tau_max: float | None = None
    snapshot_times_tau: tuple[float, ...] | None = None
    n_points_per_snapshot: int = 3600

    snapshot_window_tau: float = 0.0
    causal_eps: float = 0.0
    n_causal_bins: int = 32


class HybridTrainer:
    """
    Tile-based supervised trainer with optional PDE regularisation.
    """

    def __init__(
        self,
        model,
        phys,
        device: torch.device,
        seed: int = 42,
        max_batch_points: int = 32000,
    ):
        self.model = model
        self.phys  = phys
        self.device = device
        self.max_batch_points = int(max_batch_points)
        self.rng = np.random.default_rng(seed)
        self.history: dict[str, list[float]] = {
            k: [] for k in ["total", "state", "u", "q", "g",
                             "pde", "r1", "r2", "r3", "trace", "ic"]
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_history(self, **kw):
        for k in self.history:
            self.history[k].append(float(kw.get(k, 0.0)))

    def _tau_limit(self, ds: dict, tau_max: float) -> int:
        idx = int(np.searchsorted(ds["tau"], tau_max, side="right"))
        return max(2, min(idx, ds["tau"].size))

    def _rng_int(self, lo: int, hi_exclusive: int) -> int:
        if hi_exclusive <= lo + 1:
            return lo
        return int(self.rng.integers(lo, hi_exclusive))

    # ------------------------------------------------------------------
    # Tile sampling
    # ------------------------------------------------------------------
    def _sample_tile(
        self, ds: dict, tile_nt: int, tile_nx: int, tau_max: float, mode: str,
        super_tau_max: float | None = None,
    ) -> dict:
        tau  = ds["tau"];    xhat = ds["xhat"]
        u    = ds["u"];      q    = ds["q"];    g = ds["g"]

        # Supervised tiles are restricted to [0, super_tau_max] when set,
        # implementing Wang/Rasht-Behesht "early-time snapshots only" anchoring.
        sup_cap = float(super_tau_max) if super_tau_max is not None else float(tau_max)
        nt_lim = self._tau_limit(ds, sup_cap)
        nt     = min(tile_nt, nt_lim)
        nx     = min(tile_nx, xhat.size)

        src_ix  = int(np.argmin(np.abs(xhat - self.phys.source_xhat)))
        ifc_ixs = [int(np.argmin(np.abs(xhat - xi))) for xi in self.phys.x_interfaces_xhat]

        if mode == "source":
            # Focus early-time tiles near source region
            t0_max = max(1, min(nt_lim - nt + 1, max(2, nt_lim // 3)))
            i_t    = self._rng_int(0, t0_max)
            center = src_ix
            i_x    = max(0, min(xhat.size - nx, center - nx // 2))
        elif mode == "interface":
            t0_max = max(1, nt_lim - nt + 1)
            i_t    = self._rng_int(0, t0_max)
            center = ifc_ixs[self._rng_int(0, len(ifc_ixs))]
            jitter = self._rng_int(-max(1, nx // 8), max(2, nx // 8 + 1))
            i_x    = max(0, min(xhat.size - nx, center - nx // 2 + jitter))
        else:  # full / uniform
            t0_max = max(1, nt_lim - nt + 1)
            i_t    = self._rng_int(0, t0_max)
            i_x    = self._rng_int(0, max(1, xhat.size - nx + 1))

        tau_tile  = tau [i_t : i_t + nt]
        xhat_tile = xhat[i_x : i_x + nx]
        T, X = np.meshgrid(tau_tile, xhat_tile, indexing="ij")
        tx = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)

        sl = (slice(i_t, i_t + nt), slice(i_x, i_x + nx))
        return dict(
            tx = tx,
            u  = u[sl].ravel()[:, None].astype(np.float32),
            q  = q[sl].ravel()[:, None].astype(np.float32),
            g  = g[sl].ravel()[:, None].astype(np.float32),
        )

    def _sample_tiles_batch(self, ds: dict, stage: StageConfig):
        n_src = max(1, int(round(0.5 * stage.n_tiles)))
        n_ifc = max(1, int(round(0.3 * stage.n_tiles)))
        n_ful = max(1, stage.n_tiles - n_src - n_ifc)
        modes = ["source"] * n_src + ["interface"] * n_ifc + ["full"] * n_ful
        self.rng.shuffle(modes)

        tiles = [self._sample_tile(ds, stage.tile_nt, stage.tile_nx,
                                   stage.tau_max, m,
                                   super_tau_max=stage.super_tau_max) for m in modes]
        tx = np.concatenate([t["tx"] for t in tiles], axis=0)
        u  = np.concatenate([t["u"]  for t in tiles], axis=0)
        q  = np.concatenate([t["q"]  for t in tiles], axis=0)
        g  = np.concatenate([t["g"]  for t in tiles], axis=0)
        dev = self.device
        return (
            torch.tensor(tx, dtype=torch.float32, device=dev),
            torch.tensor(u,  dtype=torch.float32, device=dev),
            torch.tensor(q,  dtype=torch.float32, device=dev),
            torch.tensor(g,  dtype=torch.float32, device=dev),
        )

    def _sample_snapshots_batch(self, ds: dict, stage: StageConfig):
        assert stage.snapshot_times_tau is not None
        tau_grid  = ds["tau"]
        xhat_grid = ds["xhat"]
        u = ds["u"];  q = ds["q"];  g = ds["g"]
        nx = xhat_grid.size
        m  = min(int(stage.n_points_per_snapshot), nx)
        w  = float(stage.snapshot_window_tau)

        tx_all, u_all, q_all, g_all = [], [], [], []
        for t_tau in stage.snapshot_times_tau:
            t_c = float(t_tau)
            if w > 0.0:
                # Window mode: pick every tau index within ±w of the centre.
                lo = max(float(tau_grid[0]),  t_c - w)
                hi = min(float(tau_grid[-1]), t_c + w)
                i_lo = int(np.searchsorted(tau_grid, lo, side="left"))
                i_hi = int(np.searchsorted(tau_grid, hi, side="right"))
                i_hi = max(i_lo + 1, i_hi)
                time_indices = np.arange(i_lo, i_hi, dtype=np.int64)
            else:
                # Single-instant mode (Rasht-Behesht 2022 spec).
                time_indices = np.array(
                    [int(np.argmin(np.abs(tau_grid - t_c)))], dtype=np.int64
                )

            # Re-sampled at every step → spatial stochasticity.
            idx_x = self.rng.choice(nx, size=m, replace=False)
            xhat_col = xhat_grid[idx_x].astype(np.float32)

            for i_t in time_indices:
                tau_col = np.full(m, tau_grid[i_t], dtype=np.float32)
                tx_all.append(np.stack([tau_col, xhat_col], axis=1).astype(np.float32))
                u_all.append(u[i_t, idx_x].astype(np.float32)[:, None])
                q_all.append(q[i_t, idx_x].astype(np.float32)[:, None])
                g_all.append(g[i_t, idx_x].astype(np.float32)[:, None])

        tx = np.concatenate(tx_all, axis=0)
        u_ = np.concatenate(u_all,  axis=0)
        q_ = np.concatenate(q_all,  axis=0)
        g_ = np.concatenate(g_all,  axis=0)
        dev = self.device
        return (
            torch.tensor(tx, dtype=torch.float32, device=dev),
            torch.tensor(u_, dtype=torch.float32, device=dev),
            torch.tensor(q_, dtype=torch.float32, device=dev),
            torch.tensor(g_, dtype=torch.float32, device=dev),
        )

    # ------------------------------------------------------------------
    # PDE collocation points
    # ------------------------------------------------------------------
    def _sample_pde(self, n: int, tau_max: float) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        n_early = int(0.5 * n)
        n_ifc   = int(0.3 * n)
        n_uni   = n - n_early - n_ifc

        tau_e = tau_max * self.rng.random((n_early, 1), dtype=np.float32) ** 2
        x_e   = np.clip(
            self.phys.source_xhat + 0.08 * self.rng.standard_normal((n_early, 1)).astype(np.float32),
            0.0, 1.0,
        )
        if n_ifc > 0:
            ctrs = self.rng.choice(self.phys.x_interfaces_xhat, n_ifc, replace=True).astype(np.float32)
            x_i  = np.clip(ctrs[:, None] + 0.05 * self.rng.standard_normal((n_ifc, 1)).astype(np.float32), 0.0, 1.0)
            tau_i = tau_max * self.rng.random((n_ifc, 1), dtype=np.float32)
        else:
            tau_i = x_i = np.empty((0, 1), dtype=np.float32)

        tau_u = tau_max * self.rng.random((n_uni, 1), dtype=np.float32)
        x_u   = self.rng.random((n_uni, 1), dtype=np.float32)

        tx = np.vstack([
            np.hstack([tau_e, x_e]),
            np.hstack([tau_i, x_i]),
            np.hstack([tau_u, x_u]),
        ]).astype(np.float32)
        return torch.tensor(tx, dtype=torch.float32, device=self.device)

    def _pde_losses_causal(
        self,
        tx_pde: torch.Tensor,
        tau_max: float,
        eps: float,
        n_bins: int,
    ):
        if tx_pde is None or tx_pde.numel() == 0:
            z = torch.zeros((), device=self.device)
            return z, z, z, z, z

        r1, r2, r3 = self.model.residuals(tx_pde)

        u_scale     = self.model.u_scale
        q_scale     = self.model.q_scale
        g_scale     = self.model.g_scale
        omega_dim   = self.model.omega_dim

        # Per-point squared residuals, each normalised by its dominant scale.
        s1 = (r1 / q_scale).pow(2).squeeze(-1)
        s2 = (r2 / g_scale).pow(2).squeeze(-1)
        s3 = (r3 / (q_scale * omega_dim)).pow(2).squeeze(-1)

        tau = tx_pde[:, 0]
        idx = torch.clamp(
            (tau / max(tau_max, 1e-12) * n_bins).long(),
            max=n_bins - 1, min=0,
        )

        zeros = torch.zeros(n_bins, device=self.device, dtype=s1.dtype)
        counts = torch.zeros(n_bins, device=self.device, dtype=s1.dtype).scatter_add_(
            0, idx, torch.ones_like(s1)
        )
        # Per-bin mean of (r1^2 + r2^2 + r3^2) — total normalised residual.
        sum_per_bin = zeros.clone().scatter_add_(0, idx, s1 + s2 + s3)
        L_per_bin = sum_per_bin / counts.clamp(min=1.0)
        # Bins with no points contribute neither to the loss nor to the sum.
        nonempty = (counts > 0).to(L_per_bin.dtype)

        # Cumulative loss strictly BEFORE bin i (exclusive prefix sum).
        # We detach so the weights only modulate the loss, not its gradient.
        cum_excl = torch.cumsum(L_per_bin.detach() * nonempty, dim=0) - L_per_bin.detach() * nonempty
        w = torch.exp(-float(eps) * cum_excl)            # w_1 = exp(0) = 1
        w = w * nonempty                                  # zero empty bins

        denom = nonempty.sum().clamp(min=1.0)
        pde_loss = (w * L_per_bin).sum() / denom

        # Per-component losses (no causal weighting, for logging only)
        r1_loss = s1.mean()
        r2_loss = s2.mean()
        r3_loss = s3.mean()

        # Stopping signal: smallest non-empty weight (≈ 1 ⇒ converged everywhere).
        with torch.no_grad():
            w_min = w[nonempty > 0].min() if (nonempty > 0).any() else torch.tensor(0.0, device=self.device)

        return pde_loss, r1_loss, r2_loss, r3_loss, w_min

    # ------------------------------------------------------------------
    # Trace loss
    # ------------------------------------------------------------------
    def _trace_loss(self, traces: dict | None, tau_max: float) -> torch.Tensor:
        """
        traces["u_norm"] must already be divided by u_scale (normalized).
        model.evaluate_u_normalized returns u in the same O(1) space.
        This ensures the trace loss is commensurate with the state loss.
        """
        zero = torch.zeros((), device=self.device)
        if traces is None:
            return zero

        tau    = traces["tau"]
        valid  = tau <= tau_max
        tau_v  = tau[valid]
        if tau_v.size == 0:
            return zero

        losses = []
        for i, xhat_val in enumerate(traces["xhat"]):
            tx = np.stack(
                [tau_v, np.full_like(tau_v, xhat_val, dtype=np.float32)],
                axis=1,
            ).astype(np.float32)
            tx_t   = torch.tensor(tx, dtype=torch.float32, device=self.device)
            u_ref  = torch.tensor(
                traces["u_norm"][valid, i : i + 1],
                dtype=torch.float32, device=self.device,
            )
            u_pred = self.model.evaluate_u_normalized(tx_t, batch_points=self.max_batch_points)
            losses.append(torch.mean((u_pred - u_ref) ** 2))
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        stages: list[StageConfig],
        dataset: dict,
        traces: dict | None = None,
    ):
        for stage in stages:
            print("\n" + "=" * 78)
            if stage.snapshot_times_tau is not None:
                sup_desc = (f"snapshots={len(stage.snapshot_times_tau)}"
                            f"×{stage.n_points_per_snapshot}")
            else:
                sup_desc = f"tiles={stage.n_tiles}×{stage.tile_nt}×{stage.tile_nx}"
            print(f"Stage: {stage.name}  steps={stage.steps}  lr={stage.lr:.1e}  "
                  f"{sup_desc}  n_pde={stage.n_pde}  "
                  f"tau_max={stage.tau_max:.4f}  causal_eps={stage.causal_eps:.1f}")
            print("=" * 78)

            self.model.set_pde_source_enabled(stage.pde_source_enabled)
            optim = torch.optim.Adam(self.model.parameters(), lr=stage.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=stage.steps, eta_min=stage.lr * 0.05
            )

            for step in range(1, stage.steps + 1):
                optim.zero_grad(set_to_none=True)

                if stage.snapshot_times_tau is not None:
                    tx_data, u_data, q_data, g_data = self._sample_snapshots_batch(dataset, stage)
                else:
                    tx_data, u_data, q_data, g_data = self._sample_tiles_batch(dataset, stage)
                tx_pde = self._sample_pde(stage.n_pde, stage.tau_max)

                state_loss, u_loss, q_loss, g_loss = self.model.state_losses(
                    tx_data, u_data, q_data, g_data
                )
                if stage.causal_eps > 0.0 and tx_pde.numel() > 0:
                    pde_loss, r1_loss, r2_loss, r3_loss, w_min = self._pde_losses_causal(
                        tx_pde,
                        tau_max=stage.tau_max,
                        eps=stage.causal_eps,
                        n_bins=stage.n_causal_bins,
                    )
                else:
                    pde_loss, r1_loss, r2_loss, r3_loss = self.model.pde_losses(tx_pde)
                    w_min = torch.tensor(0.0, device=self.device)
                trace_loss = self._trace_loss(traces, stage.tau_max)
                ic_loss    = self.model.ic_loss(device=self.device, dtype=tx_data.dtype)

                total = (
                    stage.state_weight * state_loss
                    + stage.pde_weight  * pde_loss
                    + stage.trace_weight* trace_loss
                    + stage.ic_weight   * ic_loss
                )

                if not torch.isfinite(total):
                    print(f"  [skip] non-finite loss at step {step}")
                    continue

                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()
                sched.step()

                if step == 1 or step % stage.log_every == 0 or step == stage.steps:
                    vals = dict(
                        total=float(total), state=float(state_loss),
                        u=float(u_loss), q=float(q_loss), g=float(g_loss),
                        pde=float(pde_loss),
                        r1=float(r1_loss), r2=float(r2_loss), r3=float(r3_loss),
                        trace=float(trace_loss), ic=float(ic_loss),
                    )
                    self._append_history(**vals)
                    extra = (f"  w_min={float(w_min):.3f}"
                             if stage.causal_eps > 0.0 else "")
                    print(
                        f"  [{stage.name}] step={step:6d}  "
                        f"total={vals['total']:.3e}  "
                        f"state={vals['state']:.3e} "
                        f"(u={vals['u']:.3e} q={vals['q']:.3e} g={vals['g']:.3e})  "
                        f"trace={vals['trace']:.3e}  pde={vals['pde']:.3e}  "
                        f"ic={vals['ic']:.3e}" + extra
                    )

    # ------------------------------------------------------------------
    # LBFGS refinement
    # ------------------------------------------------------------------
    def lbfgs_refine(self, ds: dict, max_iter: int = 300, tau_max: float | None = None):
        tau_max = tau_max or float(ds["tau"].max())
        stage   = StageConfig(
            name="lbfgs", steps=1, lr=1.0,
            n_tiles=10, tile_nt=256, tile_nx=256,
            n_pde=0, state_weight=1.0, pde_weight=0.0,
            trace_weight=0.0, ic_weight=0.0,
            tau_max=tau_max,
        )
        optim = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=max_iter,
            tolerance_grad=1e-10,
            tolerance_change=1e-13,
            history_size=60,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optim.zero_grad()
            tx, u, q, g = self._sample_tiles_batch(ds, stage)
            loss, *_ = self.model.state_losses(tx, u, q, g)
            loss.backward()
            return loss

        print("\n=== LBFGS refinement ===")
        for i in range(5):
            loss_val = optim.step(closure)
            print(f"  LBFGS outer iter {i+1}: loss = {float(loss_val):.4e}")