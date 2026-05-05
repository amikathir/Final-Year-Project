"""
First-order hybrid acoustic PINN.

State variables (all dimensionless, units = metres after multiplying by scale):
    u  = displacement
    q  = du/dtau               (relation: r1 = u_tau - q = 0)
    g  = kappa_hat * du/dxhat  (relation: r2 = g - kappa_hat*u_x = 0)

Third PDE equation:
    r3 = q_tau + (2*alpha_hat + sponge_hat)*q - g_x/rho_hat - source_hat = 0
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


class FirstOrderHybridPINN(nn.Module):

    def __init__(
        self,
        network: nn.Module,
        phys,
        u_scale: float = 1.0,
        q_scale: float = 1.0,
        g_scale: float = 1.0,
        pde_source_enabled: bool = True,
    ):
        super().__init__()
        self.network = network
        self.phys    = phys
        self.pde_source_enabled = pde_source_enabled

        # Per-channel amplitude scales (physical units = metres)
        self.register_buffer("u_scale", torch.tensor(float(u_scale)))
        self.register_buffer("q_scale", torch.tensor(float(q_scale)))
        self.register_buffer("g_scale", torch.tensor(float(g_scale)))

        # Dimensionless carrier frequency  ω_dim = 2π f₀ t_ref ≈ 149
        # Used to normalise r3 (whose dominant term q_tau ~ q_scale * omega_dim)
        omega_dim = float(2.0 * math.pi * phys.f0_hz * phys.t_ref)
        self.register_buffer("omega_dim", torch.tensor(omega_dim))

        # Pre-tabulated material properties on a fine grid
        x_grid = np.linspace(0.0, 1.0, 4000, dtype=np.float32)
        self.register_buffer("x_grid",         torch.tensor(x_grid[:, None]))
        self.register_buffer("rho_hat_grid",   torch.tensor(phys.rho_hat_np(x_grid).astype(np.float32)[:, None]))
        self.register_buffer("kappa_hat_grid", torch.tensor(phys.kappa_hat_np(x_grid).astype(np.float32)[:, None]))
        self.register_buffer("alpha_hat_grid", torch.tensor(phys.alpha_hat_np(x_grid).astype(np.float32)[:, None]))
        self.register_buffer("sponge_hat_grid",torch.tensor(phys.sponge_hat_np(x_grid).astype(np.float32)[:, None]))

    # ------------------------------------------------------------------
    # Material property interpolation
    # ------------------------------------------------------------------
    def _interp1(self, x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        xq  = torch.clamp(x, 0.0, 1.0).squeeze(-1)
        n   = values.shape[0]
        pos = xq * (n - 1)
        i0  = torch.floor(pos).long()
        i1  = torch.clamp(i0 + 1, max=n - 1)
        w   = (pos - i0.float()).unsqueeze(-1)
        return (1.0 - w) * values[i0] + w * values[i1]

    def rho_hat(self, x):    return self._interp1(x, self.rho_hat_grid)
    def kappa_hat(self, x):  return self._interp1(x, self.kappa_hat_grid)
    def alpha_hat(self, x):  return self._interp1(x, self.alpha_hat_grid)
    def sponge_hat(self, x): return self._interp1(x, self.sponge_hat_grid)

    def source_hat(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.pde_source_enabled:
            return torch.zeros_like(tau)
        tau_np = tau.detach().cpu().numpy().squeeze(-1)
        x_np   = x.detach().cpu().numpy().squeeze(-1)
        s = self.phys.source_hat_np(tau_np, x_np).astype(np.float32)
        return torch.tensor(s[:, None], dtype=torch.float32, device=tau.device)

    def set_pde_source_enabled(self, enabled: bool) -> None:
        self.pde_source_enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def _ansatz_model_normalized(self, tx: torch.Tensor) -> torch.Tensor:
        """
        Returns (u_norm, q_norm, g_norm) each O(1).

        u uses hard IC ansatz tau^2 * u_raw  → u(tau=0, x) = 0 exactly.
        q, g use NO ansatz  → soft IC penalty handles q(0)=g(0)=0.

        Rationale for removing tau ansatz from q, g:
          - q ansatz: tau * q_raw → near-zero gradient for q_raw at small tau,
            where FDTD-derived q is noisiest (finite-difference of u at τ≈0).
          - g ansatz: same issue.
          - Soft IC loss (penalising q_norm, g_norm at tau=0) is cleaner.
        """
        tau = tx[:, 0:1]
        raw = self.network(tx)          # (N, 3)  O(1) raw outputs

        u_norm = (tau ** 2) * raw[:, 0:1]   # hard zero IC for u
        q_norm = raw[:, 1:2]                 # soft IC via ic_loss
        g_norm = raw[:, 2:3]                 # soft IC via ic_loss
        return torch.cat([u_norm, q_norm, g_norm], dim=1)

    def forward_normalized(self, tx: torch.Tensor) -> torch.Tensor:
        """Return (u_norm, q_norm, g_norm) in O(1) normalized space."""
        return self._ansatz_model_normalized(tx)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        """Return (u, q, g) scaled back to physical-dimensionless units (metres)."""
        y = self._ansatz_model_normalized(tx)
        return torch.cat([
            y[:, 0:1] * self.u_scale,
            y[:, 1:2] * self.q_scale,
            y[:, 2:3] * self.g_scale,
        ], dim=1)

    # ------------------------------------------------------------------
    # PDE residuals  (computed in physical space)
    # ------------------------------------------------------------------
    def _grad(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]

    def residuals(self, tx: torch.Tensor):
        tx   = tx.clone().detach().requires_grad_(True)
        tau  = tx[:, 0:1];  x = tx[:, 1:2]
        y    = self.forward(tx)
        u    = y[:, 0:1];   q = y[:, 1:2];  g = y[:, 2:3]

        grad_u = self._grad(u, tx);  u_tau = grad_u[:, 0:1];  u_x = grad_u[:, 1:2]
        grad_q = self._grad(q, tx);  q_tau = grad_q[:, 0:1]
        grad_g = self._grad(g, tx);  g_x   = grad_g[:, 1:2]

        rho_hat    = torch.clamp(self.rho_hat(x),   min=1e-6)
        kappa_hat  = torch.clamp(self.kappa_hat(x), min=1e-6)
        alpha_hat  = self.alpha_hat(x)
        sponge_hat = self.sponge_hat(x)
        src_hat    = self.source_hat(tau, x)

        r1 = u_tau - q                                                        # scale ~ q_scale
        r2 = g - kappa_hat * u_x                                              # scale ~ g_scale
        r3 = q_tau + (2.0 * alpha_hat + sponge_hat) * q - g_x / rho_hat - src_hat
        # r3 dominant term q_tau ~ q_scale * omega_dim
        return r1, r2, r3

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def state_losses(
        self,
        tx_data: torch.Tensor,
        u_data_norm: torch.Tensor,   # already divided by u_scale
        q_data_norm: torch.Tensor,   # already divided by q_scale
        g_data_norm: torch.Tensor,   # already divided by g_scale
    ):
        """All comparisons in normalized O(1) space → balanced MSE loss."""
        y      = self.forward_normalized(tx_data)
        u_loss = torch.mean((y[:, 0:1] - u_data_norm) ** 2)
        q_loss = torch.mean((y[:, 1:2] - q_data_norm) ** 2)
        g_loss = torch.mean((y[:, 2:3] - g_data_norm) ** 2)
        state_loss = u_loss + q_loss + g_loss
        return state_loss, u_loss, q_loss, g_loss

    def pde_losses(self, tx_pde: torch.Tensor | None):
        if tx_pde is None or tx_pde.numel() == 0:
            z = torch.zeros((), device=self.x_grid.device)
            return z, z, z, z

        r1, r2, r3 = self.residuals(tx_pde)

        # Normalize each residual by the natural scale of its dominant term:
        #   r1 ~ q_scale           → divide by q_scale
        #   r2 ~ g_scale           → divide by g_scale
        #   r3 ~ q_scale * omega   → divide by q_scale * omega_dim
        r1_loss = torch.mean((r1 / self.q_scale) ** 2)
        r2_loss = torch.mean((r2 / self.g_scale) ** 2)
        r3_loss = torch.mean((r3 / (self.q_scale * self.omega_dim)) ** 2)
        pde_loss = r1_loss + r2_loss + r3_loss
        return pde_loss, r1_loss, r2_loss, r3_loss

    def ic_loss(self, device: torch.device, dtype: torch.dtype, n_samples: int = 512):
        """Soft IC: enforce q=0, g=0 at tau=0 (u=0 is guaranteed by tau^2 ansatz)."""
        tx0 = torch.cat([
            torch.zeros((n_samples, 1), dtype=dtype, device=device),
            torch.rand( (n_samples, 1), dtype=dtype, device=device),
        ], dim=1)
        y = self.forward_normalized(tx0)
        return torch.mean(y[:, 1:3] ** 2)   # penalise q_norm and g_norm at tau=0

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_u_normalized(self, tx: torch.Tensor, batch_points: int = 16000) -> torch.Tensor:
        """Return u in normalized (O(1)) space — use for trace loss comparison."""
        outs = []
        for i in range(0, tx.shape[0], batch_points):
            outs.append(self.forward_normalized(tx[i : i + batch_points])[:, 0:1])
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def evaluate_u(self, tx: torch.Tensor, batch_points: int = 16000) -> torch.Tensor:
        """Return u in physical-dimensionless units (metres)."""
        outs = []
        for i in range(0, tx.shape[0], batch_points):
            outs.append(self.forward(tx[i : i + batch_points])[:, 0:1])
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def evaluate_u_on_grid(
        self,
        tau: torch.Tensor,
        x: torch.Tensor,
        batch_points: int = 30000,
    ) -> torch.Tensor:
        """Return u on a (tau, x) grid in physical-dimensionless units."""
        T, X = torch.meshgrid(tau, x, indexing="ij")
        tx = torch.stack([T.reshape(-1), X.reshape(-1)], dim=-1)
        u  = self.evaluate_u(tx, batch_points=batch_points)
        return u.reshape(T.shape)