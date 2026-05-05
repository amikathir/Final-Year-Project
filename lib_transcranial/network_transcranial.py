"""
Network architectures for the transcranial PINN.

Two classes:

  FourierFeatureEmbedding
    Maps (tau, x_hat) -> 128 sin/cos features at 32 log-uniform frequencies
    in [1, 300] rad/tau. Phases fixed to 0; frequencies frozen during
    training (register_buffer, NOT nn.Parameter). Frozen frequencies are
    the trick that fixes the gradient-explosion problem of learnable
    Fourier features at high omega (Tancik 2020 / SIREN-style failure).

  FourierMLP
    Embedding -> [Linear -> tanh] x 5 -> Linear (3 outputs).
    Hidden width 256 (~330 k params). Xavier-uniform init on hidden
    layers; small std=1e-3 normal init on the output head so the model
    starts near-zero (clean initial loss landscape).

Output is the raw 3-channel pre-ansatz tensor; the tau^2 hard-IC ansatz
on u, and the per-channel u_scale / q_scale / g_scale, are applied
downstream in pinn_transcranial.py.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fourier Feature Network
# ---------------------------------------------------------------------------

class FourierFeatureEmbedding(nn.Module):
    """
    Maps [tau, xhat] (N,2) → concatenated sin/cos features (N, 2*n_freqs).

    Frequencies are sampled log-uniformly over [omega_low, omega_high] to cover
    both slow envelope and fast 500-kHz carrier (omega_dim ≈ 149 rad/tau).
    Separate but identical frequency sets are used for tau and xhat.

    output size = 2 * n_freqs  (n_freqs sin + n_freqs cos for tau,
                                n_freqs sin + n_freqs cos for xhat  →  4 * (n_freqs//2))
    Wait — re-read: we have n_freqs//2 distinct frequencies; sin+cos gives n_freqs per
    input dimension; two input dimensions → 2 * n_freqs total features.
    """

    def __init__(
        self,
        n_freqs: int = 64,
        omega_low: float = 1.0,
        omega_high: float = 300.0,
        learnable: bool = False,
    ):
        super().__init__()
        n_half = n_freqs // 2
        log_freqs = torch.linspace(math.log(omega_low), math.log(omega_high), n_half)
        freqs = torch.exp(log_freqs)          # (n_half,)
        phases = torch.zeros(n_half)          # deterministic — no random seed issues

        if learnable:
            self.freqs  = nn.Parameter(freqs)
            self.phases = nn.Parameter(phases)
        else:
            self.register_buffer("freqs",  freqs)
            self.register_buffer("phases", phases)

        # Each dimension contributes n_half sin + n_half cos = n_freqs features.
        # Two dimensions (tau, xhat) → 2 * n_freqs features total.
        self.out_features = 2 * n_freqs

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        # tx: (N, 2)
        tau = tx[:, 0:1]
        x   = tx[:, 1:2]

        # (N, 1) × (1, n_half) → (N, n_half)
        args_tau = tau * self.freqs.unsqueeze(0) + self.phases.unsqueeze(0)
        args_x   = x   * self.freqs.unsqueeze(0) + self.phases.unsqueeze(0)

        tau_feat = torch.cat([torch.sin(args_tau), torch.cos(args_tau)], dim=-1)
        x_feat   = torch.cat([torch.sin(args_x),   torch.cos(args_x)],   dim=-1)
        return torch.cat([tau_feat, x_feat], dim=-1)

class FourierMLP(nn.Module):
    """
    Fourier Feature MLP.

    Architecture:
        FourierFeatureEmbedding → [Linear → Tanh] × hidden_layers → Linear

    The embedding pre-computes sin/cos features at log-uniform frequencies so the
    MLP only needs to learn amplitude/phase patterns, not the frequency structure.
    This avoids the gradient-explosion problem of deep SIREN at high omega_0.

    Default: n_freqs=64 → embed_dim=128, hidden=256, 4 layers → ~350K parameters.
    """

    def __init__(
        self,
        n_freqs: int = 64,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 3,
        omega_low: float = 1.0,
        omega_high: float = 300.0,
        learnable_freqs: bool = False,
    ):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(
            n_freqs=n_freqs,
            omega_low=omega_low,
            omega_high=omega_high,
            learnable=learnable_freqs,
        )

        embed_dim = self.embedding.out_features

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(hidden_layers):
            lin = nn.Linear(in_dim, hidden_features)
            # Tanh-friendly initialisation: Xavier uniform
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            in_dim = hidden_features

        final = nn.Linear(hidden_features, out_features)
        nn.init.zeros_(final.bias)
        nn.init.normal_(final.weight, std=1e-3)
        layers.append(final)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(x))
