"""
Generate a wavefield video from a trained PINN checkpoint.

Usage:
    python generate_video.py
    python generate_video.py --checkpoint pinn_transcranial_hybrid.pt --output wavefield.mp4
    python generate_video.py --output wavefield.gif   # GIF fallback (no ffmpeg needed)

Requires:
    - matplotlib, numpy, torch
    - ffmpeg (for .mp4 output) or Pillow (for .gif output)
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from lib_transcranial.physics import TranscranialHybridPhysics
from lib_transcranial.network_transcranial import FourierMLP
from lib_transcranial.pinn_transcranial import FirstOrderHybridPINN


def _sanitise_phys_cfg(cfg: dict) -> dict:
    """
    Translate legacy physics-config keys saved in older checkpoints to the
    names the current TranscranialHybridPhysics constructor accepts.

    Older checkpoints used the layer-name-as-prefix convention
    (``skull_exit_thickness_m``); the current code uses the side-as-prefix
    convention (``exit_skull_thickness_m``).  Anything still unrecognised
    after the rewrite is dropped with a warning so the script keeps running
    on legacy artefacts.
    """
    if not cfg:
        return {}
    valid = set(inspect.signature(TranscranialHybridPhysics.__init__).parameters)
    valid.discard("self")

    out: dict = {}
    for k, v in cfg.items():
        new_k = k
        # rewrite e.g. "skull_exit_thickness_m" → "exit_skull_thickness_m"
        if "_exit_" in k:
            head, tail = k.split("_exit_", 1)
            candidate = f"exit_{head}_{tail}"
            if candidate in valid:
                new_k = candidate
        if new_k in valid:
            out[new_k] = v
        else:
            print(f"  [warn] dropping unknown physics-config key: {k!r}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate wavefield video from PINN checkpoint")
    parser.add_argument("--checkpoint", type=str, default="pinn_transcranial_hybrid_normal.pt",
                        help="Path to .pt checkpoint (default: pinn_transcranial_hybrid_normal.pt)")
    parser.add_argument("--output", type=str, default="wavefield_video.mp4",
                        help="Output video path, .mp4 or .gif (default: wavefield_video.mp4)")
    parser.add_argument("--nt", type=int, default=300,
                        help="Number of time frames (default: 300)")
    parser.add_argument("--nx", type=int, default=400,
                        help="Number of spatial points (default: 400)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for output (default: 150)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Reconstruct physics from saved config (or use defaults for old checkpoints)
    phys_cfg = _sanitise_phys_cfg(ckpt.get("physics_config", {}))
    phys = TranscranialHybridPhysics(**phys_cfg) if phys_cfg else TranscranialHybridPhysics()
    phys.print_summary()

    # Reconstruct network
    net_cfg = ckpt.get("network_config", {})
    net_type = ckpt.get("network_type", "FourierMLP")
    if net_type != "FourierMLP":
        raise ValueError(f"Unknown network type: {net_type}")
    net = FourierMLP(**net_cfg) if net_cfg else FourierMLP()

    scales = ckpt["scales"]
    model = FirstOrderHybridPINN(
        network=net,
        phys=phys,
        u_scale=scales["u_scale"],
        q_scale=scales["q_scale"],
        g_scale=scales["g_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {net_type}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    # ------------------------------------------------------------------
    # Evaluate wavefield on grid
    # ------------------------------------------------------------------
    print(f"Evaluating wavefield on {args.nt} x {args.nx} grid ...")
    tau = torch.linspace(0.0, phys.tau_max, args.nt, device=device)
    x = torch.linspace(0.0, 1.0, args.nx, device=device)

    with torch.no_grad():
        u = model.evaluate_u_on_grid(tau, x, batch_points=40000).cpu().numpy()

    t_us = phys.tau_to_us(tau.cpu().numpy())
    x_mm = phys.xhat_to_mm(x.cpu().numpy())

    # Color scale from 99.5th percentile
    vmax = max(float(np.quantile(np.abs(u), 0.995)), 1e-20)
    print(f"  u range: [{u.min():.3e}, {u.max():.3e}], vmax={vmax:.3e}")

    # ------------------------------------------------------------------
    # Create animation
    # ------------------------------------------------------------------
    print("Creating animation ...")
    fig, (ax_profile, ax_field) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # Top panel: spatial profile at current time
    line, = ax_profile.plot(x_mm, u[0, :], "b-", lw=1.0)
    ax_profile.set_xlim(x_mm.min(), x_mm.max())
    ax_profile.set_ylim(-vmax * 1.2, vmax * 1.2)
    for mm in phys.x_interfaces_mm:
        ax_profile.axvline(mm, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax_profile.set_xlabel("x (mm)")
    ax_profile.set_ylabel("u (m)")
    title_text = ax_profile.set_title(f"t = {t_us[0]:.2f} \u00b5s")
    ax_profile.grid(True, alpha=0.3)

    # Bottom panel: x-t wavefield with moving time cursor
    im = ax_field.imshow(
        u.T,
        extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
        aspect="auto", origin="lower", cmap="seismic",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    for mm in phys.x_interfaces_mm:
        ax_field.axhline(mm, color="gray", ls="--", lw=0.8, alpha=0.7)
    cursor = ax_field.axvline(t_us[0], color="lime", lw=1.5, alpha=0.8)
    ax_field.set_xlabel("t (\u00b5s)")
    ax_field.set_ylabel("x (mm)")
    ax_field.set_title("Wavefield (x-t)")
    plt.colorbar(im, ax=ax_field, label="u (m)")

    plt.tight_layout()

    def update(frame):
        line.set_ydata(u[frame, :])
        title_text.set_text(f"t = {t_us[frame]:.2f} \u00b5s")
        cursor.set_xdata([t_us[frame], t_us[frame]])
        return line, title_text, cursor

    anim = FuncAnimation(fig, update, frames=args.nt,
                         interval=1000 // args.fps, blit=False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = args.output
    print(f"Saving to {output} ({args.nt} frames, {args.fps} fps, {args.dpi} dpi) ...")
    if output.endswith(".gif"):
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=2000)
    anim.save(output, writer=writer, dpi=args.dpi)
    print(f"Done: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
