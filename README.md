# Transcranial Hybrid PINN

A Fourier-feature, first-order Physics-Informed Neural Network for **1D
transcranial ultrasound wave propagation**, supervised by a conservative
FDTD reference solver. Produces a continuous, differentiable surrogate of
the displacement wavefield through a five-layer head model
(scalp - skull - brain - skull - scalp) at a 500 kHz carrier.

> *This is the BMED4010 Final Year Project of Amaresh Kathiresan
> (HKU EEE, 2025/26), supervised by A/Prof. Wei-Ning Lee, co-supervised by
> Dr Rachel Kwan, with technical mentorship from Dr Haotian Guan.*

---

## What this code does

End-to-end pipeline:

```
        FDTD_1D_transcranial_500kHz.py
                    │
                    ▼
    transcranial_fdtd1d_reference_full.npz   ← high-fidelity reference
                    │
                    ▼
           preprocess_fdtd_data.py
                    │
                    ▼
    pinn_training_full_normalized.npz        ← O(1)-normalised (u, q, g)
                    │
                    ▼
           main_transcranial.py              ← train PINN
                    │
                    ▼
    pinn_transcranial_hybrid_normal.pt       ← trained checkpoint
                    │
                    ├── generate_video.py            → wavefield_video.mp4
                    └── generate_report_figures.py   → report_figures/*.png
```

The PINN learns three coupled fields `(u, q, g)` — displacement, particle
velocity, and stiffness-weighted gradient — and is regularised by the
first-order acoustic residuals `r1 = u_τ − q`, `r2 = g − κ̂·u_x`,
`r3 = q_τ + 2α̂·q − g_x/ρ̂ − ŝ`.

---

## Repository layout

```
okada_pinn/
├── README.md                              ← this file
├── FDTD_1D_transcranial_500kHz.py         ← reference solver (5-layer leapfrog)
├── preprocess_fdtd_data.py                ← FDTD → (u, q, g) normalised tiles
├── main_transcranial.py                   ← training entry point
├── generate_video.py                      ← wavefield animation from .pt
├── generate_report_figures.py             ← all FYP report figures
│
└── lib_transcranial/
    ├── physics.py                         ← TranscranialHybridPhysics + tissue layers
    ├── network_transcranial.py            ← FourierFeatureEmbedding + FourierMLP
    ├── pinn_transcranial.py               ← FirstOrderHybridPINN model wrapper
    └── optimiser_transcranial.py          ← HybridTrainer + StageConfig + L-BFGS polish
```

Other top-level scripts (`main_2d_fat.py`, `main_fourier.py`, `cranial.py`, ...)
belong to earlier prototypes and are NOT part of the production transcranial
pipeline. The four files in `lib_transcranial/` plus the five top-level
scripts above are everything you need.

---

## Prerequisites

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0 (CUDA 11.8+ recommended for training; CPU works for
  inference)
- **NumPy**, **SciPy**, **Matplotlib**
- **ffmpeg** (`generate_video.py` falls back to GIF via Pillow if ffmpeg
  is missing)
- **GPU**: tested on an RTX 5070 (12 GB). Training fits in ~3 GB VRAM;
  any modern GPU will work.

Install with:

```bash
python -m venv .venv
source .venv/bin/activate         # Linux / WSL
# or  .venv\Scripts\activate      # Windows PowerShell

pip install --upgrade pip
pip install torch numpy scipy matplotlib pillow
```

---

## Quickstart — reproduce the headline result

From a fresh checkout:

```bash
# 1. Generate the FDTD reference (~8 s on CPU)
python FDTD_1D_transcranial_500kHz.py

# 2. Preprocess FDTD into normalised training tensors (~5 s)
python preprocess_fdtd_data.py

# 3. Train the PINN (~45 min on RTX 5070)
python main_transcranial.py

# 4. Animate the trained model (~30 s)
python generate_video.py

# 5. Regenerate the report figures (~1 min)
python generate_report_figures.py
```

Expected output after step 3:

```
[supervised_full]   step=10000   total ~ 4.9e-3   state ~ 4.5e-3 ...
[pde_refinement]    step= 5000   total ~ 6.2e-3   pde   ~ 1.0e-3 ...
[cooldown]          step= 2000   total ~ 3.8e-3   ...
=== LBFGS refinement ===
  LBFGS outer iter 1: loss = 2.3e-3
  ...
  LBFGS outer iter 5: loss = 4.9e-4
```

Headline metrics:
- Global relative L² error: **~7 %**
- Probe trace correlations: **0.98 - 0.997**
- Wall-clock training time: **~45 min** on RTX 5070

---

## Configuration

The most-edited knobs:

| Parameter | File | Line | Default | Effect |
|---|---|---|---|---|
| Carrier frequency | `main_transcranial.py` | CLI flag `--frequency` | 500 kHz | Source center frequency |
| Training mode | `main_transcranial.py` | CLI flag `--mode` | `normal` | `normal` (production) or `early_time` (time-marching) |
| Brain thickness | `lib_transcranial/physics.py` | 44 | 66.8 mm | Layer geometry; affects total L |
| Skull α | `lib_transcranial/physics.py` | 52 | 18 dB/cm/MHz | Material attenuation |
| Fourier band | `lib_transcranial/network_transcranial.py` | 34-35 | 1-300 rad/τ | Embedding spectrum |
| n_freqs | `lib_transcranial/network_transcranial.py` | 33 | 64 | Embedding dimensionality |
| Hidden layers / width | `lib_transcranial/network_transcranial.py` | 86-87 | 5 / 256 | MLP capacity |
| Stage 1 LR | `main_transcranial.py` | 418 | 5 × 10⁻⁴ | Adam supervised warm-up |
| Stage 2 PDE weight | `main_transcranial.py` | 429 | 0.005 | Physics regularisation strength |

To change the geometry to the older 73.2 mm configuration that matches the
report's slide 5:

```python
# lib_transcranial/physics.py line 44
brain_thickness_m: float = 46.8e-3,    # was 66.8
```

CLI arguments are listed in `python main_transcranial.py --help`.

---

## Outputs

After a full run the project root contains:

| File | Produced by | Contents |
|---|---|---|
| `transcranial_fdtd1d_reference_full.npz` | FDTD solver | `u_ref` (n_t, n_x), `t_us`, `x_mm`, source / geometry metadata |
| `transcranial_fdtd1d_reference.npz` | FDTD solver | Subsampled snapshot every 0.5 µs (lighter, for plots) |
| `pinn_training_full_normalized.npz` | preprocessor | `u, q, g` (O(1)-normalised), `tau, xhat`, `*_scale` factors |
| `pinn_training_traces_normalized.npz` | preprocessor | Probe time series at x = 1, 3.2, 13.2, 40, 70 mm |
| `pinn_transcranial_hybrid_normal.pt` | training | Model weights + `physics_config` + `network_config` + scales |
| `wavefield_video.mp4` | `generate_video.py` | Animated wavefield (FuncAnimation, 30 fps) |
| `report_figures/*.png` | `generate_report_figures.py` | All figures referenced by the FYP report |

---

## Architecture in one paragraph

`(τ, x̂) ∈ [0, τ_max] × [0, 1]` enters a fixed Fourier-feature layer that
expands to 128 sin/cos features at 32 log-uniform frequencies in
[1, 300] rad/τ; this is followed by a 5×256 tanh MLP with Xavier-uniform
init, projecting to 3 channels. A **hard τ² ansatz** on the displacement
channel guarantees `u(0, x) = u_τ(0, x) = 0` exactly; soft IC penalties
pull `q(0, x) ≈ g(0, x) ≈ 0`. The training loss combines state MSE on
random space-time tiles, PDE residuals at random collocation points,
trace MSE at five physical probes, and the IC penalty, scheduled across
three Adam stages followed by an L-BFGS polish.

See `pinn_transcranial.py` for the (u, q, g) wrapper and
`network_transcranial.py` for the embedding/MLP. Stage configurations
live in `main_transcranial.py:build_stages_normal()`.
---

## Training stages (normal mode)

| Stage | Steps | LR | PDE weight | Purpose |
|---|---|---|---|---|
| `supervised_full` | 10 000 | 5 × 10⁻⁴ | 0 | Pure data fit on FDTD tiles |
| `pde_refinement` | 5 000 | 1 × 10⁻⁴ | 0.005 | Activate physics residual at small weight |
| `cooldown` | 2 000 | 2 × 10⁻⁵ | 0 | Pure data polish at low LR |
| `lbfgs_refine` | 5 outer × 300 inner | strong-Wolfe | 0 | Quasi-Newton final polish |

LR cosine-anneals to `0.05 · LR` within each Adam stage. Gradient clip at
`max_norm = 1.0` is applied throughout.

The `--mode early_time` curriculum replaces the three Adam stages with
four time-marching stages of growing `tau_max`; useful when long-horizon
causal collapse is the failure mode.

---

## Known issues / roadmap

| Issue | File | Status |
|---|---|---|
| FDTD damping missing factor of `c` (under-attenuation by ≈ 1500-2800×) | `FDTD_1D_transcranial_500kHz.py:374-376` | Fix mapped; introduces PINN training instability without IC stiffening |
| PINN soft IC on `q, g` insufficient under stiff damping | `lib_transcranial/pinn_transcranial.py:102-103` | Recommended fix: `q_norm = τ·raw[:, 1:2]`, `g_norm = τ·raw[:, 2:3]` |
| Geometry mismatch (code: L = 93.2 mm, report: L = 73.2 mm) | `lib_transcranial/physics.py:44` | Pick one; recommend reverting brain_thickness to 46.8 mm |
| Frequency-independent attenuation around carrier | physics | Future work — Szabo power-law |
| Per-configuration training (not an operator) | architecture | Future work — DeepONet extension |

See the report's *Limitations* section for fuller discussion.

---

## Common commands

```bash
# Train with the alternative early-time curriculum
python main_transcranial.py --mode early_time --snapshot-spacing-us 3.0

# Train at a different carrier frequency
python main_transcranial.py --frequency 250

# Generate a GIF instead of MP4 (no ffmpeg needed)
python generate_video.py --output wavefield.gif

# Diagnose a checkpoint
python -c "
import torch
ck = torch.load('pinn_transcranial_hybrid_normal.pt', map_location='cpu',
                weights_only=False)
print('keys     :', list(ck.keys()))
print('phys_cfg :', ck.get('physics_config'))
print('net_cfg  :', ck.get('network_config'))
print('scales   :', ck['scales'])
"
```

---

## File-by-file reference

### `FDTD_1D_transcranial_500kHz.py`
Conservative explicit-leapfrog FDTD on a 2401-node grid. Heterogeneous
density `ρ(x)`, bulk modulus `κ(x)`, attenuation `α(x)`, and Gaussian-
modulated tone-burst source. Sponge layers absorb outgoing radiation.
Run directly to (re)generate the reference NPZ.

### `preprocess_fdtd_data.py`
Constructs `q = ∂u/∂τ` and `g = κ̂·∂u/∂x̂` from full-resolution FDTD `u` by
central differencing, *then* subsamples in time and space. Normalises each
channel to RMS = 1 and saves the per-channel scale factors. Run once after
each FDTD regeneration.

### `lib_transcranial/physics.py`
`TranscranialHybridPhysics` holds geometry, tissue tables, coordinate
conversions (`τ ↔ s`, `x̂ ↔ m`), and dimensionless versions of every
physical field. Pure NumPy. The PINN consumes only its tabulated buffers;
no PyTorch ops live here.

### `lib_transcranial/network_transcranial.py`
`FourierFeatureEmbedding` (32 log-uniform frequencies, frozen) feeds
`FourierMLP` (5 × 256 tanh, Xavier init, 3-channel head). ~330 k
parameters. Output is the raw 3-channel pre-ansatz tensor.

### `lib_transcranial/pinn_transcranial.py`
`FirstOrderHybridPINN` wraps the network with the τ² hard-IC ansatz on
`u`, soft-IC `q, g` (currently), per-channel `*_scale` buffers,
tabulated tissue properties, source term, and the three first-order
residuals. Hosts loss helpers (`state_losses`, `pde_losses`, `ic_loss`).

### `lib_transcranial/optimiser_transcranial.py`
`StageConfig` dataclass + `HybridTrainer` class. Implements tile / snapshot
samplers, PDE collocation sampler, optional Wang-Sifan causal weighting,
the staged Adam loop, and the L-BFGS polish.

### `main_transcranial.py`
CLI entry. Loads NPZ, builds physics + network + optimiser, calls
`HybridTrainer.train(stages, dataset, traces)`, then
`HybridTrainer.lbfgs_refine(...)`, saves the checkpoint, plots a
side-by-side wavefield comparison.

### `generate_video.py`
Loads a `.pt` checkpoint, queries the model on a (n_t × n_x) grid,
animates the spatial profile + space-time wavefield with a moving cursor.
Includes a backward-compatibility shim for legacy checkpoint key names
(`*_exit_*` ↔ `exit_*_*`).
---

## Citation

If you build on this work, please cite:

```bibtex
@thesis{kathiresan_2026_transcranial_pinn,
  author       = {Kathiresan, Amaresh},
  title        = {{FDTD}-Supervised Physics-Informed Neural Networks
                  for One-Dimensional Transcranial Ultrasound Wave Propagation},
  school       = {The University of Hong Kong},
  year         = {2026},
  type         = {{BEng} {Final} {Year} {Thesis}},
}
```
The Fourier-feature embedding draws on Tancik et al. 2020. The first-order
PINN formulation follows Raissi et al. 2019. The training curriculum and
stability mitigations follow Wang & Perdikaris 2022, Krishnapriyan et al.
2021, and Daw et al. 2023. Reference physics constants are from Pinton
et al. 2012 and Goss et al. 1978. The base code structure draws from okada39
at https://github.com/okada39/pinn_wave.
---

## License

MIT (or as governed by the HKU EEE department's project release policy).

---

## Acknowledgements

Primary supervisor Associate Professor Wei-Ning Lee, co-supervisor Dr Rachel Kwan,
PhD mentor Mr Haotian Guan, and the open-source PINN implementations of
Raissi, Wang, and okada39 that informed early prototyping.
