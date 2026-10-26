# Conditional NF Visual Debugger

A static, browser-based debugger for the conditional normalizing flow. It reads
one telemetry file and does not import torch or MadGraph at runtime.

## Run

Regenerate the telemetry from the repository root. The MadGraph Fortran library
resolves `Cards/ident_card.dat` relative to the CWD, so the dump **must** run
from inside the standalone directory:

```bash
REPO=C:/Users/laiji/test/conditional-nf-integration
PY=C:/Users/laiji/.conda/envs/workspace/python.exe
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" "$PY" "$REPO/scripts/dump_web_telemetry.py"
```

That writes `web_visualizer/telemetry_data.json` and a `telemetry_data.js`
wrapper (so the page also works when opened directly from disk).

Serve the folder:

```bash
cd web_visualizer
python -m http.server 8000
# open http://127.0.0.1:8000/
```

Deep links are supported for sharing a state:

```
?energy=1950&step=3&beta=2&density=5000
```

`energy` is snapped to the nearest computed condition (400, 650, 1000, 1500 or
1950 GeV); `beta` is an index into [0.00, 0.08, 0.25, 0.55, 1.00]; `step` is
1–4; `density` is 500, 2000 or 5000.

## Panels

1. **Macro execution & data flow** — the call hierarchy from the driver through
   the search engine and the marginal/copula splines to the phase-space map and
   the Fortran `matrix2py` subprocess. Click a node to inspect its active
   function and device. The dashed band marks the `unit.detach().cpu().numpy()`
   boundary where GPU tensors return to the CPU for the matrix element.
2. **Coordinate-space evolution** — the same 5,000 base samples at each of the
   four stages: uniform base, marginals only, the full copula proposal, and the
   kinematic map. The blue field is the local density; the orange points are
   the samples. The 2→2 process makes the kinematic stage a one-dimensional
   curve in `p_T`–`η`, which the panel notes explicitly.
3. **Annealing bridge & target peaks** — the density
   `p_β(x) ∝ q₀(x)^{1−β} f(x)^β` on an 80 × 80 grid, with `q₀` uniform, plus its
   one-dimensional projection. At β = 0 the density is flat; at β = 1 it is the
   matrix-element target.
4. **Importance sampling diagnostics** — the histogram of `log₁₀ W`, and gauges
   for ESS/N, `Î_NF` versus the Sobol reference, and the pull. The histogram
   uses 2,000 trace samples; the gauges read the 500,000-sample validation
   summary, which is why they do not fluctuate with the density selector.

## Data

`telemetry_data.json` (3.8 MB) contains:

| key | contents |
|---|---|
| `meta` | checkpoint, device, dtype, conditions, β grid, sample counts, seed |
| `condition_meta` | per-energy marginal edge peaks (the 1.49 → 16.19 style diagnostic) |
| `stages` | per condition: `u`, `x_marg`, `u_copula`, `x_copula`, `kinematic.{pt,eta,mtt}` (5,000 samples) |
| `annealing` | per condition: `x0`, `x1`, `log10_density[β][x1][x0]` (80 × 80 × 5) |
| `weights` | per condition: `log10_w` (2,000 samples), `ess_sample`, `nf`, `sobol`, `pull` |

The model is the **smoke GPU** checkpoint
`results/gg_ttbar_copula_smoke_gpu/madgraph_gg_ttbar_2_final_model.pt`, as
specified. The validation gauges come from
`results/gg_ttbar_validation_smoke_gpu/summary.csv`.

The β grid starts at 0.00 for the flat-uniform visual; the smoke training
schedule itself starts at 0.02.
