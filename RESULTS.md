# RESULTS — local Windows rebuild

*What was produced on this machine, and how it compares with the curated
results in `results_public/`.*

Machine: Windows 11, 16 logical cores, 15 GB RAM, **no GPU**.
Python 3.10.14 (conda `workspace`), torch 2.3.0+cpu.
MadGraph5_aMC@NLO 3.6.7 built locally with the MSYS2 MinGW-w64 toolchain.

---

## 1. What was completed

The Linux session (`PRD.md` Part II) built and ran **one** of the four
scattering processes. This session closed the breadth gap — **all three LO
partonic processes are now built, trained and validated**:

| process | dim | standalone | wrapper | integrand | trained | validated |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| gg → t t̄ | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| gg → t t̄ H | 5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| u ū → W⁺W⁻ → 4ℓ | 8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| fully decayed t t̄ H | 20 | — | — | — | — | — |

MadGraph now runs **natively on Windows** — no WSL, no MSYS2 login shell, no
Linux container. The matrix element is a real `.pyd` extension loaded by the
ordinary conda interpreter.

### Still open

* **Fully decayed 20-D t t̄ H.** Needs MadEvent and LHAPDF — a separate
  installation from the LO partonic standalones built here (`PRD.md` §16).
* **The missing Bayesian graph file.** `run_bayesian_selected_copula_benchmark.py`
  defaults to `results/bayesian_delta_edge_selection/median_probability_graph.csv`,
  and no script in this repository produces that path — the wiring between the
  edge-pruning experiment (which writes `final_graph_edges.csv` under
  `results/iterative_bayesian_edge_pruning_frozen_null/`) and this consumer was
  never committed. Closing it means running the pruning experiment (about an
  hour even reduced) and adding the adapter. `PRD.md` §6.1 flags the same gap.
  It affects a synthetic benchmark only, not the scattering results.

---

## 2. Integrand validation

`experiments/validate_madgraph_process_integrand.py`, 300–500 random points.
These numbers are a property of the generated matrix element and phase-space
map, so they are directly comparable with the Linux session.

| quantity | gg_ttbar (2D) | gg_tth (5D) | uu_ww_4l (8D) |
|---|---:|---:|---:|
| dimension | 2 | 5 | 8 |
| max 4-momentum residual | 2.27e-13 | 4.55e-13 | 4.55e-13 |
| max mass-shell residual | 3.49e-10 | 5.24e-10 | 3.49e-10 |
| minimum Jacobian | 0.0115863 | 0.0481910 | 0.4040888 |
| finite fraction | 1 | 1 | 1 |
| positive integrand fraction | 1 | 1 | 1 |
| throughput [points/s] | ~97 000 | ~53 000 | ~143 000 |

The `gg_ttbar` row reproduces the Linux run **to every printed digit**
(`max_momentum_residual = 2.27373675443e-13`,
`minimum_jacobian = 0.0115863199095`) — an independent confirmation that the
Windows rebuild produced the same matrix element. Throughput is about **2×**
the Linux figure (47 203 points/s there).

---

## 3. Published reference values

Taken from `results_public/`, produced by the authors with a fully trained
(`--speed-profile thorough`) model on CUDA. These are the comparison baseline
for §4.

### gg → t t̄ (`results_public/gg_ttbar_validation`)

| E [GeV] | Sobol [pb] | NF [pb] | ESS/N |
|---:|---:|---:|---:|
| 400 | 12.84494348 | 12.84584966 | 0.99843 |
| 650 | 16.91657141 | 16.91567916 | 0.99361 |
| 1000 | 11.36813869 | 11.36611982 | 0.98867 |
| 1500 | 6.860367541 | 6.857248807 | 0.98351 |
| 1950 | 4.776029942 | 4.775922990 | 0.98082 |

### gg → t t̄ H (`results_public/gg_tth_5d_validation`)

| E [GeV] | Sobol [pb] | NF [pb] | ESS/N |
|---:|---:|---:|---:|
| 500 | 2.685470079e-03 | 2.685431409e-03 | 0.98674 |
| 700 | 3.212181671e-02 | 3.211670672e-02 | 0.98337 |
| 1000 | 4.120062297e-02 | 4.119143938e-02 | 0.97805 |
| 1500 | 4.242535762e-02 | 4.241317648e-02 | 0.96971 |
| 1950 | 4.127233827e-02 | 4.125766702e-02 | 0.95898 |

### u ū → W⁺W⁻ → 4ℓ (`results_public/uu_ww_4l_validation`)

| E [GeV] | Sobol [pb] | NF [pb] | ESS/N |
|---:|---:|---:|---:|
| 300 | 5.299237502e-02 | 5.294113198e-02 | 0.94123 |
| 500 | 2.823612929e-02 | 2.825247097e-02 | 0.93957 |
| 800 | 1.472623360e-02 | 1.471579676e-02 | 0.93497 |
| 1200 | 7.999827156e-03 | 8.033685771e-03 | 0.93259 |
| 1950 | 3.786850757e-03 | 3.735747519e-03 | 0.92663 |

---

## 4. Training

`experiments/run_madgraph_process_conditional_copula.py --speed-profile smoke`,
run from inside each `generated/<proc>_standalone/`. Roughly 5 min wall per
process on 16 CPU cores.

| process | dim | learned graph | edges | ESS | Î | log w std |
|---|---:|:---|---:|---:|---:|---:|
| gg → t t̄ | 2 | `{0: [], 1: [0]}` | 1 | 0.80540 | 2.632163e-08 | 0.4975 |
| gg → t t̄ H | 5 | full AR | 10 | 0.81586 | 9.646322e-11 | 0.6997 |
| u ū → W⁺W⁻ → 4ℓ | 8 | full AR | 28 | 0.45349 | 3.697135e-11 | 1.1589 |

The 2-D graph collapses to a **single edge**, which is the expected outcome for
a two-body process: once each coordinate has its own conditional marginal there
is almost nothing left for the copula to model. Both higher-dimensional
processes keep the full autoregressive graph — the extra structure is real, not
noise.

The 8-D model trains to a noticeably lower ESS than the other two (0.45 against
0.81), with a correspondingly wider log-weight spread (0.50 → 1.16). Read this
as a statement about *this* model and *this* budget, not as a dimension law:
the three processes differ in resonance structure and matrix-element complexity
as well as in dimension, so nothing here isolates dimension as the cause. What
it does say is that the 8-D run ended its 300-step budget with a weaker
proposal than the other two — and §5.4 shows what that costs, which turns out
to be variance rather than accuracy.

### 4.1 Cross-machine check on `gg → t t̄`

| quantity | this session (Windows) | Linux session (`PRD.md` §15.3) |
|---|---:|---:|
| graph | `{0: [], 1: [0]}` | `{0: [], 1: [0]}` |
| edges | 1 | 1 |
| ESS | 0.80540 | 0.8072 |
| Î | 2.632163e-08 | 2.6337e-08 |
| log Î | −17.452875 | — |
| log w std | 0.497503 | 0.4935 |

Two independent builds of the matrix element, two different operating systems,
two different Fortran toolchains — and the trained proposal lands on the same
one-edge graph with ESS agreeing to **0.2 %** and the integral to **0.06 %**.
That is the strongest single check in this document: it exercises the generated
matrix element, the phase-space map, the α_s convention, the weighted-cache
construction and the whole annealing pipeline at once.

---

## 5. Held-out validation

`experiments/validate_madgraph_process_integral.py` with the published
defaults (`--sobol-power 18 --sobol-replicates 8 --nf-evals 500000
--seed 20260920`), so the Sobol column is directly comparable.

### 5.1 The Sobol column is identical to the published one

`gg → t t̄` (reference: `results_public/gg_ttbar_validation`)

| E [GeV] | mine [pb] | `results_public` [pb] | rel. diff |
|---:|---:|---:|---:|
| 400 | 12.84494348 | 12.84494348 | 0.00e+00 |
| 650 | 16.91657141 | 16.91657141 | 0.00e+00 |
| 1000 | 11.36813869 | 11.36813869 | 0.00e+00 |
| 1500 | 6.860367541 | 6.860367541 | 0.00e+00 |
| 1950 | 4.776029942 | 4.776029942 | 0.00e+00 |

`gg → t t̄ H` (reference: `results_public/gg_tth_5d_validation`)

| E [GeV] | mine [pb] | `results_public` [pb] | rel. diff |
|---:|---:|---:|---:|
| 500 | 0.002685470079 | 0.002685470079 | 0.00e+00 |
| 700 | 0.03212181671 | 0.03212181671 | 0.00e+00 |
| 1000 | 0.04120062297 | 0.04120062297 | 0.00e+00 |
| 1500 | 0.04242535762 | 0.04242535762 | 0.00e+00 |
| 1950 | 0.04127233827 | 0.04127233827 | 0.00e+00 |

Not "close" — **bit-identical, at every energy, for both processes**. The
randomized Sobol integral depends on the matrix element normalisation, the
external-particle ordering, the phase-space map, its Jacobian, α_s and the QMC
protocol. All of them match the authors' original build. This is the end-to-end
proof that the Windows/MinGW rebuild is faithful — and for the 5-D process it
also covers a genuinely non-trivial four-particle final state.

### 5.2 `gg → t t̄`: NF against the fully trained reference model

| E [GeV] | ESS/N (mine) | ESS/N (published) | Î (mine) [pb] | Î (published) [pb] | pull |
|---:|---:|---:|---:|---:|---:|
| 400 | 0.81244 | 0.99843 | 12.840842 | 12.845850 | −0.57 |
| 650 | 0.98024 | 0.99361 | 16.914919 | 16.915679 | −0.19 |
| 1000 | 0.96616 | 0.98867 | 11.369199 | 11.366120 | +0.89 |
| 1500 | 0.95037 | 0.98351 | 6.859430 | 6.857249 | +0.86 |
| 1950 | 0.96110 | 0.98082 | 4.7762168 | 4.775923 | +0.18 |

Every pull is below 0.9σ, and from 650 GeV up the reduced model lands within
**1–3 %** of the fully trained model's ESS.

The 400 GeV point is the outlier, and the reason is structural rather than a
method failure: the training condition domain is
√ŝ ∈ [360, 2000] GeV, so 400 GeV sits almost on the lower edge, where the flow
sees the least training data. `PRD.md` §15.4 reports the same edge behaviour
for the Linux smoke run (ESS 0.820 there vs 0.812 here).

### 5.3 `gg → t t̄ H`: NF against the fully trained reference model

| E [GeV] | ESS/N (mine) | ESS/N (published) | Î (mine) [pb] | Î (published) [pb] | pull |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.84114 | 0.98674 | 0.0026856195 | 0.0026854314 | +0.11 |
| 700 | 0.91694 | 0.98337 | 0.032121843 | 0.032116707 | +0.34 |
| 1000 | 0.90357 | 0.97805 | 0.041206888 | 0.041191439 | +0.74 |
| 1500 | 0.85647 | 0.96971 | 0.0424544 | 0.042413176 | +1.54 |
| 1950 | 0.79701 | 0.95898 | 0.04122193 | 0.041257667 | −1.12 |

Largest pull 1.54σ. Note that unlike the 2-D case these error bars are
*meaningful* — the Sobol stderr here is of order 1e-6 pb, not 1e-14, so the
pulls are not dominated by the NF side alone.

ESS degrades toward the top of the range (0.841 at 500 GeV down to 0.797 at
1950 GeV) whereas the reference model stays above 0.95 throughout. The 5-D
integrand has more structure for the copula to capture, so a 300-step budget
costs more here than it did in 2-D. That is the expected direction for a
training-budget shortfall; closing it needs the `thorough` profile.

### 5.4 `u ū → W⁺W⁻ → 4ℓ`: NF against the fully trained reference model

| E [GeV] | ESS/N (mine) | ESS/N (published) | Î (mine) [pb] | Î (published) [pb] | pull |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.81404 | 0.94123 | 0.052938873 | 0.052941132 | −0.06 |
| 500 | 0.86042 | 0.93957 | 0.028259296 | 0.028252471 | +0.36 |
| 800 | 0.86495 | 0.93497 | 0.014718625 | 0.014715797 | +0.29 |
| 1200 | 0.85626 | 0.93259 | 0.0080336289 | 0.0080336858 | −0.01 |
| 1950 | 0.80582 | 0.92663 | 0.0037370138 | 0.0037357475 | +0.42 |

The best-agreeing process of the three: **largest pull 0.42σ**. Even on a
300-step budget the doubly resonant 8-D integrand is reproduced accurately —
the learned proposal places samples well enough that the point estimates are
essentially indistinguishable from the fully trained reference.

What it costs is *variance*, not accuracy: ESS/N sits at 0.81–0.86 against the
published 0.93–0.94. That is the honest reading — a weaker proposal means wider
error bars at fixed sample count, and the error bars quoted above are the
model's own.

Note the contrast with the training-time ESS of 0.4535 (§4). The two numbers
measure different things and the training-time one is by far the more
pessimistic; the held-out estimator is what matters, and it is ~2× better.

### 5.5 Budget caveat

All models here are `--speed-profile smoke` (300 KL steps). The published
numbers come from the `thorough` profile (30 000 steps) — roughly 100× the
budget. The Sobol agreement in §5.1 is unaffected by this (it tests the
generated matrix element, not the flow); only the ESS and NF columns reflect
the reduced training.

### 5.6 Across the sweep

| process | dim | max pull vs published | ESS/N range (mine) | ESS/N range (published) |
|---|---:|:---:|:---:|:---:|
| gg → t t̄ | 2 | 0.89σ | 0.812 – 0.980 | 0.981 – 0.998 |
| gg → t t̄ H | 5 | 1.54σ | 0.797 – 0.917 | 0.959 – 0.987 |
| u ū → W⁺W⁻ → 4ℓ | 8 | 0.42σ | 0.806 – 0.865 | 0.927 – 0.941 |

There is no monotone "harder in higher dimension" story here, and it would be
wrong to invent one: the 8-D process agrees best and the 5-D process worst.
The three processes differ in more than dimension — the 5-D case has a third
massive particle and a phase space with genuinely more shape for the copula to
learn, while the 8-D case is dominated by two narrow resonances that the
marginals capture well. The repository's own documentation warns that this
sweep changes dimension, resonance structure and matrix-element complexity
simultaneously, so it is not a controlled dimension-only study; these results
are consistent with that caveat rather than in tension with it.

---

## 6. GPU runs

See `RUNBOOK_GPU.md` for the setup and commands.

**Hardware.** NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB (~4.3 GiB usable
with a desktop session), driver 546.92. The conda `workspace` environment
shipped `torch 2.3.0+cpu` — the CPU-only build — so `torch.cuda.is_available()`
was `False` and every prior run had silently fallen back to CPU. It was
replaced with `torch 2.3.0+cu121`; the version is held at 2.3.0 so the
`numpy<2` pin stays valid. The 12-test suite passes after the swap.

### 6.1 `gg → t t̄`, smoke — CPU vs GPU

Identical config, identical budget, only `--device` differs:

| | CPU | GPU |
|---|---:|---:|
| training wall clock | **303 s** | **304 s** |
| validation wall clock | — | 208 s |
| learned graph | 1 edge | 1 edge |
| final ESS | 0.8054 | 0.8074 |
| final Î | 2.632163e-08 | 2.647052e-08 |
| log w std | 0.4975 | 0.5145 |

**The GPU changes the runtime by 0.3 %.** The two models are statistically
indistinguishable — the same one-edge graph, ESS within 0.25 %, Î within 0.6 %;
the residual scatter is a different training trajectory, since the RNG stream
differs by device.

The Sobol column of the GPU validation is **bit-identical** to
`results_public/gg_ttbar_validation` at all five energies (relative difference
`0.00e+00`), so the build-level check is unaffected by the device.

GPU held-out NF results:

| E [GeV] | NF [pb] | ESS/N | pull vs Sobol |
|---:|---:|---:|---:|
| 400 | 12.835344 ± 0.0067 | 0.879 | −1.43 |
| 650 | 16.916319 ± 0.0027 | 0.987 | −0.09 |
| 1000 | 11.37313 ± 0.0042 | 0.935 | +1.18 |
| 1500 | 6.861171 ± 0.0026 | 0.933 | +0.31 |
| 1950 | 4.7780877 ± 0.0014 | 0.960 | +1.50 |

### 6.2 Why the GPU buys nothing here

Two compounding causes, both verified in the code rather than inferred.

**(a) The matrix element is the ceiling, and it runs on the CPU.** It is
evaluated one event per Python call:

```python
for index, event in enumerate(momenta):
    values[index] = float(self.evaluate(p, self.alpha_s, -1))
```

The published `gg_ttbar` validation ran 500 000 evaluations in 5.34 s — about
**93 600/second** — *with the flow already on CUDA*. The matrix-element
throughput measured on this box, on **one CPU core**, is **96 600/second**.
Those numbers match, which says the matrix element dominates the wall clock
whether or not the flow is on a GPU.

**(b) The driver round-trips host↔device every chunk.** From
`run_madgraph_process_conditional_copula.py`:

```python
unit_numpy = unit.detach().cpu().double().numpy()          # GPU → host
for start in range(0, len(unit_numpy), me_chunk):
    pieces.append(evaluator.log_integrand(...))             # CPU matrix element
return torch.as_tensor(np.concatenate(pieces), device=unit.device, ...)  # host → GPU
```

Each chunk of `--me-chunk` (default 512) samples on the GPU, copies to the
host, evaluates the matrix element on one CPU core, and copies back. The GPU's
gain on the flow is spent crossing the bus, and then the CPU does the part that
actually costs time.

### 6.3 Consequence for the Alpaka port

**Porting the flow to Alpaka cannot deliver a ≥10× throughput gate on this
architecture** — not because Alpaka is the wrong tool, but because the flow is
not where the time is. Three routes actually move the number:

1. remove the round trip (co-resident sampling and evaluation) — small change,
   worth doing regardless;
2. move the matrix element to the GPU — substantially harder than the flow, and
   not currently in the proposal's scope. This is where the real 10× lives;
3. benchmark at a multiplicity where the flow dominates.

Because the gate is defined on total event-generation rate, and the matrix
element is CPU-bound, no amount of Alpaka work on the flow alone will pass it.

### 6.4 Thorough on GPU

All three processes at `--speed-profile thorough --device cuda`, run
sequentially, writing to `results/<proc>_copula_thorough_gpu/` and
`results/<proc>_validation_thorough_gpu/`.

#### `gg → t t̄` — the published thorough model, reproduced

Wall clock **5 h 05 min** (10:36:11 → 15:41:16).

| quantity | ours (thorough, GPU) | published (`gg_ttbar_thorough_unbuffered`) | rel. diff |
|---|---:|---:|---:|
| learned graph | 1 edge | 1 edge | — |
| ESS | 0.8407084868615254 | 0.8406046449897572 | **1.2e-04** |
| Î | 2.635615770700724e-08 | 2.6354984203294262e-08 | **4.5e-05** |
| log w std | 0.4529427119180449 | 0.4525016120941321 | 9.7e-04 |

ESS to **0.012 %**, Î to **0.0045 %**. This is the strongest model-level
agreement in this document: an independent build, a different Fortran
toolchain, a different operating system, and the full `thorough` training
budget landing on the authors' own published numbers.

Note this is a much tighter agreement than the smoke model achieved (§4) —
0.8054 vs 0.8406 there, against 1.2e-04 relative here. Training budget, not the
build, was the limiting factor, exactly as suspected.

#### How the ESS evolved during training

| stage | ESS |
|---|---:|
| adaptive round 0 | 0.840517 |
| adaptive round 1 | 0.841694 |
| adaptive round 2 | 0.839931 |
| χ² fine-tuning | 0.840877 |
| **final** | **0.840708** |

Two observations worth passing to Brown:

* **The third adaptive round was counterproductive** — round 2 finished *below*
  round 1 (0.839931 vs 0.841694), and the χ² stage only partly recovered it.
  On this process the final `adaptive_rounds = 3` setting does not pay for
  itself.
* **The χ² stage is frozen.** Its loss, local ESS and global ESS were
  *bit-identical* across 4 200 consecutive steps (loss −0.560217, ESS 0.840877)
  — roughly 23 minutes of wall clock producing byte-identical output. The
  profile's `chi2_max_steps = 6000` is far more than this process uses.

Both are budget-provisioning issues, not code faults, but together they account
for a meaningful share of the 5-hour runtime.

#### Note on file placement

The sweep script passed `--output` to the validation command but not to the
training one, so the driver fell back to its relative-path default and wrote to
`generated/gg_ttbar_standalone/results/gg_ttbar_conditional_copula/`. The files
were moved to `results/gg_ttbar_copula_thorough_gpu/` afterwards; the checkpoint
and summary are unaffected. The script is corrected for future runs.

#### `gg → t t̄ H` and `u ū → W⁺W⁻ → 4ℓ` — not completed

The sweep was stopped at **18:53 on 2026-10-16** so the machine could be shut
down.

| process | state when stopped |
|---|---|
| `gg → t t̄ H` | KL round 0 of 3, step ~3 000, val NLL −0.734990. **Partial and discarded** — no checkpoint is written until a run finishes. |
| `u ū → W⁺W⁻ → 4ℓ` | never started |

`gg_ttbar`'s result above is complete and unaffected; it was written to disk at
15:41.

Runtime observed: `gg_ttbar` ran at 0.5–1.0 s/step and took 5 h 05 min;
`gg_tth`'s 5-D matrix element runs at 1.5–2.0 s/step, so it was on course for
roughly 10–12 h, and the 8-D process would be slower again.

**There are no intermediate checkpoints**, so an interrupted thorough run
restarts from zero. Any future attempt needs an uninterrupted block for the
whole process.

To resume, run `mg5_scratch/run_thorough_gpu_resume.sh`, which skips the
completed `gg_ttbar`, trains and validates the remaining two, and fixes the
`--output` bug described above.

---

## 7. How to reproduce

See `RUNBOOK.md` for the full procedure, or run:

```bash
bash scripts/run_all_windows.sh all          # build + integrand + train + validate
python scripts/compare_validation.py \
    --mine results/gg_ttbar_validation_smoke \
    --theirs results_public/gg_ttbar_validation
```
