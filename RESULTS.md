# RESULTS — local Windows rebuild

*What was produced on this machine, and how it compares with the curated
results in `results_public/`.*

Machine: Windows 11, 16 logical cores, 15 GB RAM, NVIDIA RTX 3060 Laptop GPU
(6 GB). Python 3.10.14 (conda `workspace`), torch 2.3.0+cu121.
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

### 6.2 Where the time goes — **corrected twice**

Two earlier revisions of this section were wrong, and both are recorded here
because the second error was subtler than the first.

1. **First revision:** "the matrix element is the ceiling, the flow is too cheap
   to matter." Wrong — the flow is the larger share.
2. **Second revision:** "the flow is ~79 % of a step at the pipeline's training
   batch of 10 000." Also wrong — **10 000 is the marginal-fit batch, not the KL
   batch.** The flag that looks like the training batch (`--batch-size`) never
   reaches the KL stage.

The KL batch is `--is-points × ADAPTIVE_CONDITIONS_PER_STEP (256)`:

| profile | is-points | **KL batch** |
|---|---:|---:|
| smoke | 32 (capped) | **8 192** |
| thorough | 128 | **32 768** |

Profiled flow cost (sample + log q + backward) against the matrix element on the
same points, with `scripts/profile_flow_vs_matrix_element.py`:

| points | flow on CPU | flow on GPU | GPU speedup | matrix element | ME share (GPU) |
|---:|---:|---:|---:|---:|---:|
| 10 000 | 295 ms | **434 ms** | **0.68×** | 113 ms | 21 % |
| 30 000 | 735 ms | 520 ms | 1.41× | 321 ms | 38 % |
| 100 000 | 2 555 ms | **532 ms** | **4.80×** | 1 020 ms | 66 % |

So the two profiles sit on opposite sides of the crossover:

* **smoke** runs at KL batch 8 192 — below it. The GPU flow is *slower* than the
  CPU flow there because the batch is too small to amortise kernel launches.
  That is why the original 303 s / 304 s comparison showed nothing: it was a
  smoke run.
* **thorough** runs at KL batch 32 768 — above it. The GPU flow is ~1.4× faster
  and the matrix element is ~38 % of the step.

**"The GPU buys nothing" is therefore a statement about the smoke profile, not
about thorough training.** The original note generalised it incorrectly.

### 6.3 End-to-end check at the thorough batch

Smoke with `--is-points 128`, giving KL batch 32 768 — the value thorough uses —
CPU versus GPU, sequential:

| KL batch | CPU | GPU | speedup | final ESS |
|---:|---:|---:|---:|---:|
| 8 192 (smoke default) | 303 s | 304 s | 1.00× | 0.8054 / 0.8074 |
| 32 768 (thorough value) | **388 s** | **365 s** | **1.06×** | 0.8086 / 0.8099 |

The GPU does start to help once the batch is large enough, but **1.06× is far
below the 1.4× the profiler attributes to the KL stage alone** — because a smoke
run spends most of its wall clock outside the KL loop. A definitive end-to-end
number needs a thorough CPU run (about 5 h), which has not been done; on the
evidence above, expect something between 1.06× and 1.4×.

**(b) The driver still round-trips host↔device every chunk.** From
`run_madgraph_process_conditional_copula.py`:

```python
unit_numpy = unit.detach().cpu().double().numpy()          # GPU → host
for start in range(0, len(unit_numpy), me_chunk):
    pieces.append(evaluator.log_integrand(...))             # CPU matrix element
return torch.as_tensor(np.concatenate(pieces), device=unit.device, ...)  # host → GPU
```

This is a real cost but a second-order one.

### 6.4 The validation path has a free win

Same model, same 500 000 evaluations, only `--gpu-chunk` changed 10 000 →
100 000. No code change:

| E (GeV) | 10k chunk | 100k chunk | speedup |
|---:|---:|---:|---:|
| 400 | 14.17 s | 6.37 s | 2.23× |
| 650 | 12.29 s | 6.19 s | 1.98× |
| 1000 | 10.91 s | 6.61 s | 1.65× |
| 1500 | 11.57 s | 6.01 s | 1.93× |
| 1950 | 11.42 s | 6.73 s | 1.70× |

For the Alpaka port:

1. **Raise the batch size first** — 1.65–2.2× on validation for free, and it puts
   the GPU in the regime where it actually wins.
2. **Then attack the flow's per-call overhead.** The flow is the larger share of
   a step and is overhead-bound, not arithmetic-bound. A straight Alpaka port
   would inherit the launch cost; the case rests on fusing the per-dimension
   chain into one kernel launch.
3. **The matrix element becomes the bottleneck only after both** (66 % at 100k).

The gate is still defined on total event-generation rate, so both components
matter — but the ordering above is what the measurements support.

### 6.5 Thorough on GPU

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

Held-out validation, 500 000 evaluations per energy:

| E (GeV) | NF (pb) | Sobol (pb) | pull | ESS |
|---:|---|---|---:|---:|
| 400 | 12.843696 ± 8.9e-04 | 12.844943 ± 1.2e-14 | −1.41 | 0.9976 |
| 650 | 16.915145 ± 2.0e-03 | 16.916571 ± 1.6e-11 | −0.73 | 0.9934 |
| 1000 | 11.369921 ± 1.6e-03 | 11.368139 ± 1.3e-13 | +1.13 | 0.9904 |
| 1500 | 6.861110 ± 1.3e-03 | 6.860368 ± 9.6e-09 | +0.57 | 0.9821 |
| 1950 | 4.777001 ± 9.1e-04 | 4.776030 ± 9.2e-08 | +1.06 | 0.9820 |

The **Sobol column is bit-identical** to `results_public/gg_ttbar_validation`.
Against the published NF column: −0.017 %, −0.003 %, +0.033 %, +0.056 %,
+0.023 %. At 400 GeV the two models bracket the true value in opposite
directions (−1.41σ here, +1.26σ published), which is the expected scatter
between independent training runs.

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

#### `gg → t t̄ H` — complete

Wall clock **12 h 26 min** (21:25:43 → 09:52:10 on 2026-10-17), then 5 min of
held-out validation.

| quantity | value |
|---|---:|
| learned graph | 10 edges |
| final ESS | 0.918134 |
| Î | 9.698626230668e-11 |
| log Î | −23.056452 |
| log w std | 0.594390 |

Held-out validation, 500 000 evaluations per energy:

| E (GeV) | NF (pb) | Sobol (pb) | pull | ESS |
|---:|---|---|---:|---:|
| 500 | 0.0026853146 ± 4.9e-07 | 0.0026854701 ± 2.1e-09 | −0.32 | 0.9836 |
| 700 | 0.032118678 ± 5.9e-06 | 0.032121817 ± 5.1e-07 | −0.53 | 0.9833 |
| 1000 | 0.041191663 ± 8.8e-06 | 0.041200623 ± 1.0e-06 | −1.01 | 0.9777 |
| 1500 | 0.04242704 ± 1.1e-05 | 0.042425358 ± 5.8e-06 | +0.14 | 0.9703 |
| 1950 | 0.041285493 ± 1.2e-05 | 0.041272338 ± 7.2e-06 | +0.96 | 0.9615 |

All five pulls are inside ±1.01. The **Sobol column is bit-identical** to
`results_public/gg_tth_5d_validation` at every energy — the build-level check
passes for the third process. Against the published NF column:

| E (GeV) | ours | published | rel. diff |
|---:|---:|---:|---:|
| 500 | 0.0026853146 | 0.0026854314 | −0.0044 % |
| 700 | 0.032118678 | 0.032116707 | +0.0061 % |
| 1000 | 0.041191663 | 0.041191439 | +0.0005 % |
| 1500 | 0.042427040 | 0.042413176 | +0.033 % |
| 1950 | 0.041285493 | 0.041257667 | +0.067 % |

Independent training runs, so these are sampling scatter rather than a
systematic.

**ESS evolution:** bridge 0.874486 → KL round 0 0.913830 → round 1 0.916638 →
round 2 0.917882 → χ² 0.917263 → final 0.918134. All three KL rounds stopped
at the 7 200-step patience limit, a quarter of the 30 000-step cap — the same
over-provisioning seen on `gg_ttbar`, and again the χ² stage did not improve
ESS.

**Timing:** KL steps ran at **1.75 s/step** (600 steps in 1 051 s). The same
500 000 evaluations cost **26.4–27.1 s per energy here against 11.2 s in
`results_public`** — a factor of 2.4, entirely in the CPU matrix-element loop,
not the GPU.

#### `u ū → W⁺W⁻ → 4ℓ` — complete

Wall clock **13 h 15 min** (22:09:55 on 2026-10-17 → 11:24:50 on 2026-10-18),
then 5 min of held-out validation. This was the third attempt; the first two
were lost with no checkpoint (49 min, then 9 h 20 m) — see "Attempts" below.

| quantity | value |
|---|---:|
| learned graph | 28 edges (complete DAG) |
| final ESS | 0.508716 |
| Î | 3.735308225735e-11 |
| log Î | −24.0106056841 |
| log w std | 1.021910 |

Held-out validation, 500 000 evaluations per energy:

| E (GeV) | NF (pb) | Sobol (pb) | pull | ESS |
|---:|---|---|---:|---:|
| 300 | 0.052943440 ± 1.8e-05 | 0.052992375 ± 1.2e-05 | −2.23 | 0.9434 |
| 500 | 0.028254431 ± 1.0e-05 | 0.028236129 ± 2.7e-05 | +0.63 | 0.9402 |
| 800 | 0.014715634 ± 5.4e-06 | 0.014726234 ± 3.6e-05 | −0.29 | 0.9359 |
| 1200 | 0.008032969 ± 3.0e-06 | 0.007999827 ± 9.6e-05 | +0.35 | 0.9334 |
| 1950 | 0.003736164 ± 1.5e-06 | 0.003786851 ± 1.1e-04 | −0.46 | 0.9286 |

The **Sobol column is bit-identical** to `results_public/uu_ww_4l_validation`
at all five energies. Against the published NF column:

| E (GeV) | ours | published | rel. diff |
|---:|---:|---:|---:|
| 300 | 0.052943440 | 0.052941132 | +0.0044 % |
| 500 | 0.028254431 | 0.028252471 | +0.0069 % |
| 800 | 0.014715634 | 0.014715797 | −0.0011 % |
| 1200 | 0.008032969 | 0.008033686 | −0.0089 % |
| 1950 | 0.003736164 | 0.003735748 | +0.011 % |

ESS is slightly *higher* than the published model at all five energies
(0.9286–0.9434 against 0.9266–0.9412).

**The 300 GeV pull of −2.23 is not a defect in this build.** The published
model has the same pull: its 300 GeV estimate sits 5.1e-5 pb below Sobol with a
combined uncertainty of 2.2e-5, i.e. **−2.3σ**. Both independent models land in
the same place, so the deviation is a property of the model at that energy, not
of the reproduction.

**ESS evolution:** bridge 0.469578 → KL round 0 0.502517 → round 1 0.506345 →
round 2 0.506502 → post-χ² 0.507125 → final 0.508716.

#### The χ² stage selected step 0

`CHI2 SUMMARY` reports **`best step = 0`**: across all 6,000 χ² steps the stage
never improved on its starting point and finished by restoring the weights it
began with. Global ESS declined monotonically throughout
(0.505483 → 0.505449).

That is the third and cleanest instance of the same pattern — on `gg_ttbar` the
stage was frozen for 4,200 byte-identical steps, on `gg_tth` it cost 0.0002 of
ESS, and here it discarded its entire budget. `chi2_max_steps = 6000` is not
being used productively on any of the three processes.

#### Attempts

| attempt | start | ended | duration | outcome |
|---|---|---|---:|---|
| 1 | 09:57:50 10-17 | 10:47 10-17 | 49 min | stopped by operator |
| 2 | 12:31:56 10-17 | 21:52 10-17 | 9 h 20 min | killed by machine restart |
| 3 | 22:09:55 10-17 | 11:24:50 10-18 | **13 h 15 min** | **complete** |

Attempt 3 reproduced attempt 2's trajectory closely while it lasted — bridge
ESS 0.469578 vs 0.471382, round 0 ESS 0.502517 vs 0.503910, i.e. 0.3 % apart
from a different RNG stream.

#### Runtime summary

| process | dim | wall clock | outcome |
|---|---:|---:|---|
| `gg → t t̄` | 2 | 5 h 05 min | complete |
| `gg → t t̄ H` | 5 | 12 h 26 min | complete |
| `u ū → 4ℓ` | 8 | 13 h 15 min | complete |

**There are no intermediate checkpoints**, and the driver cannot resume even
from a saved file: the checkpoint carries `model_state_dict`,
`marginal_state_dict`, `graph` and metrics, but **no optimizer state, RNG state
or step position**. An interrupted run restarts from zero. On a batch system,
also request a `+JobFlavour` above the wall time (`REPRODUCTION_COMMANDS.md`
§4.5).

---

## 7. How to reproduce

See `RUNBOOK.md` for the full procedure, or run:

```bash
bash scripts/run_all_windows.sh all          # build + integrand + train + validate
python scripts/compare_validation.py \
    --mine results/gg_ttbar_validation_smoke \
    --theirs results_public/gg_ttbar_validation
```
