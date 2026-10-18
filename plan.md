# plan.md — MANGO Phase 1, Oregon workstream

**Scope of this plan:** Oregon's side of MANGO Phase-I (DE-FOA-0003612, Focus
Area 14C). Oregon (Majewski, Le Goff) owns the **GPU framework**. BNL owns the
MadGraph data and the validation-metric contract; Brown owns the CNF /
flow-matching architecture. The Months 1–3 charter on this side is:

1. learn the `conditional-nf-integration` pipeline,
2. establish a baseline GPU validation,
3. start on Alpaka.

Items 1 and 2 are **done** — the work below records what they produced, because
the second one changed what Alpaka should be pointed at.

**Reference:** `MANGO_2026_narrative_27Apr2026.pdf` — §4 (milestones and sprint
split), §5 (data sources and models), §6 (decision gate metrics).

---

## 1. Where we start from

The `conditional-nf-integration` repository is a working conditional
normalizing flow pipeline for MadGraph matrix elements. It has been rebuilt and
run end to end on a local Windows machine with an RTX 3060 Laptop GPU, and all
three of its LO processes are now trained **and** validated at the full
`thorough` budget on the GPU.

| asset | state |
|---|---|
| MadGraph 3.6.7 standalone build chain | working — generate → compile → validate, natively on Windows (MSYS2/MinGW f2py) |
| LO processes, thorough-trained on GPU | `gg→t t̄` (2D, 5 h 05 min), `gg→t t̄H` (5D, 12 h 26 min), `u ū→W⁺W⁻→4ℓ` (8D, 13 h 15 min) |
| held-out validation, all three | Sobol column **bit-identical** to `results_public/`; NF column inside the quoted uncertainties |
| exact-likelihood importance-sampling estimator | working, with ESS/N diagnostics |
| conditional flow (marginals × copula, conditioned on √ŝ) | working, one model per process across the whole energy range |
| GPU execution | working end to end (`--device cuda`), and **measured to be no faster** — §2 |

Final ESS: 0.840708 (2D), 0.918134 (5D), 0.508716 (8D). The 8D model's held-out
validation reaches ESS 0.9286–0.9434 across 300–1950 GeV, and agrees with the
published NF column to within 0.011 % at every energy.

### What each deliverable rests on

The repo gives Oregon the evaluation harness and the validation protocol. It
does **not** give a continuous flow (discrete cubic-spline marginals plus an
autoregressive copula, not a neural ODE) and — more relevant here — it does not
give any **GPU-native** code. Everything that runs on the GPU today is PyTorch
eager mode.

---

## 2. The finding that shapes the Alpaka plan

**This is the most important result of the baseline GPU work, and it should
drive Phase 1 scoping.** It is a measurement, not a prediction.

### The GPU buys nothing on this workload

`gg→t t̄`, identical config and budget, flow on CPU versus on the RTX 3060:

| | CPU | GPU |
|---|---:|---:|
| training wall clock | **303 s** | **304 s** |
| final ESS | 0.8054 | 0.8074 |
| final Î | 2.632163e-08 | 2.647052e-08 |

0.3 % — inside the run-to-run scatter. The two models are statistically
indistinguishable.

### Why

**(a) The matrix element is the ceiling, and it is on the CPU.** It is evaluated
one event per Python call:

```python
for index, event in enumerate(momenta):
    values[index] = float(self.evaluate(p, self.alpha_s, -1))
```

The published `gg_ttbar` validation ran 500 000 evaluations in 5.34 s
(~93 600/s) *with the flow already on CUDA*. Measured here on **one CPU core**:
**96 600/s**. Those match, which says the matrix element sets the wall clock
whether or not the flow is on a GPU.

**(b) The driver forces a host↔device round trip every chunk.** Each
`--me-chunk` (default 512) of samples is generated on the GPU, copied to the
host, evaluated on one CPU core, and copied back. Whatever the GPU wins on the
flow is spent crossing the bus.

Supporting evidence: the same 500 000 evaluations took 26.4–27.1 s here against
11.2 s in `results_public`. That factor of 2.4 is single-thread CPU throughput,
not GPU — this box's core is slower than the machine the published timings came
from.

### What this means for Alpaka

**Porting the flow to Alpaka cannot, by itself, deliver the ≥10× throughput
gate.** Not because Alpaka is the wrong tool, but because the flow is not where
the time is. Three routes actually move the number:

1. **Remove the round trip** — co-resident sampling and evaluation, nothing
   crossing the bus per chunk. Small change, worth doing regardless.
2. **Move the matrix element to the GPU** — substantially harder than the flow
   and not currently in the proposal's scope. This is where the real 10× lives.
3. **Choose a workload where the flow dominates** — high multiplicity, where
   the matrix element is relatively cheaper and the phase space harder.

Route 1 is a small change. Route 2 is a decision, because if the gate is defined
on total event-generation rate and the matrix element is CPU-bound, no amount of
Alpaka work on the flow will pass it.

**A cheaper CPU-side win exists, and it has now been measured:** the per-event
matrix-element calls are independent, so a multiprocessing pool around
`log_integrand` scales with cores. No Alpaka, no GPU. On this machine (8
physical cores) a 16-thread pool gives **6× — 614 000 evals/s against 103 000 on
one core** (§3.4). That, not the serial rate, is the reference the Alpaka work
has to beat.

---

## 3. Phase 1 — Months 1–3 deliverables

**Objective in one line:** by the end of Month 3, Alpaka compiles and runs a
real kernel from this pipeline on an Oregon GPU, the Python↔C++ boundary is
fixed and documented, and the throughput baseline that any Alpaka result must be
judged against has been measured on the actual bottleneck.

### 3.1 Deliverable A — baseline GPU validation (done)

All three processes trained and held-out-validated at `--speed-profile thorough
--device cuda`. Reproduced on `gg_tth`:

| E (GeV) | NF (pb) | Sobol (pb) | pull | ESS |
|---:|---|---|---:|---:|
| 500 | 0.0026853146 | 0.0026854701 | −0.32 | 0.9836 |
| 700 | 0.032118678 | 0.032121817 | −0.53 | 0.9833 |
| 1000 | 0.041191663 | 0.041200623 | −1.01 | 0.9777 |
| 1500 | 0.04242704 | 0.042425358 | +0.14 | 0.9703 |
| 1950 | 0.041285493 | 0.041272338 | +0.96 | 0.9615 |

The Sobol column is bit-identical to `results_public/gg_tth_5d_validation` at
all five energies, which is the build-level check: it is hardware-independent,
so it holds across institutions. The NF column is not — different RNG streams
give different training trajectories — and the tables do not currently
distinguish the two.

*Acceptance:* met. Three processes, trained and validated, reproducible.

### 3.2 Deliverable B — Alpaka hello-world on a real backend

The goal is not speed; it is proving the toolchain end to end on hardware
Oregon controls.

* Alpaka installed and building against the CUDA backend (and, if Talapas is
  used, its module system) on at least one GPU node;
* a kernel from this pipeline reimplemented in Alpaka and checked
  bit-for-bit (or to stated tolerance) against the PyTorch reference on the
  same inputs;
* the first candidate is the **monotone cubic-Hermite marginal evaluation** —
  it is the innermost per-sample loop, pure arithmetic, no data-dependent
  control flow, and therefore the most tractable first port.

*Acceptance:* one Alpaka kernel, one backend, matched against PyTorch, with the
discrepancy quantified.

### 3.3 Deliverable C — the Python↔C++ boundary

The boundary is a design decision, not a detail, and it is Oregon's to make
because it determines what Alpaka can own. It must be frozen in Phase 1 so BNL's
data format and Brown's model interface do not move underneath it.

Decisions to record:

* where the flow lives (Python/PyTorch versus Alpaka) at each stage of the
  pipeline: training, sampling, density evaluation;
* what crosses per batch — samples, log-density, weights — and in what layout;
* whether the matrix element stays CPU-bound behind the boundary (the status
  quo), moves to the device (the 10× route), or is co-scheduled so nothing
  crosses per chunk (Route 1 in §2);
* the binding mechanism (pybind11 or equivalent) and the error/threading model
  across it.

*Acceptance:* a written interface note plus a compiling stub both sides agree
on.

### 3.4 Deliverable D — the throughput baseline, measured on the bottleneck

The proposal's gate is ≥10× against a **32-core** CPU reference. Before any
Alpaka number is quoted, establish what that reference actually is and what the
per-core rate is on this workload:

| measurement | status |
|---|---|
| matrix element, 1 core | **done** — 103 000 evals/s (`gg_ttbar`, 2D, 400 GeV) |
| matrix element, N cores (multiprocessing pool) | **done** — see below |
| flow density evaluation, GPU eager | partially measured via the 0.3 % result |
| flow density evaluation, Alpaka | Phase 2 |

Measured with `scripts/benchmark_matrix_element_parallel.py`, 2 000 000 points,
AMD Ryzen 7 6800H (8 physical cores / 16 threads, 45 W laptop part):

| workers | evals/s | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 103 033 | 1.00 | 100 % |
| 2 | 169 218 | 1.64 | 82 % |
| 4 | 297 756 | 2.89 | 72 % |
| 8 | 475 464 | 4.61 | 58 % |
| 16 | **614 298** | **5.96** | 37 % |

Two things follow. First, **the CPU baseline is not 1 core** — a 16-thread pool
gets ~6× with no GPU and no Alpaka, so any Alpaka speedup must be measured
against the parallel number, not the serial one. Second, this is a laptop: 8
physical cores sharing one thermal budget with the GPU. A 32-core server part
will scale better and at higher per-core clock, so the real reference for the
gate is faster than 614 000 evals/s. The ≥10× target therefore sits well above
what this measurement reaches, which is exactly why §2's route 2 (moving the
matrix element to the device) is the only path that changes the answer.

*Acceptance:* met for the CPU side. The flow-side numbers (eager and Alpaka) are
Phase 2.

---

## 4. Next week

Ordered by what unblocks what.

**1. Measure the multi-core matrix-element rate — done.**

Measured (details in §3.4): a 16-thread multiprocessing pool reaches
**614 000 evals/s against 103 000 on one core, a 6× speedup**, on a laptop with
8 physical cores. Two consequences:

- **The serial number is not the baseline.** Any Alpaka result has to be quoted
  against the parallel CPU number, or it will look better than it is.
- **A 32-core server reference will be faster again.** The gate is 10× versus
  that reference, so the bar is higher than 614 000 evals/s, not lower.

Still open: the same measurement on a process at the high-multiplicity end of
the ladder, where the flow carries more of the work — that is where the gate
could actually be won.

**2. Decide the repository home for MANGO work.** `conditional-nf-integration`
is a third party's published repository, currently carrying our commits on a
branch of a clone. MANGO needs its own repository before Sprint 1 delivers
anything, with this pipeline pulled in as a dependency or a vendored module.
Decide this before more Alpaka code lands, not after.

**3. Get Alpaka building on Oregon hardware.** Talapas (120 GPUs per the
proposal) is the natural target. The local RTX 3060 is enough to prove the
toolchain — Alpaka supports CUDA on consumer cards — but the module system and
backend selection differ, so do one, then the other.

**4. Agree with Brown what the flow interface must expose.** Alpaka needs a
fixed contract: sample, log-density, and the conditioning input. Get that
agreed before writing kernels, because a kernel written against the wrong
interface is thrown away.

**5. Write up the GPU non-speedup for the collaboration.** The measurement in
§2 reframes the throughput work and is not yet in anyone's slide deck. One page,
the two numbers, the two causes, the three routes — it should reach BNL and
Brown before Sprint 2 planning, not after.

**6. Fix the "no intermediate checkpoint" problem before the next long run.**
The 8D thorough run was lost twice — 49 min, then 9 h 20 m — because
`run_adaptive_integrand_structure_search.py` calls `torch.save` once, at the
very end, and the checkpoint carries no optimizer or RNG state, so it cannot be
resumed even from a saved file. Any multi-hour run on shared hardware will hit
this. Periodic checkpointing is a small change and pays for itself the first
time a job is preempted.

---

## 5. Risks worth naming now

**The throughput gate may be defined on a quantity Alpaka cannot improve.**
§2 is the evidence. If the gate is total event-generation rate and the matrix
element stays on the CPU, the flow port cannot pass it — not because the port is
poor, but because the ported component is not the bottleneck. **This is the item
most worth resolving before Phase 2 scoping**, and it is a collaboration
decision, not an Oregon one.

**The benchmark dimension determines the answer.** Methods of this class do not
win in low dimension: on a 2D synthetic target, VEGAS beats the flow on both
accuracy and wall-clock, and the repository's headline "VEGAS fails here" result
is **32D**. (That 2D comparison was a smoke-budget run, so treat the specific
figures as illustrative — but the direction is not an artefact of the budget.)
Benchmark on the high-multiplicity end and state the dimension beside every
number.

**Coverage may invalidate the accuracy gate.** The repository's 32D benchmark
reports 95 % coverage of 0.7875; an 8D run showed the same signature — point
estimates agreeing to 0.42σ while ESS/N sat at 0.81–0.86. Accurate estimates
with poor ESS is the signature of heavy importance-weight tails: unbiased, but
with error bars too small. If that holds at Phase 1 scale, a comparison "within
theoretical uncertainty" can pass or fail for the wrong reason. That is BNL's
metric to own, but it is the assumption under every number Alpaka will report.

**Porting PyTorch eager to Alpaka is a rewrite, not a port.** The models are
Python/PyTorch; Alpaka is C++17. Nothing moves across automatically. Budget
accordingly, and pick the first kernel for tractability (§3.2) rather than for
speed.

**Training cost is excluded from the throughput claim, and should stay
explicit.** The gate compares generation rate against a CPU reference. If the
flow has to be retrained per process *and* per energy point, the amortised
argument weakens. The conditional-√ŝ design answers this — one model per process
across the whole energy range — and it is worth putting in front of the
collaboration, because it is not currently in the narrative.

**Budget over-provisioning is costing hours per run.** Across all three
processes, the KL cap of 30 000 steps is never approached — every round stops at
7 200 by patience — and the χ² stage never pays for itself: on the 8D process
`CHI2 SUMMARY` reports `best step = 0`, i.e. it ran its full 6 000 steps and
then restored the weights it started with. Any wall-clock number quoted to the
collaboration should note that a meaningful share of the thorough runtime is
spent on stages that do not improve the result.

---

## 6. Decisions needed from the collaboration

1. **Is the throughput gate defined on total event-generation rate, or on the
   flow alone?** This determines whether Portland's Alpaka work can pass it.
2. Which repository hosts MANGO code, and the licence/attribution arrangement
   with the upstream `conditional-nf-integration` author.
3. The exact 32-core CPU reference configuration for the throughput gate — what
   it runs, at what dimension.
4. GPU allocation and who administers it (Talapas).
5. The operational definition of the accuracy gate (which uncertainty band),
   which BNL owns and every performance number inherits.
6. Whether the conditional-√ŝ reuse goes into the Phase 2 narrative.
