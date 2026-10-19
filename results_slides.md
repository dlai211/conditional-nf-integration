---
marp: true
title: MANGO — Oregon Phase 1 status
---

# MANGO — Oregon workstream
## Conditional normalizing flows for MadGraph integration

**Phase 1, Months 1–3: status and findings**

<!-- notes:
Oregon's charter was three things: learn the pipeline, establish a GPU
baseline, start on Alpaka. Two are done. The second one produced a finding
that changes what the third should be — that's the spine of this deck.
-->

---

# What we were asked to do

| # | Charter item | Status |
|---|---|---|
| 1 | Learn `conditional-nf-integration` | **done** |
| 2 | Baseline GPU validation | **done** — and it changed the plan |
| 3 | Start on Alpaka | not started |

Plus: the pipeline previously could not be run anywhere without CPU limits or a
container. It now builds and runs end to end on a workstation.

<!-- notes:
Be direct that item 3 hasn't started. The honest reason is that item 2
invalidated the assumption item 3 was going to be built on.
-->

---

# The pipeline in one slide

```
configs/processes/<proc>.json
   │
   ├─ 1. generate + compile     MadGraph standalone → matrix2py extension
   ├─ 2. integrand check        4-momentum, mass shells, Jacobian > 0
   ├─ 3. train                  conditional flow as an importance-sampling proposal
   └─ 4. validate               NF estimate vs independent Sobol QMC
```

The flow proposes; the estimator stays exact:

```
Î = (1/N) Σ f(xₙ|c) / r_θ(xₙ|c)          ESS/N = (Σw)² / (N Σw²)
```

A bad proposal costs variance, never correctness. That is the safety property.

<!-- notes:
The key conceptual point: this is importance sampling with a learned proposal.
The integral is unbiased for any positive r. Training only buys ESS.
The conditioning variable c is sqrt(s-hat), so one model covers the whole
energy range instead of one per point.
-->

---

# Reproducing the published results

All three processes at the full `thorough` budget, on the GPU:

| process | dim | wall clock | final ESS | Î |
|---|---:|---:|---:|---:|
| `gg → t t̄` | 2 | 5 h 05 min | 0.840708 | 2.6356e-08 |
| `gg → t t̄ H` | 5 | 12 h 26 min | 0.918134 | 9.6986e-11 |
| `u ū → 4ℓ` | 8 | 13 h 15 min | 0.508716 | 3.7353e-11 |

Independent build, different Fortran toolchain, different operating system.

<!-- notes:
These are our numbers, produced from scratch on Windows. ESS for the 8D model
looks low but that is expected — the published summary is 0.5078, and the
held-out validation is much higher.
-->

---

# The build check: Sobol is bit-identical

Two columns in every validation table, and they mean different things:

| column | hardware-dependent | what it proves |
|---|---|---|
| **Sobol QMC** | **no** | the build — phase space, Jacobian, matrix element |
| **NF estimate** | yes (RNG differs) | the model — how well training converged |

**All three processes: the Sobol column is bit-identical to `results_public`.**

NF column agrees to 0.004 – 0.067 % (independent training runs → sampling scatter).

<!-- notes:
This is the strongest statement in the deck. It means our independent build
reproduces the authors' physics exactly. The NF column can never be
bit-identical because the RNG stream differs by device.
-->

---

# Held-out validation

Pulls against the exact Sobol reference:

| process | pull range | ESS range |
|---|---|---:|
| `gg → t t̄` | −1.41 … +1.13 | 0.982 – 0.998 |
| `gg → t t̄ H` | −1.01 … +0.96 | 0.962 – 0.984 |
| `u ū → 4ℓ` | −2.23 … +0.63 | 0.929 – 0.943 |

The **−2.23 at 300 GeV is not a defect**: the published model has the same pull
(−2.3σ). Two independent models, same place.

<!-- notes:
Worth pre-empting the obvious question about the 2.23 sigma point. We checked
the published table — their pull is -2.3. So this is a property of the model at
that energy, not of our reproduction.
-->

---

# Baseline GPU validation — the null result

Identical `gg_ttbar` smoke run, only `--device` differs:

| | CPU | GPU |
|---|---:|---:|
| training wall clock | **303 s** | **304 s** |
| final ESS | 0.8054 | 0.8074 |

**The GPU bought 0.3 %.**

The first explanation written down — "the matrix element is the ceiling" — was
**wrong**. Profiling it directly gave a different answer.

<!-- notes:
Be explicit that we got this wrong first. The corrected version is the next
slide. This is a good example of why measuring beats reasoning from the code.
-->

---

# The correction: it depends on batch size

Flow (sample + log q + backward) vs matrix element, same points:

| points | flow CPU | flow GPU | GPU speedup | ME share |
|---:|---:|---:|---:|---:|
| 10 000 | 295 ms | **434 ms** | **0.68× — GPU loses** | 21 % |
| 30 000 | 735 ms | 520 ms | 1.41× | 38 % |
| 100 000 | 2 555 ms | **532 ms** | **4.80×** | 66 % |

The pipeline trains at **10 000** points per step. There the flow is ~79 % of
the step, and the GPU is **slower than the CPU** — too little work to amortise
kernel launches.

<!-- notes:
This is the central technical result. It is not that the flow is cheap; it is
that the GPU is starved. At 10k points the flow is the bottleneck and the GPU
cannot help. At 100k the GPU wins 4.8x and the matrix element takes over.
-->

---

# A lever that works today, with no code change

Same model, same 500 000 evaluations, only `--gpu-chunk` 10 000 → 100 000:

| E (GeV) | 10k chunk | 100k chunk | speedup |
|---:|---:|---:|---:|
| 400 | 14.17 s | 6.37 s | **2.23×** |
| 650 | 12.29 s | 6.19 s | 1.98× |
| 1000 | 10.91 s | 6.61 s | 1.65× |
| 1500 | 11.57 s | 6.01 s | 1.93× |
| 1950 | 11.42 s | 6.73 s | 1.70× |

<!-- notes:
A pure config change. Same checkpoint, same number of evaluations, identical
statistics within scatter. This is the cheapest win available and it should be
adopted regardless of what happens with Alpaka.
-->

---

# The CPU baseline is not one core either

Matrix-element throughput, multiprocessing pool (8-core laptop):

| cores | evals/s | speedup |
|---:|---:|---:|
| 1 | 103 033 | 1.00× |
| 8 | 475 464 | 4.61× |
| 16 | **614 298** | **5.96×** |

But at a 10 000-point batch the matrix element is only ~21 % of a step, so 6×
there is worth **~1.2× overall**. Right number for the gate definition; not a
shortcut.

<!-- notes:
This is the number the proposal's gate is defined against — 10x versus a
32-core CPU reference. A 32-core server will scale better than this laptop, so
the reference is faster than 614k evals/s.
-->

---

# What this means for Alpaka

**The flow is worth accelerating — ~79 % of a training step — but it is
overhead-bound, not arithmetic-bound.**

A straight port would inherit the same per-launch cost in C++.

> The case for Alpaka rests on **fusing the per-dimension chain into a single
> kernel launch**.

That gives the first deliverable a measurable acceptance test:

1. Measure the flow's fixed cost per call today.
2. Show a fused kernel collapses it.

<!-- notes:
This is a better position than the earlier note implied, and a more specific
one. We are not asking Alpaka to speed up arithmetic that is already fast; we
are asking it to remove overhead that PyTorch eager cannot.
-->

---

# Findings for the authors — budget over-provisioning

**The χ² stage never pays for itself.**

| process | ESS entering χ² | χ² best | outcome |
|---|---:|---:|---|
| 2→2 | 0.840873 | 0.840877 | 4,200 identical steps, then −0.0007 |
| 2→3 | 0.917502 | 0.917263 | −0.0002 |
| 2→4 | 0.506192 | 0.506192 | **`best step = 0`** — whole budget discarded |

Also: `kl_max_steps = 30 000` is never approached — every round stops at
**7 200** by patience, on all three processes.

<!-- notes:
The 8D case is unambiguous: CHI2 SUMMARY reports best step = 0, meaning it ran
6000 steps and restored its starting weights. Around an hour of compute for
nothing, and it happens on every process.
-->

---

# Findings for the authors — no checkpoint, no resume

`torch.save` is called **once**, at the very end.

The checkpoint holds model weights, marginals, graph and metrics — but **no
optimizer state, no RNG state, no step position**. It cannot be resumed even
from a saved file.

`u ū → 4ℓ` needed three attempts:

| attempt | duration | outcome |
|---|---:|---|
| 1 | 49 min | stopped |
| 2 | 9 h 20 min | machine restart — **all lost** |
| 3 | 13 h 15 min | complete |

<!-- notes:
A small change — periodic checkpointing — would have saved nine hours here.
It pays for itself the first time a shared node preempts a job.
-->

---

# Next steps

**This week**

1. Re-measure the training loop at a larger batch (32768) — CPU vs GPU.
2. Raise `--gpu-chunk` in the validation path — 1.65–2.2× for free.
3. Write up the batch-size finding for BNL and Brown.

**Then**

4. Alpaka hello-world on the CUDA backend (local 3060, then Talapas).
5. First kernel: cubic-Hermite marginal, measured as fixed-cost reduction.
6. Freeze the Python↔C++ boundary with Brown.

<!-- notes:
Item 1 is running now. Item 2 needs no discussion. Item 3 is the one that
changes other people's plans, so it should not wait for Sprint 2.
-->

---

# Decisions needed

1. **What exactly is the throughput gate measured on** — total event-generation
   rate, or the flow alone, and at what multiplicity?
2. **The 32-core CPU reference configuration** — what it runs, at what dimension.
3. **Repository home for MANGO code** — still a clone of a third-party repo.
4. **GPU allocation and administration** (Talapas).
5. **Operational definition of the accuracy gate** (BNL).

<!-- notes:
Item 1 is the one that shapes the Alpaka design. Items 3 and 4 have the longest
lead time. Item 5 is BNL's, but every performance number inherits it.
-->

---

# Summary

- Three processes **trained and validated** at full budget, GPU, from an
  independent build.
- The **Sobol column reproduces bit-for-bit** — the physics build is correct.
- The GPU's apparent uselessness was **a batch-size problem**, not a cheap flow.
- Corrected finding: **the flow dominates and is overhead-bound** → Alpaka
  should target launch fusion, and there is a **1.65–2.2× win available today**
  from a config change.

<!-- notes:
Close on the correction, not on the original claim. The useful outcome of
Phase 1 so far is that we now know what to optimize and what it is worth.
-->
