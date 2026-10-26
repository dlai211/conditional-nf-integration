# Higher-dimensional conditional NF — reproduction notes

**Date:** 2026-10-22  
**Scope:** the `sparse_wave_32` synthetic target and the `uu_ww_4l` 8-D run.  
**Question:** does the conditional NF "break" in higher dimensions, and if so,
where?

## Short answer

The **model code does not crash** at 32-D. Sampling, `log_prob`, the defensive
mixture and a full backward pass all work at 32-D. What breaks is the
**training setup**:

1. the smoke profile uses a five-point annealing schedule that is too coarse for
   32-D, so the weighted cache collapses to an effective sample size of ~3;
2. the smoke profile's cache (32 conditions × 128 points = 4,096) is far too
   small for a 32-dimensional model, so the marginal and bridge fits overfit
   and early-stop;
3. the likelihood loops over dimensions and runs 40 bisection steps per
   element, which makes 32-D training ~3.5× slower per step than 8-D before
   any budget increase.

The first two are reproduced below and are fixable. The third is a scaling wall,
not a correctness bug.

## What was run

```bash
# 32-D smoke with the default (coarse) schedule
python experiments/run_adaptive_integrand_structure_search.py \
    --benchmarks sparse_wave_32 --speed-profile smoke --device cuda \
    --no-compile-kernels --no-compile-model --output /tmp/sparse32_smoke

# 32-D smoke with the fine high-dimensional schedule and reduced inverse cost
python experiments/run_adaptive_integrand_structure_search.py \
    --benchmarks sparse_wave_32 --speed-profile smoke --device cuda \
    --no-compile-kernels --no-compile-model \
    --inverse-iterations 12 \
    --temperatures 0.01 0.025 0.05 0.1 0.25 1.0 \
    --output /tmp/sparse32_check
```

## Observation 1 — the coarse schedule collapses

`sparse_wave_32`, smoke profile, default schedule `(0.02, 0.08, 0.25, 0.55, 1.00)`:

| stage β | median normalized ESS | median raw ESS | what happened |
|---:|---:|---:|---|
| 0.02 | 0.400 | 51.2 | usable |
| 0.08 | **0.022** | **2.87** | one or two points dominate |
| 0.25 | **0.008** | **1.06** | cache is effectively one sample |

The marginal fit then diverges from its validation set
(`train NLL -4.56` against `val +0.095`) and early-stops on the degenerate cache.
This is the failure mode that looks like "the code breaks": no exception, but
the run stops learning.

## Observation 2 — the fine schedule restores the cache

Same target, schedule `(0.01, 0.025, 0.05, 0.1, 0.25, 1.0)`, 12 inverse steps:

| stage β | median normalized ESS | median raw ESS |
|---:|---:|---:|
| 0.01 | 0.757 | **96.9** |
| 0.025 | 0.256 | **32.7** |

The fine schedule keeps the cache usable at the point where the coarse one had
already collapsed. The run was stopped after the β=0.025 cache, so stages above
0.025 were not verified in this session.

Even with the fine schedule, the **smoke cache is too small**: the marginal fit
goes `train 0.034 → -0.259` while `val 0.079 → 0.096`, and the bridge fit goes
`train -0.392 → -0.959` while `val 0.194 → 0.369`. Both early-stop. This is a
budget issue, not a numerical bug: 4,096 points cannot fit 32 marginals plus a
31-conditional copula. The thorough profile uses 1,024 conditions × 512 points
and does not have this problem.

## Observation 3 — the core model is dimension-generic

These all pass at 32-D with the full autoregressive graph (496 edges):

- `MarginalBank(32)`, `StructuredCopula(32, full_ar_graph(32))`,
  `MarginalCopulaModel` construction;
- `model.sample(16, cond)` and `model.log_prob(x, cond)`;
- `(-log_prob).backward()` through all 32 marginals and 31 copula conditionals;
- `sample_from_proposal(..., uniform_fraction=0.1)`;
- `MarginalCopulaFlow(32, ...)` from the bin sweep, including its backward pass.

So the "break" is not a shape, context-dimension or autograd bug in the model
classes.

## Observation 4 — the per-dimension inverse is the scaling wall

The marginal likelihood calls `ConditionalCubicFlow1D.inverse` once per
dimension; each call runs 40 bisection iterations and then one Newton step.
Measured with batch 2,048 on the RTX 3060 (float32, forward + backward):

| dim | 40 steps | 12 steps | 8 steps | 4 steps |
|---:|---:|---:|---:|---:|
| 8 | 454 ms | 220 ms | 140 ms | 118 ms |
| 32 | **1,608 ms** | **750 ms** | **616 ms** | **464 ms** |

Accuracy against the 40-step reference, float32, 4,096 points:

| steps | max \|Δu\| | max \|Δlog-Jacobian\| |
|---:|---:|---:|
| 12 | 2.4e-7 | 1.0e-5 |
| 8 | 2.4e-7 | 1.1e-5 |
| 4 | 2.0e-5 | 1.7e-3 |

The existing Newton correction is what makes 8–12 steps sufficient; the
bisection only has to supply a safe starting point. 4 steps starts to degrade.

The same per-dimension loop exists in the copula likelihood (31 conditionals at
32-D), so the full bridge step is slower than the marginal measurement above.

## Observation 5 — a dtype bug in the sampling helper (secondary)

With a **float64** model, the defensive mixture raised:

```
RuntimeError: Index put requires the source and destination dtypes match,
got Double for the destination and Float for the source.
```

`torch.rand` was called without `dtype` in `sample_from_proposal`,
`StructuredCopula.sample_u`, and the bin-sweep `MarginalCopulaFlow.sample`.
The float32 pipeline never hits this, but any double-precision evaluation or
test does.

## Changes made

| change | file |
|---|---|
| smoke profile uses `HIGH_DIM_MARGINAL_TEMPERATURES` when `DIM >= 32` | `experiments/run_adaptive_integrand_structure_search.py` |
| `--inverse-iterations` (default 40; 8–12 for high-dimensional training) | same |
| `--temperatures` override for the fixed schedule | same |
| dtype-correct uniform draws in the proposal and copula samplers | same, `run_bayesian_selected_copula_benchmark.py`, `run_copula_bin_sweep.py` |

The default behaviour is unchanged: the schedule change only applies to
`DIM >= 32` under `--speed-profile smoke`, and the iteration default remains 40.
The upstream test suite still passes (`12 passed`).

## Recommendations

1. **Do not use `--speed-profile smoke` to judge 32-D accuracy.** Use `fast`,
   `balanced` or `thorough`; they use the adaptive/fine schedule and a
   dimension-appropriate cache.
2. For 32-D training, pass `--inverse-iterations 8` or `12`. The 2–2.6× speedup
   is free at the 1e-5 log-Jacobian level.
3. If a smoke run at 32-D is needed, override the schedule with
   `--temperatures 0.01 0.025 0.05 0.1 0.25 1.0` and accept that the smoke
   cache is still too small for accuracy.
4. The next real fix is to **vectorize the per-dimension marginals** (stack the
   32 MLPs and evaluate them in one batched pass) and/or replace the 40-step
   bisection with a Newton solve initialized from the bin's linear estimate.
   The latter is what `--inverse-iterations` approximates without changing the
   evaluation reference.
5. The full 32-D `thorough` run has **not** been completed here; it is the
   remaining verification. With 40 steps it is several times the 8-D 13-hour
   run; with 12 steps the scaling is more favourable but still long.
