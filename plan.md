# plan.md — MANGO Phase 1 workstream plan

**Scope of this plan:** the BNL workstream of MANGO Phase-I (DE-FOA-0003612, Focus
Area 14C), whose Months 1–3 charter in the proposal is *"Generate MadGraph data
& establish validation metrics."* The other two institutions own the GPU
framework (Oregon) and the CNF/flow-matching architecture (Brown); this plan
covers what feeds them and what they are measured against.

**Reference:** `MANGO_2026_narrative_27Apr2026.pdf` — §4 (milestones and sprint
split), §5 (data sources and models), §6 (decision gate metrics).

---

## 1. Where we start from

The `conditional-nf-integration` repository is a working conditional
normalizing flow pipeline for MadGraph matrix elements. As of this session it
has been rebuilt and run end to end on Windows, and it supplies most of the
scaffolding Phase 1 would otherwise have to build from scratch:

| asset | state |
|---|---|
| MadGraph 3.6.7 standalone build chain | working, generate → compile → validate |
| LO processes built and integrand-validated | `gg→t t̄` (2D), `gg→t t̄H` (5D), `u ū→W⁺W⁻→4ℓ` (8D) |
| reproduction of independent Sobol integrals | bit-identical to `results_public/`, 15/15 |
| exact-likelihood importance-sampling estimator | working, with ESS/N diagnostics |
| conditional flow (marginals × copula, conditioned on √ŝ) | working |
| synthetic benchmarks with known integrals | working (6 targets incl. 32D) |
| GPU | **none** — everything so far is CPU-only |

Two things it does **not** have, and both are central to MANGO: a **continuous**
normalizing flow (the repo uses discrete cubic-spline marginals plus an
autoregressive copula, not a neural ODE), and any **GPU** execution at all.

### The gap is not the whole distance

The proposal's own plan is to start from RealNVP/affine coupling and *then* move
to continuous flows. The repo already establishes the evaluation harness, the
phase-space maps and the validation protocol that both will be measured with.
The Phase 1 job is therefore to build the target processes and the metric
contract, and to hand Brown a common harness — not to write a CNF from zero.

---

## 2. Phase 1 — Months 1–3

**Objective in one line:** by the end of Month 3, the Z+jets ladder exists in
the pipeline, the validation-metric contract all three institutions will use is
frozen and implemented, and the first reference dataset is archived — so that
Brown's CNF work and Oregon's Alpaka work have something real to be measured
against from the first day of Sprint 2.

### 2.1 Deliverable A — the Z+jets process ladder

The proposal's baseline is LO `q q̄ → Z g`; its stated next step is jet
multiplicity. The existing phase-space code already covers the shapes needed,
which makes this cheaper than it looks:

| target | final state | multiplicity | phase space | reuses |
|---|---|---|---|---|
| the proposal's baseline | `q q̄ → Z g` | 2→2 | 2D | existing `two_body` map as-is |
| one more jet | `q q̄ → Z g g` | 2→3 | 5D | existing `three_body` map as-is |
| two more jets | `q q̄ → Z g g g` | 2→4 | 8D | needs a *generic* 4-body map (the existing 8D map is the specialised diboson one) |

(Final states are written out rather than counted as "Z + n jets", because the
jet counting is ambiguous — `q q̄ → Z g` is itself the LO Z+1-jet process.)

*Acceptance:* each process passes `validate_madgraph_process_integrand.py`
(4-momentum conservation at the 1e-13 level, mass shells, strictly positive
Jacobian, integrand finite and positive everywhere sampled), and its Sobol
cross-check reproduces against an independent implementation.

**Why this ladder matters beyond "more processes":** it is the only way to test
the throughput gate honestly (see §2.4).

### 2.2 Deliverable B — the validation-metric contract

This is the workstream's headline deliverable and the thing with the longest
downstream reach, because Brown and Oregon both need it. One module, one
protocol, versioned:

| metric | definition | why |
|---|---|---|
| cross-section agreement | `(σ_NF − σ_MG)/σ_MG` and a pull, at fixed QMC protocol | the proposal's Physics Accuracy gate |
| ESS/N | `(Σw)²/(N Σw²)` on fresh samples | proposal quality; already working |
| **coverage** | fraction of held-out conditions where the true integral lies in the quoted 68/95 % interval | **currently broken — see §4** |
| throughput | events/s, single GPU, vs a 32-core CPU importance-sampling reference | the proposal's Computational Throughput gate |
| reproducibility protocol | fixed seeds, fixed QMC power and replicate count | what made the 15/15 bit-identical check possible |

*Acceptance:* a single entry point that takes a process + checkpoint and emits
all five numbers in one table, with the protocol version recorded in the output
so two institutions can tell whether their numbers are comparable.

### 2.3 Deliverable C — the reference dataset

The proposal commits to generating data on the fly and archiving it on the DOE
American Science Cloud with metadata.

* a frozen, versioned set of ME-weighted phase-space points for each process in
  the ladder, at a fixed grid of √ŝ conditions;
* a metadata schema covering process, √ŝ range, α_s, MadGraph and model version,
  card hashes, phase-space map version, and the generation commit;
* a manifest with content hashes, so a consumer can verify what they have.

*Acceptance:* someone at Brown can pull the dataset, verify it against the
manifest, and train on it without asking us a question.

### 2.4 Deliverable D — the throughput benchmark, done where it can be won

**A strategic point that should be settled in Phase 1, not Phase 2.** The
proposal's gate is ≥10× faster than a 32-core CPU importance-sampling reference.
Methods of this class do *not* win in low dimension, and the repository's own
numbers say so: on a 2D synthetic target VEGAS beats the flow on both accuracy
and wall-clock, while the repository's headline result — the one the whole
"VEGAS fails here" claim rests on — is **32D**.

That 2D comparison was a deliberately tiny smoke-budget run, so it should not be
quoted as a settled number. But the direction is not an artefact of the budget:
axis-adaptive importance sampling is genuinely strong in low dimension, and the
case for learned proposals is about dimension. Treat the principle as
well-founded and the specific figures as illustrative.

*Therefore:* benchmark the throughput gate on the high-multiplicity end of the
ladder (Z + 2 jets and up), and state the dimension alongside every throughput
number. A gate measured on `q q̄ → Z g` alone would likely be reported as a
failure of the method when it is a failure of the benchmark choice.

*Acceptance:* a documented throughput protocol plus one measured number on real
GPU hardware, with the dimension stated.

### 2.5 Deliverable E — the architecture bridge

Not to build the CNF — that is Brown's — but to make sure the harness is ready
for it:

* implement a RealNVP/affine-coupling baseline on the Z+jets ladder (the
  proposal's stated first architecture), which exercises the harness against a
  second, structurally different model;
* specify the interface a CNF must satisfy to plug into the same metrics module
  (density evaluation, sampling, conditioning);
* confirm the conditional-√ŝ machinery carries over, so the CNF is trained once
  per process rather than once per energy point.

### 2.6 Phase 1 exit criteria

Phase 1 is done when:

1. the Z+jets ladder builds, validates and Sobol-checks;
2. the metrics module produces all five numbers from one command, versioned;
3. coverage is measured for every process and either fixed or characterised;
4. the reference dataset is frozen, hashed and archived;
5. one throughput number exists on real GPU hardware, at a stated dimension;
6. the RealNVP baseline runs through the same harness.

---

## 3. Next week

Six concrete items, ordered by what unblocks what.

**1. Add `q q̄ → Z g` to the pipeline.**
New `configs/processes/qq_zg.json`, then the same three commands that already
work for `gg_ttbar`: generate, integrand-validate, Sobol-check. This is
mechanical — the 2-body map needs no changes — and it is the process the
proposal's Figure 2 already shows, so it should exist before anything else.

**2. Decide the repository home for MANGO work.**
`conditional-nf-integration` is a third party's published repository (the BNL
author's), currently carrying our commits on a branch of a clone. MANGO needs
its own repository before Sprint 1 delivers anything, with this pipeline pulled
in as a dependency or a vendored module. Decide this before more code lands,
not after.

**3. Get GPU access settled.**
Everything built so far is CPU-only because this machine has no GPU. The
throughput deliverable (§2.4) cannot start without one. Candidate routes:
Oregon's Talapas cluster (120 GPUs per the proposal), BNL SDCC, or a cloud
allocation. Raise it this week — it has the longest lead time of anything here.

**4. Agree the metric definitions with BNL colleagues.**
Specifically: what "agreement within theoretical uncertainty" means
operationally for the accuracy gate (the proposal quotes 2–3 % for Z+jets NNLO,
up to 15 % at high p_T^Z — which of these is the pass mark?), and what exactly
the 32-core CPU reference in the throughput gate is running. These two
definitions determine whether Phase 1 delivers the right numbers.

**5. Write the coverage/calibration note.**
The existing 32D result reports 0.7875 coverage against a nominal 0.95, and an
8D run here showed the same signature — point estimates agreeing to sub-σ while
ESS sits low. Write up what is known, on which processes, at which budgets, as
a short internal note. It is the highest-risk item for the Phase 2 go/no-go
(§4) and it is cheap to document now while the numbers are fresh.

**6. Draft the dataset metadata schema.**
A one-page field list, reviewed before any generation run, so the first dataset
does not have to be regenerated to add provenance fields.

---

## 4. Risks worth naming now

**Coverage may invalidate the accuracy gate.** The proposal's Physics Accuracy
metric is agreement within theoretical uncertainty — but that is only testable
if the quoted uncertainties are calibrated. The repository's own 32D benchmark
reports 95 % coverage of 0.7875, and this session's 8D run showed the same
signature at lower dimension: point estimates agreeing with the reference to
0.42σ while ESS/N sat at 0.81–0.86 against a published 0.93. Accurate point
estimates with poor ESS is the signature of heavy importance-weight tails —
unbiased, but error bars too small. If that holds at Phase 1 scale, a
comparison "within theoretical uncertainty" can pass or fail for the wrong
reason. **This is the item most worth resolving early**, and it is why §3 item 5
is on next week's list rather than Phase 2's.

**The throughput gate is benchmark-dependent.** See §2.4. Pick the dimension
before picking the headline number.

**The CPU→GPU transition is a research task, not a port.** The existing pipeline
is Python/PyTorch; Alpaka is C++. The boundary between the two — where the flow
runs, where the matrix element is evaluated, what crosses per batch — is a
design decision that belongs to Oregon and Brown but needs our data format and
metric interface to be stable first. Freeze the interfaces in Phase 1.

**Training cost is excluded from the throughput claim, and should stay
explicit.** The proposal compares generation rate against a CPU reference. If
the flow has to be retrained per process *and* per energy point, the amortised
argument weakens. The conditional-in-√ŝ design already answers this — one model
per process across the whole energy range — and it is worth putting in front of
the collaboration as a Phase 1 talking point, because it is not currently in the
narrative.

---

## 5. Decisions needed from the collaboration

1. Which repository hosts MANGO code, and what the licence/attribution
   arrangement is with the upstream `conditional-nf-integration` author.
2. The operational definition of the accuracy gate (which uncertainty band).
3. The exact 32-core CPU reference configuration for the throughput gate.
4. GPU allocation and who administers it.
5. Whether the conditional-√ŝ reuse goes into the Phase 2 narrative.
