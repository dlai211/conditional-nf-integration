> **Current work (2026-10-26):** see [`HANDOFF.md`](HANDOFF.md) — the balanced
> `gg_zjj` training run was stopped before it wrote a checkpoint. The handoff
> has the restart command and the 200k uniform comparison that follows.

# Execution Policy

The rules below are **scoped by machine**. Read the section for the machine you
are on. The shared-login-node limits in §1–§2 apply to nothing else.

---

## 0. Local machine — no limits

`C:\Users\laiji\test` on Windows 11 is a personal workstation with no
shared-facility CPU policy. On this machine:

- **Long runs are fine.** The `thorough` profiles take 5–20 h; `gg_ttbar` took
  5 h 05 min and `gg_tth` 12 h 26 min. Run them here directly.
- **Use every core.** Do not prepend `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1` or any other thread guard. Those exist to protect a
  shared login node that is not involved here.
- **Background jobs are fine.** `nohup … &` and the harness's background mode
  are both allowed.
- **No batch system.** No HTCondor, no `.sub` files, no `+JobFlavour` — those
  are lxplus requirements (§2–§3).
- **GPU runs** pass `--device cuda` and **drop** `--no-compile-kernels` so the
  deterministic integrand kernel can compile.
- **Use `--speed-profile thorough`** for publication-quality numbers. `smoke`
  exists for checking that the plumbing works, not for results.

Note that the GPU is *not* the bottleneck on this workload — the matrix element
is evaluated in a single-threaded Python loop on the CPU. See
`REPRODUCTION_COMMANDS.md` §9.2. Adding threads or a bigger GPU here buys
little; `--me-chunk` and a faster CPU are the levers that matter.

---

## 1. Shared login nodes (LxPlus / AF / SDCC) — hard limits

You are running on a shared cluster login node.
You MUST adhere to these strict limits to avoid account suspension:

- NEVER run full training loops, parameter sweeps, or multi-seed runs in the
  local terminal.
- NEVER run processes longer than 3 minutes locally.
- For local smoke tests or syntax checks, ALWAYS prepend single-core/low-thread
  env vars:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1 python <script> --smoke`
- Never spawn background jobs using `&` or concurrent subshells without user
  approval.

## 2. Batch Execution Rules (HTCondor, lxplus only)

- Any script that trains a model, sweeps hyperparameters, or runs integration
  searches MUST be dispatched via HTCondor batch.
- When continuing or executing sweeps:
  1. Generate or update a `.sub` submit file and an executable `.sh` wrapper.
  2. Test the wrapper once locally for 1 step/batch using `--smoke`.
  3. Submit using `condor_submit <file>.sub`.
  4. Inspect status with `condor_q` and check `/tmp/` or designated log dirs.

## 3. CERN-Specific Submission Requirements (EosSubmit)

This repository lives on `/eos`. CERN's **standard batch schedds reject any
submit file that references `/eos` paths**, failing with:

```
ERROR: Failed to commit job submission into the queue.
     | Standard batch schedds cannot use /eos paths directly within the submit file.
```

### 3.1 Always load the EosSubmit schedd module first

There is no submit-file attribute and no `condor_submit` flag for this — it is
selected by the shell environment, and it does not persist across shells:

```bash
module load lxbatch/eossubmit
condor_submit sweeps.sub
```

Every `condor_*` command that must see these jobs (`condor_q`, `condor_rm`)
needs the module loaded in that same shell. To go back to the standard
schedds: `module unload lxbatch/eossubmit` or `module load lxbatch/share`.

Reference: https://batchdocs.web.cern.ch/local/eossubmit.html

### 3.2 `executable` must be an absolute `/eos` path

`initialdir` is used for I/O files but **not** for the executable. A bare
`executable = run_batch.sh` is rejected even with `initialdir` set. Use:

```
executable = /eos/user/j/jlai/mango/conditional-nf-integration/run_batch.sh
initialdir = /eos/user/j/jlai/mango/conditional-nf-integration
```

Under EosSubmit the schedd transparently rewrites this to
`root://eosuser.cern.ch//eos/...` (xrootd), so no FUSE or shared filesystem is
traversed by the job.

### 3.3 Other EosSubmit constraints

- Every job-referenced file (executable, log, stdout/err, inputs) must live on
  `/eos`. AFS paths are not accepted.
- **One log file per cluster**: the `log` name may use `$(ClusterId)` but must
  **not** contain `$(ProcId)`. (`output` and `error` may use both.)

### 3.4 Budget flags are mandatory in the submit file

The defaults of these drivers are *not* smoke-sized, and a job that exceeds its
`+JobFlavour` wall time is killed with no auto-resubmit and no checkpoint:

- `run_adaptive_integrand_structure_search.py` defaults to
  `--speed-profile thorough` (`kl_max_steps=30_000` vs smoke's 300) — a
  multi-day job. Always pass `--speed-profile smoke` for a batch smoke test.
- `run_copula_bin_sweep.py` without `--smoke` runs
  4 benchmarks × 3 dims × 4 bins × 3 seeds = **144 models**. Always pass
  `--smoke` for a batch smoke test.

`+JobFlavour` is a maximum wall time, not a reservation: a job that finishes
early still exits early, so requesting a longer flavour costs nothing and
avoids mid-run kills. **A thorough run needs `tomorrow` (24 h), not `workday`
(8 h)** — `gg_tth` measured 12 h 26 min, and the driver writes no intermediate
checkpoint, so a kill at the wall time loses the whole run.

### 3.5 Thread allocation for batch slots

`venv/bin/activate` installs `OMP_NUM_THREADS=1` guards for interactive use on
the shared login node. `run_batch.sh` captures the batch allocation *before*
sourcing `activate` and re-applies it afterwards, so `request_cpus` is honoured
instead of being silently overridden by the guard. Local smoke tests of the
wrapper should pass `OMP_NUM_THREADS=1` explicitly.

Note: `OMP_NUM_THREADS` caps PyTorch's intra-op pool only. The inter-op pool
(`torch.get_num_interop_threads()`, typically = core count) is not
environment-controllable in torch 2.14 and would need an in-process
`torch.set_num_interop_threads(1)`. It is not exercised by eager-mode training
here.
