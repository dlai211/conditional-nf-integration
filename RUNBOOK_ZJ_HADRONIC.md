# RUNBOOK — hadronic `pp → Z + j` campaign (Linux/ROCm)

Commands used for the `zj_hadronic_low` / `zj_hadronic_high` campaign on the
`odyssey` node. Paths are absolute as used. The results and their caveats are
in `RESULTS.md` §8.

## 0. Prerequisites

- Repo: `/home/users/dlai/conditional-nf-integration`
- MG5 3.6.7: `/storage/users/dlai/mango-linux/MG5_aMC_v3_6_7`,
  wrapper `/storage/users/dlai/mango-linux/bin/mg5_aMC`
- Env: `/storage/users/dlai/mango-linux/env_linux.sh`
- 4× AMD MI300A (ROCm), 192 logical CPUs, 512 GB RAM
- Configs: `configs/processes/zj_hadronic_low.json`,
  `configs/processes/zj_hadronic_high.json`
- Standalone: `generated/zj_hadronic_standalone` (six `P1_*` subprocesses)

## 1. Environment and LHAPDF

```bash
source /storage/users/dlai/mango-linux/env_linux.sh

# LHAPDF conda env (once)
CONDA_PKGS_DIRS=/storage/users/dlai/conda_pkgs \
  /packages/miniconda-24.9.2/bin/conda create -y \
  -p /storage/users/dlai/lhapdf-env -c conda-forge python=3.10 lhapdf

export LHAPDF_DATA_PATH=/storage/users/dlai/lhapdf-env/share/LHAPDF
export PYTHONPATH=/storage/users/dlai/lhapdf-env/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/storage/users/dlai/lhapdf-env/lib:$LD_LIBRARY_PATH
# Needed for the py310 env to load libLHAPDF (CXXABI_1.3.15):
export LD_PRELOAD=/storage/users/dlai/lhapdf-env/lib/libstdc++.so.6

# The CLI must run with the env Python and explicit --listdir/--pdfdir.
/storage/users/dlai/lhapdf-env/bin/python \
  /storage/users/dlai/lhapdf-env/bin/lhapdf \
  --listdir "$LHAPDF_DATA_PATH" --pdfdir "$LHAPDF_DATA_PATH" \
  install NNPDF23_lo_as_0130_qed
```

The matching set is **`NNPDF23_lo_as_0130_qed`, LHAPDF ID 247000**, the set
behind MG5's `nn23lo1`. ID 230000 is `NNPDF23_nlo_as_0119` (NLO,
α_s = 0.119) and does **not** match the baseline. Quick check:

```bash
python3 -c "import lhapdf; p=lhapdf.mkPDF('NNPDF23_lo_as_0130_qed',0); print(p.alphasQ(91.188), p.xfxQ(2,0.1,91.188**2))"
```

## 2. Standalone generation (six subprocess groups)

```bash
cd /home/users/dlai/conditional-nf-integration
python3 experiments/setup_madgraph_process.py \
  --config configs/processes/zj_hadronic_low.json \
  --mg5 /storage/users/dlai/mango-linux/bin/mg5_aMC
```

This writes `generated/zj_hadronic_standalone` and compiles the per-`P1_*`
`matrix2py` wrappers. `@1` in MG5's log is a no-op for this process: the old
`g u > z u @1`, the explicit `g u > z u` and the `p p > z j` `P1_gu_zu` were
checked at identical momenta and are bit-identical.

## 3. Balanced training — 12 models

Driver: `/storage/users/dlai/mango-linux/run_hadronic_balanced_one.sbatch`
(48 CPUs, 1 GPU, `--speed-profile balanced --batch-size 30000 --is-points 128
--me-chunk 512 --me-workers 48`, timestamped log through
`scripts/timestamp_log.py`).

```bash
cd /storage/users/dlai/mango-linux

# wave 1 — low-range g q channels
for spec in "gu_zu 0" "gd_zd 1" "gux_zux 2" "gdx_zdx 3"; do
  set -- $spec
  sbatch --job-name=hb-low-$1 \
    --export=ALL,CONFIG_NAME=zj_hadronic_low,CHANNEL=$1,GPU_ID=$2 \
    run_hadronic_balanced_one.sbatch
done

# wave 2 — remaining low q q~ and first high g q
sbatch --job-name=hb-low-uux_zg  --export=ALL,CONFIG_NAME=zj_hadronic_low,CHANNEL=uux_zg,GPU_ID=0 run_hadronic_balanced_one.sbatch
sbatch --job-name=hb-low-ddx_zg  --export=ALL,CONFIG_NAME=zj_hadronic_low,CHANNEL=ddx_zg,GPU_ID=1 run_hadronic_balanced_one.sbatch
sbatch --job-name=hb-high-gu_zu  --export=ALL,CONFIG_NAME=zj_hadronic_high,CHANNEL=gu_zu,GPU_ID=2 run_hadronic_balanced_one.sbatch
sbatch --job-name=hb-high-gd_zd  --export=ALL,CONFIG_NAME=zj_hadronic_high,CHANNEL=gd_zd,GPU_ID=3 run_hadronic_balanced_one.sbatch

# wave 3 — remaining high channels
sbatch --job-name=hb-high-gux_zux --export=ALL,CONFIG_NAME=zj_hadronic_high,CHANNEL=gux_zux,GPU_ID=0 run_hadronic_balanced_one.sbatch
sbatch --job-name=hb-high-gdx_zdx --export=ALL,CONFIG_NAME=zj_hadronic_high,CHANNEL=gdx_zdx,GPU_ID=1 run_hadronic_balanced_one.sbatch
sbatch --job-name=hb-high-uux_zg  --export=ALL,CONFIG_NAME=zj_hadronic_high,CHANNEL=uux_zg,GPU_ID=2 run_hadronic_balanced_one.sbatch
sbatch --job-name=hb-high-ddx_zg  --export=ALL,CONFIG_NAME=zj_hadronic_high,CHANNEL=ddx_zg,GPU_ID=3 run_hadronic_balanced_one.sbatch
```

Outputs:

```text
results/zj_hadronic_{low,high}_{channel}_copula_balanced_gpu/
  madgraph_zj_hadronic_{low,high}_{channel}_2_final_model.pt
  madgraph_zj_hadronic_{low,high}_{channel}_2_final_summary.csv
logs/hadronic-balanced-hb-{low,high}-{channel}-<jobid>.log
```

Four jobs × 48 CPUs run concurrently; the campaign wall was 1.19 h and the
summed job time 4.30 h. The low `g q` models reach ESS ≈ 0.96, low `q q~`
≈ 0.78, high ≈ 0.31.

## 4. Exact partonic MadEvent interpolation

```bash
cd /storage/users/dlai/mango-linux
sbatch --job-name=hadronic-interpolation run_hadronic_interpolation_madevent.sbatch
```

The driver evaluates 50 linear points in `[114, 250]` and 50 in
`[200, 2000]` (94 unique) for each of the six channels with
`validate_madgraph_process_with_madevent.py --allow-outside-condition
--survey-points 10000 --survey-iterations 6 --refine-accuracy 0.001`.
Outputs:

```text
results/zj_hadronic_madevent_interpolation/<channel>/comparison.csv
```

In this run the exact scan showed the high-range models agreeing at
1228.57 GeV (e.g. `gu_zu` 0.6086 vs 0.60815 pb) and the low-range `gu_zu`
20–25× low at 114–120 GeV.

## 5. Luminosity integration

```bash
cd /storage/users/dlai/mango-linux
sbatch --job-name=hadronic-integration --export=ALL,GPU_ID=0 \
  run_hadronic_integration.sbatch
```

The driver sets the LHAPDF paths and the `LD_PRELOAD` above, then runs

```bash
cd "$REPO/generated/zj_hadronic_standalone"
python3 -u "$REPO/experiments/integrate_hadronic_process.py" \
  --device cuda --output "$REPO/results/zj_hadronic_integral"
```

It evaluates each NF on 60 log-spaced points per range, multiplies by
`dL/dsqrt(s-hat)`, integrates, sums the six channels, and adds the
0.1389 ± 0.00026 pb `[2000, 13000]` tail. Outputs:

```text
results/zj_hadronic_integral/summary.json
results/zj_hadronic_integral/per_channel.csv
results/zj_hadronic_integral/grid_<channel>.csv
results/zj_hadronic_integral/interpolation_{low,high}.csv
```

Recorded result: **13765.5 ± 0.59 pb**, 1.50× the native 9147.5 ± 10.32 pb.
This is **not a valid result**; the low-range luminosity is 5–13× high and the
low `gu_zu` model is 20–25× low at the threshold.

## 6. Plots and benchmark table

```bash
cd /home/users/dlai/conditional-nf-integration
python3 scripts/plot_nf_diagnostics.py
python3 scripts/make_hadronic_benchmark_table.py
```

Outputs:

```text
results/zj_hadronic_integral/plots/loss_ess_low.png
results/zj_hadronic_integral/plots/loss_ess_high.png
results/zj_hadronic_integral/plots/interpolation_low.png
results/zj_hadronic_integral/plots/interpolation_high.png
results/zj_hadronic_integral/benchmark_table.md
```

## 7. Write RESULTS.md

```bash
python3 /storage/users/dlai/mango-linux/make_hadronic_results.py
```

This replaces the old §8 smoke campaign with the hadronic §8 (idempotent:
it searches for `## 8. Z + jets smoke campaign` and truncates there).

## 8. Native fixed-scale reference

The `p p > z j` MadEvent directory is in
`/storage/users/dlai/mango-linux/scratch_ppzj/ppzj_madevent`.

Run card for the comparison (set with the `update_run_card` helper in
`experiments/validate_madgraph_process_with_madevent.py`):

```text
lpp1 = lpp2 = 1
ebeam1 = ebeam2 = 6500
pdlabel1 = pdlabel2 = nn23lo1
ptj = 20
etaj = 2.5
dsqrt_shat = 0
dsqrt_shatmax = -1
fixed_ren_scale = True
fixed_fac_scale1 = fixed_fac_scale2 = True
scale = 91.188
dsqrt_q2fact1 = dsqrt_q2fact2 = 91.188
use_syst = False
```

Commands:

```bash
cd /storage/users/dlai/mango-linux/scratch_ppzj/ppzj_madevent
# command file: survey total_fixed --points=20000 --iterations=8 --accuracy=0.002
#               refine 0.001 3
/usr/bin/time -f 'wall_seconds=%e' -o /tmp/native_time.txt \
  ./bin/madevent total_fixed_commands.txt
head -1 SubProcesses/results.dat
```

Recorded: **9147.5 ± 10.32 pb**, 39.4 s, 1,360,036 integration points. The
default dynamic-scale run is **9010 ± 10 pb** and is not the NF comparison
number because the NF side uses the fixed scale.

## 9. Cut-removal test (optional, negative result)

The low config can be run without `p_T,j > 20` / `|eta_j| < 2.5`:

```bash
cd /home/users/dlai/conditional-nf-integration/generated/zj_hadronic_standalone
source /storage/users/dlai/mango-linux/env_linux.sh
python3 -u /home/users/dlai/conditional-nf-integration/experiments/run_madgraph_process_conditional_copula.py \
  --config /home/users/dlai/conditional-nf-integration/configs/processes/zj_hadronic_low.json \
  --channel gu_zu --speed-profile smoke --device cpu --me-workers 8 \
  --output /home/users/dlai/conditional-nf-integration/results/_test_zj_low_nocuts_gu_zu_smoke
```

The uncut integrand is collinearly divergent: the smoke's final ESS is
**0.0029** versus **0.865** with cuts. The cuts are the jet definition, not
the source of the low-model bias; do not use the uncut low config for
production. Restore the `cuts` block in `zj_hadronic_low.json` before any
integration.

## 10. Known issues / troubleshooting

- **`lha_read` / `ident_card.dat` warning**: every training/evaluation process
  must run with CWD inside `generated/zj_hadronic_standalone`; the sbatch
  drivers already `cd` there. Running from the repo root makes the matrix
  element initialization fail/hang.
- **`CXXABI_1.3.15` ImportError**: the conda `libLHAPDF.so` needs the conda
  libstdc++. Export
  `LD_PRELOAD=/storage/users/dlai/lhapdf-env/lib/libstdc++.so.6` for the
  py310 env; the gcc 13.2 libstdc++ only provides 1.3.14 and py310 provides
  1.3.13.
- **LHAPDF install “No PDFs known”**: run the `lhapdf` script with the env
  Python and pass `--listdir`/`--pdfdir` explicitly (see §1).
- **PDF ID mismatch**: `nn23lo1` is the LO as_0130 set, LHAPDF ID **247000**;
  the run-card ID 230000 is `NNPDF23_nlo_as_0119`.
- **`@1`**: MG5 prints it in the diagram log, but for `g u > z u` the
  generated amplitude is identical with and without `@1`; the configs use the
  no-`@1` process string.
- **Scale**: the native default is dynamic (9010 pb); the fixed-scale
  comparison used with the NF is 9147.5 pb.
- **NF normalization failure**: the 13765.5 pb NF total is invalid. The
  high-range models match the exact partonic scan; the low-range `gu_zu`
  model is 20–25× low at 114–120 GeV, and the luminosity is 5–13× high in the
  low range. Do not quote the NF total as a physics result. The planned fix is
  a log-`sqrt(s-hat)` condition feature plus log-uniform condition sampling
  for the six low models, followed by a luminosity audit.
- **NF performance**: for this 2-D integral the NF is not competitive:
  generation uses 72M samples / 12.7 min and training 3.84×10⁹ samples /
  4.30 h, versus 1.36M samples / 39.4 s for native MadEvent.
