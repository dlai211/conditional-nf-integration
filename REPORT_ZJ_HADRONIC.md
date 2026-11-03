# Hadronic `pp -> Z + j` with conditional normalizing flows

**Status report — 2026-11-02**

This report covers the 12-model conditional-NF campaign for hadronic
`pp -> Z + j` at 13 TeV, the partonic validation, the LHAPDF luminosity
integration, the low/high ESS split, and the current normalization
discrepancy. It supersedes the incorrect "`gu_zu` 20–25× low" and
"luminosity 5–13× high" statements in the earlier `RESULTS.md` and
`RUNBOOK_ZJ_HADRONIC.md` drafts.

## Executive summary

- **12 trained models are ready**: six low-range and six high-range
  conditional NFs, 506 KB each, 6.0 MB total, with final summaries. They are
  local; `.pt` is gitignored and a fresh clone will not contain them.
- **The partonic models are correct.** All six channels agree with the exact
  partonic MadEvent scan: within **0.6%** at 114 GeV and within **0.3%** at
  1228.57 GeV. The old 118.76 pb entry is `uux_zg`, not `gu_zu`.
- **The NF hadronic grid is internally consistent.** The NF grid gives
  **13765.38 pb**; convolving the **exact partonic MadEvent** scan with the
  same LHAPDF luminosity gives **13761.97 pb**, a ratio of **1.00025**.
- **The unresolved discrepancy is native-vs-convolution, not NF-vs-matrix
  element.** The native fixed-scale MadEvent total is **9147.5 ± 10.32 pb**,
  **1.50× lower** than the LHAPDF convolution. The native `etaj=5.0` baseline
  is 11668.4 ± 72.03 pb; even with a wider jet acceptance it is below the NF
  `|eta_j|<2.5` total, which points at the native side. A no-rapidity native
  test gives **11915 ± 7.35 pb**; removing the eta cut accounts for ~60% of
  the gap, but the NF is still ~15.5% above it. The missing lab-frame boost
  is not the full explanation.
- **Do not quote 13765.5 pb as the physical cross section.** The correct
  physics reference for the stated native setup is **9147.5 ± 10.32 pb**.
  The NF total is an internally consistent convolution of the partonic models
  with LHAPDF, but the native/LHAPDF reconciliation is not finished.
- **ESS depends on which ESS is meant.** Training/held-out ESS is lower in
  the high range (0.231–0.313) than in the low range (0.775–0.963).
  Integration-grid ESS is high in both (low mean 0.962, high mean 0.972); the
  only low integration point is the 114 GeV threshold (0.467).

## 1. Setup

| item | value |
|---|---|
| process | `p p > z j`, full LO, no `@1` |
| collider | proton–proton, 13 TeV |
| subprocess groups | `gu_zu`, `gd_zd`, `gux_zux`, `gdx_zdx`, `uux_zg`, `ddx_zg` |
| low condition range | `sqrt(s-hat)` in [114, 250] GeV |
| high condition range | `sqrt(s-hat)` in [200, 2000] GeV |
| integration split | low [114, 250], high [250, 2000]; no overlap |
| jet cuts | `p_T,j > 20 GeV`, `|eta_j| < 2.5` |
| scale | fixed renormalization and factorization scale 91.188 GeV |
| alpha_s | 0.130 |
| PDF | LHAPDF `NNPDF23_lo_as_0130_qed`, ID 247000, member 0 |
| native PDF label | MG5 `nn23lo1`, which maps to `NNPDF23_lo_as_0130_qed_mem0.grid` |
| b quark | excluded on both sides by `maxjetflavor=4` |

### 1.1 c and s are included

The channel names are matrix-element representatives, not the full PDF
flavor list. The configs explicitly include c and s:

| channel | matrix element | `pdf_pairs` |
|---|---|---|
| `gu_zu` | `g u > z u` | `[21,2]`, `[21,4]` (g u, g c) |
| `gd_zd` | `g d > z d` | `[21,1]`, `[21,3]` (g d, g s) |
| `gux_zux` | `g u~ > z u~` | `[21,-2]`, `[21,-4]` (g u~, g c~) |
| `gdx_zdx` | `g d~ > z d~` | `[21,-1]`, `[21,-3]` (g d~, g s~) |
| `uux_zg` | `u u~ > z g` | `[2,-2]`, `[4,-4]` (u u~, c c~) |
| `ddx_zg` | `d d~ > z g` | `[1,-1]`, `[3,-3]` (d d~, s s~) |

At LO the c and s matrix elements are identical to u and d for massless
quarks; only the PDF weight differs, and the `pdf_pairs` supply it. The
native `p p > z j` includes the same six groups. The comparison is therefore
complete in the four-flavor scheme; it is not missing c or s. It excludes b
on both sides.

## 2. Models and training

| range | channels | checkpoints | total size | final summaries |
|---|---|---:|---:|---|
| low [114, 250] | 6 | 6 | 3.0 MB | 6 |
| high [200, 2000] | 6 | 6 | 3.0 MB | 6 |
| total | 12 | 12 | 6.0 MB | 12 |

Each checkpoint is `506 KB` and is accompanied by a `*_final_summary.csv`.
The `*_history.csv` files in this checkout are empty (2 bytes); the raw
training logs are not archived locally. The checkpoints and summaries are
therefore the usable artifacts. The `.pt` files are gitignored; copy them
if you move to another machine or clone fresh.

### 2.1 Training/held-out ESS and `std_logw`

| range | channel | ESS | std log w |
|---|---|---:|---:|
| low | `gu_zu` | 0.9592 | 0.1563 |
| low | `gd_zd` | 0.9614 | 0.1541 |
| low | `gux_zux` | 0.9613 | 0.1560 |
| low | `gdx_zdx` | 0.9628 | 0.1557 |
| low | `uux_zg` | 0.7755 | 0.4628 |
| low | `ddx_zg` | 0.7829 | 0.4609 |
| high | `gu_zu` | 0.3125 | 1.1277 |
| high | `gd_zd` | 0.3126 | 1.1279 |
| high | `gux_zux` | 0.3126 | 1.1287 |
| high | `gdx_zdx` | 0.3114 | 1.1282 |
| high | `uux_zg` | 0.2306 | 1.2194 |
| high | `ddx_zg` | 0.2309 | 1.2184 |

The high range has the **lower training/held-out ESS**: 0.231–0.313 versus
0.775–0.963 low. The low `q q~` channels (`uux_zg`, `ddx_zg`) are the weakest
low channels; the high `q q~` channels are the weakest overall.

**No retraining is needed for the current report.** The models are usable
for the partonic validation and the LHAPDF integral. Retraining the high
models would only be justified if high-region sampling efficiency is itself
a deliverable.

## 3. Partonic validation

The NF is compared with explicit partonic MadEvent at the same fixed scale,
alpha_s, and cuts. The low-range point at 114 GeV:

| channel | NF (pb) | MadEvent (pb) | NF/MadEvent |
|---|---:|---:|---:|
| `gu_zu` | 4.778 ± 0.023 | 4.754 ± 0.003 | 1.005 |
| `gd_zd` | 6.106 ± 0.027 | 6.096 ± 0.004 | 1.002 |
| `gux_zux` | 4.784 ± 0.021 | 4.754 ± 0.003 | 1.006 |
| `gdx_zdx` | 6.088 ± 0.026 | 6.096 ± 0.004 | 0.999 |
| `uux_zg` | 117.99 ± 0.57 | 118.76 ± 0.08 | 0.994 |
| `ddx_zg` | 151.84 ± 0.73 | 152.30 ± 0.10 | 0.997 |

The high-range point at 1228.57 GeV:

| channel | NF (pb) | MadEvent (pb) | NF/MadEvent |
|---|---:|---:|---:|
| `gu_zu` | 0.6086 ± 0.0004 | 0.60815 ± 0.00024 | 1.0007 |
| `gd_zd` | 0.7796 ± 0.0004 | 0.77988 ± 0.00031 | 0.9996 |
| `gux_zux` | 0.6083 ± 0.0004 | 0.60815 ± 0.00024 | 1.0003 |
| `gdx_zdx` | 0.7803 ± 0.0009 | 0.77988 ± 0.00031 | 1.0005 |
| `uux_zg` | 2.4108 ± 0.0017 | 2.4029 ± 0.0054 | 1.0033 |
| `ddx_zg` | 3.0919 ± 0.0021 | 3.0814 ± 0.0069 | 1.0034 |

**Correction to the earlier documentation.** The old low-range table paired
`gu_zu` NF 4.78 pb with MadEvent 118.76 pb and concluded that `gu_zu` was
20–25× low. 118.76 pb is `uux_zg`; `gu_zu` is 4.754 pb. All six low channels
agree at the 0.6% level. The six-channel **summed** curve in
`interpolation_low.png` is not a per-channel `gu_zu` curve; it is the sum over
all six channels.

## 4. Hadronic integration

The integration evaluates each channel on a 60-point log-spaced grid per
range, multiplies by the LHAPDF luminosity `dL/dsqrt(s-hat)`, integrates over
`sqrt(s-hat)`, sums the six channels, and adds the 0.1389 ± 0.00026 pb
`[2000, 13000]` tail. The high grid starts at 250 GeV, so the 200–250 GeV
overlap is not double-counted.

| channel | NF low | NF high | NF total | exact low | exact high | exact total | NF/exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gu_zu` | 3026.47 | 435.13 | 3461.60 | 3020.87 | 442.90 | 3463.77 | 0.9994 |
| `gd_zd` | 2638.44 | 348.25 | 2986.69 | 2633.26 | 354.76 | 2988.01 | 0.9996 |
| `gux_zux` | 1040.07 | 111.60 | 1151.68 | 1037.92 | 114.00 | 1151.92 | 0.9998 |
| `gdx_zdx` | 1490.09 | 164.64 | 1654.73 | 1487.20 | 168.08 | 1655.28 | 0.9997 |
| `uux_zg` | 2136.70 | 151.92 | 2288.63 | 2129.85 | 155.18 | 2285.03 | 1.0016 |
| `ddx_zg` | 2085.56 | 136.49 | 2222.05 | 2078.43 | 139.52 | 2217.96 | 1.0018 |
| **total** | **12417.33** | **1348.05** | **13765.38** | **12387.53** | **1374.44** | **13761.97** | **1.00025** |

The "exact" columns are the explicit partonic MadEvent scan convolved with the
same LHAPDF luminosity. The NF grid agrees with that exact convolution at
**0.03%** overall. The table totals are the `[114, 2000]` grid before the
constant `[2000, 13000]` tail; adding the 0.1389 pb tail gives 13765.52 pb,
matching `summary.json`. The low range dominates the integral: 12417 of
13765 pb (90.2%); the high range contributes 1348 pb (9.8%).

### 4.1 Native references

| native run | cuts | cross section |
|---|---|---:|
| fixed scale 91.188 GeV | `p_T,j>20`, `|eta_j|<2.5` | **9147.5 ± 10.32 pb** |
| dynamic scale | `p_T,j>20`, `|eta_j|<2.5` | 9010 ± 10 pb |
| fixed scale, `etaj=5.0` | `p_T,j>20`, `|eta_j|<5.0` | 11668.4 ± 72.03 pb |

The last row is a different acceptance and is not the NF comparison point.
It is still a useful consistency check: the wider `etaj=5.0` native run is
**below** the NF `|eta_j|<2.5` total despite accepting more phase space. The
native side is the outlier relative to the LHAPDF convolution of the exact
partonic input.

The native dynamic-scale slice diagnostic is 645 pb in [0, 120] GeV,
6020 pb in [120, 200] GeV, 2336 pb in [200, 2000] GeV, and 0.139 pb in
[2000, 13000] GeV. These slices are not range-matched to the NF
low [114, 250] / high [250, 2000] split, so use them for shape only; the
range-matched NF/exact comparison is the table above. The machine-readable
summary of the native/NF comparison is
[`native_vs_nf.csv`](results/zj_hadronic_integral/native_vs_nf.csv).

### 4.1.1 Splitting at 200 GeV

Because the native slices naturally split at 200 GeV, the saved NF grid can be
split at 200 without retraining. The NF low side uses the low model over
[114, 200], the NF high side uses the low model over [200, 250] plus the high
model over [250, 2000]. The exact convolution is split the same way.

| region | NF (pb) | exact convolution (pb) | native dynamic slice (pb) |
|---|---:|---:|---:|
| low [114, 200] | 10883.96 | 10858.09 | 6665 [0, 200] |
| high [200, 2000] | 2881.42 | 2909.34 | 2336 [200, 2000] |

The native low [0, 200] is effectively [114, 200] because the `Z + j`
threshold is at 114 GeV. The NF/native ratios are about **1.63 low** and
**1.23 high** (using the dynamic native slices); the total ratio is
about 1.53. The native fixed-scale total is 9147.5 ± 10.32 pb, but a
fixed-scale slice breakdown was not saved. A range-matched native run with
`dsqrt_shat` = 114/250 and 250/2000 is the clean way to get exact fixed-scale
low/high numbers.

The NF split in the table is a post-processing split of the saved 250 GeV
grid: the low model covers 200–250 GeV, and the high model starts at 250 GeV.
It sums exactly to 13765.38 pb. A true integration split at 200 GeV is now
available with `integrate_hadronic_process.py --split 200`; that uses the high
model over 200–250 GeV and may differ slightly from the post-processed table.
The exact split sums to about 13767 pb, 5 pb above the 13761.97 pb total,
because the exact scan has a 0.04 GeV boundary offset at 200 GeV; this is
negligible for the comparison.

### 4.1.2 Native no-rapidity test

To test whether the discrepancy is the missing lab-frame boost in the 1D
\(\sqrt{\hat s}\) factorization, native `p p > z j` was rerun with
`etaj = -1.0` (no jet rapidity cut), keeping `ptj = 20`, `maxjetflavor = 4`,
`nn23lo1`, and the same fixed scale 91.188 GeV. The high-statistics survey
used 100,000 points × 10 iterations (3,600,036 points):

| native run | jet acceptance | scale | cross section |
|---|---|---|---:|
| nominal | `|eta_j| < 2.5` | fixed 91.188 | 9147.5 ± 10.32 pb |
| no rapidity cut | `etaj = -1` | fixed 91.188 | **11915 ± 7.35 pb** |
| wide rapidity | `|eta_j| < 5.0` | dynamic | 11668.4 ± 72.03 pb |
| NF + LHAPDF | `|eta_j| < 2.5` | fixed 91.188 | 13765.5 ± 0.59 pb |

A lower-statistics survey (20,000 points × 8) gave 11924 ± 16.59 pb, consistent
with the high-statistics value. The `refine` step failed on this Windows build
with `max_process not define`; the survey central value is stable between the
two survey sizes.

There is **no validated NF no-rapidity total**. The only cut-removal test was
a low-range `gu_zu` smoke: removing both `p_T,j` and `|eta_j|` cuts collapses
the final ESS to 0.0029 (versus 0.865 with cuts) because the uncut
`g u > z u` integrand is collinearly divergent. A no-eta NF would need a
retrained/regulated target; it is not the same as the native no-eta run.
Removing all cuts has no finite LO cross section.

Removing the rapidity cut increases the native result by 2768 pb (30.2%),
but the NF is still **1850 pb (15.5%) above the no-rapidity native result**.
The NF applies `|eta_j| < 2.5`, so a correct NF should be **below** the
no-rapidity native value; it is above it. Therefore the hypothesis that the
entire discrepancy is the missing lab-frame boost is **not confirmed**.
The no-rapidity test accounts for about 60% of the 4618 pb gap and leaves a
residual ~15% normalization difference.

### 4.2 What the 1.50× discrepancy is and is not

- It is **not** a c/s omission: c/s are included by `pdf_pairs`; b is excluded
  on both sides.
- It is **not** the 200–250 GeV overlap: the high grid starts at 250 GeV.
- It is **not** the partonic NF: all six channels match exact MadEvent at
  0.6% low and 0.3% high.
- It is **not the trapezoid quadrature**: the 60-point grid gives 13765.38 pb;
  subsampling the same grid gives 13674.14 pb at 30 points (0.66% lower),
  13569.92 pb at 20 points (1.4% lower), and 13451.78 pb at 15 points (2.3%
  lower). The independent 50-point exact convolution gives 13761.97 pb.
  A 50% quadrature error is not plausible.
- It is **not strictly the missing lab-frame boost**: the no-rapidity native
  run gives 11915 pb, so removing the cut accounts for 2768 pb (60% of the
  4618 pb gap), but the NF is still 1850 pb (15.5%) above the no-rapidity
  native value. A residual normalization difference remains.
- It is **not an NF-specific integration error**: the exact partonic
  convolution reproduces the NF grid at 0.03%. A shared luminosity/convention
  error is not ruled out by this check.
- It is **between** the LHAPDF convolution (13762 pb) and the native
  `p p > z j` total (9147.5 pb). The nominal PDF set is the same, so the
  remaining candidates are native PDF/luminosity implementation and native
  run-card details; the PDF-matched native run and per-subprocess comparison
  are the decisive tests.

The decisive check is a native run with `pdlabel=lhapdf`, `lhaid=247000`,
the same cuts, and fixed scale 91.188 GeV, followed by a per-subprocess
comparison with `results/zj_hadronic_madevent_interpolation/<channel>/
comparison.csv`. Until that is done, **do not quote the NF total as a
physics cross section**.

## 5. Low/high ESS

There are two different ESS numbers and they answer different questions.

| metric | low range | high range |
|---|---|---:|
| training / held-out ESS | 0.775–0.963 (mean 0.900) | 0.231–0.313 (mean 0.285) |
| integration-grid ESS | mean 0.962; min 0.467 at 114 GeV | mean 0.972; min 0.735 |

Per channel:

| channel | low training | high training | low integration | high integration |
|---|---:|---:|---:|---:|
| `gu_zu` | 0.9592 | 0.3125 | 0.9623 | 0.9685 |
| `gd_zd` | 0.9614 | 0.3126 | 0.9642 | 0.9700 |
| `gux_zux` | 0.9613 | 0.3126 | 0.9639 | 0.9798 |
| `gdx_zdx` | 0.9628 | 0.3114 | 0.9680 | 0.9700 |
| `uux_zg` | 0.7755 | 0.2306 | 0.9578 | 0.9715 |
| `ddx_zg` | 0.7829 | 0.2309 | 0.9585 | 0.9713 |

- **Training/held-out ESS is lower in the high range** by about a factor of
  3. This is the ESS shown in the model summaries and the original loss/ESS
  plots.
- **Integration ESS is high in both ranges.** The low range is slightly lower
  only because of the 114 GeV threshold point; the rest of the low grid is
  ≈0.96–0.99. The high grid is ≈0.97. The high range contributes only ~10%
  of the integral, so its lower training ESS does not dominate the integral
  error.
- The per-channel `ess_mean` in `summary.json` (≈0.965) is the **integration**
  ESS averaged over both grid ranges; it does not show the training-ESS drop.

### Why is the high-range training ESS so low?

The training/held-out ESS is a **global ESS over uniformly sampled
conditions**, not a per-condition ESS. `evaluate_ess()` draws a condition
\(z\sim U(0,1)\) and then concatenates the weights from all sampled
conditions before computing

\[
\mathrm{ESS}=\frac{(\sum_i w_i)^2}{N\sum_i w_i^2}.
\]

In the high range, \(z\) maps to \(\sqrt{\hat s}\in[200,2000]\) uniformly.
The target normalization changes strongly over that factor-10 range, so the
combined weights have a large spread across conditions. That global ESS is
0.23–0.31, and the corresponding global `std_logw` is 1.13–1.22. The low
range spans only 114–250 GeV (factor 2.2), so its global ESS is 0.78–0.96 and
`std_logw` is 0.15–0.46.

The integration, by contrast, conditions on **one fixed \(\sqrt{\hat s}\) at
a time**. At each grid point the proposal is compared only with the target at
that energy, not with a mixture of all conditions. The per-condition ESS is
therefore high: 0.96–0.99 over most of the high grid (mean 0.972), with a
minimum of 0.735 for `gd_zd` near 278 GeV. This is why the high model has a
low global/training ESS but a high integration ESS.

The low global ESS does not mean the high model is bad for the conditional
integral. It would matter if you sampled the high model jointly over
\(\sqrt{\hat s}\) without conditioning, for example in a direct hadronic
generation. For the quadrature, the per-condition ESS is the relevant
metric. The high `q q~` channels (`uux_zg`, `ddx_zg`) have the lowest global
ESS (0.2306/0.2309) because their condition dependence is stronger; their
per-condition integration ESS is still ≈0.97.

The corrected summary figure is
`results/zj_hadronic_integral/plots/ess_summary.png`; the data are in
`results/zj_hadronic_integral/ess_summary.csv`.

## 6. Plots

### Corrected interpolation plots

`plots/interpolation_low.png` and `plots/interpolation_high.png` now show:

- the **six-channel summed** partonic cross section, not a single channel;
- the NF curve and the exact MadEvent points;
- a lower **NF/MadEvent ratio panel** with a line at 1.0.

The low-range **summed** ratio is within 0.996–1.001 (per-channel
0.994–1.006); the high-range summed ratio is within 0.998–1.003
(per-channel 0.9996–1.0034). The exact `comparison.csv` contains both the
50-point low scan [114, 250] and the 50-point high scan [200, 2000]. The high
plot now selects the scan matching the NF grid; previously it also included
the dense low-scan tail 200–250 GeV, which produced the visible cluster at
200 GeV. The high scan itself is equally spaced at 36.73 GeV. The plots are
partonic; they do not validate the hadronic normalization by themselves.

### ESS summary

`plots/ess_summary.png` has two panels:

- **training/held-out ESS**: high range visibly lower;
- **integration ESS**: both ranges high, with the low 114 GeV point the only
  low outlier.

### Legacy trajectory plots

`plots/loss_ess_low.png` and `plots/loss_ess_high.png` are the original
training-trajectory plots. The raw server logs are not archived in this
checkout, so they could not be regenerated locally. The plotting code now
uses **color = channel, line style = stage**, has separate stage and channel
legends, and labels the final ESS as held-out; a future run with the server
logs will produce corrected versions. The report uses `ess_summary.png`.

## 7. What to quote

| quantity | value | status |
|---|---:|---|
| Native MadEvent, fixed scale 91.188 GeV | **9147.5 ± 10.32 pb** | **physics reference for the native setup** |
| Native MadEvent, dynamic scale | 9010 ± 10 pb | alternative scale choice |
| Native `etaj=5.0` | 11668.4 ± 72.03 pb | different acceptance; consistency only |
| NF partonic vs MadEvent | 0.994–1.006 (low), 0.9996–1.0034 (high) | **valid NF validation** |
| NF grid vs exact partonic convolution | 13765.38 vs 13761.97 pb (1.00025) | **valid internal check** |
| NF hadronic total with LHAPDF | 13765.5 ± 0.59 pb | not a production cross section until native reconciliation |

Recommended wording for the report:

> The conditional NF reproduces all six partonic `pp -> Z + j` subprocesses at
> the 0.6% level or better. Convolving the exact partonic MadEvent scan with
> LHAPDF gives 13761.97 pb, in agreement with the NF grid at 0.03%. The
> native fixed-scale MadEvent total is 9147.5 ± 10.32 pb, 1.50× lower than
> the LHAPDF convolution; this native-vs-convolution discrepancy is under
> investigation and is not attributed to the NF or to missing c/s. The
> training/held-out ESS is lower in the high range (0.23–0.31) than in the
> low range (0.78–0.96), while the integration ESS is high in both.

The 0.59 pb NF error is statistical only. It is not comparable to the
10.32 pb native integration error, and it does not cover the 1.50×
normalization discrepancy.

### 7.1 What “number of samples” means

| row | count | what it counts |
|---|---:|---|
| MadGraph native | 1,360,036 | VEGAS phase-space points used for the direct `p p > z j` integration; each point evaluates the matrix element and PDFs |
| NF training | 3,838,650,240 | training sample budget: cache builds, cached KL batches, cached chi-square batches, final ESS; **not** all unique matrix-element calls and **not** independent integration samples |
| NF generation | 72,000,000 | 6 channels × (60 low + 60 high) grid points × 100,000 conditional-NF samples per point; the hadronic integral is assembled by quadrature from these 720 point estimates |

The training count is built from the per-model budget:

```text
10 caches × 131,072
+ 8,800 KL steps × 30,000
+ n_chi2 steps × 16,384
+ 100,000 final ESS
```

The 10 is the number of `WEIGHTED CACHE:` records in each server log. The
benchmark uses 131,072 as the representative balanced cache size
(256 conditions × 512 points); it does not parse each cache's actual
`conditions`/`points-per-condition` lines, so the total is a budget estimate,
not an exact unique-evaluation count.

For example, low `gu_zu` has 3,600 chi-square steps:

```text
10 × 131,072 + 8,800 × 30,000 + 3,600 × 16,384 + 100,000
= 324,393,120 evaluations
```

The 12-model sum is 3,838,650,240. The cache builds are actual target
evaluations; the KL and chi-square “batches” are drawn from weighted caches
and reuse points, so the 3.84B figure is a training **work/budget** count, not
3.84B unique matrix-element calls. None of it enters the final integral's
error formula; that error comes from the 72M generation samples.

The generation count is **not** 72M independent samples of the hadronic
integral. It is 720 separate conditional estimates, each with 100,000
samples. The per-point estimates are independent, and the quadrature combines
them.

### 7.2 How the NF error is calculated

At one grid point, with \(N=100{,}000\) samples, the flow (plus the defensive
uniform mixture) proposes \(x_i\sim q(x_i\mid\sqrt{\hat s})\), and the
importance weight is

\[
w_i=\frac{f(x_i)}{q(x_i\mid\sqrt{\hat s})},
\qquad
\hat I=\frac1N\sum_{i=1}^N w_i.
\]

The code uses the log-weight form in
[`nf_estimate()`](experiments/validate_madgraph_process_integral.py#L114):

```python
logw = logf - logq
estimate = exp(logsumexp(logw) - log(N))
second   = exp(logsumexp(2 * logw) - log(N))
variance = max(0, second - estimate**2)
stderr   = sqrt(variance / N)
ess      = exp(2*logsumexp(logw) - log(N) - logsumexp(2*logw))
```

So the per-point standard error is the usual importance-sampling error,

\[
\delta\hat I=\sqrt{\frac{1}{N}\left(\frac1N\sum_i w_i^2-\hat I^2\right)}.
\]

The normalized effective sample size is

\[
\mathrm{ESS}=\frac{\left(\sum_i w_i\right)^2}{N\sum_i w_i^2},
\qquad
N_{\rm eff}=\mathrm{ESS}\times N.
\]

`std_logw` is the standard deviation of the finite \(\log w_i\) values;
`max_weight_over_mean` and `cut_pass_fraction` are the other weight
diagnostics in `nf_estimate()`.

The grid integration multiplies each point by the trapezoid weight
\(\Delta_j\) and the LHAPDF luminosity \(L_j\):

\[
\sigma_c=\sum_j \Delta_j L_j \hat\sigma_{cj},
\qquad
\delta\sigma_c=
\sqrt{\sum_j \left(\Delta_j L_j \delta\hat\sigma_{cj}\right)^2}.
\]

This is [`trapz_with_error()`](experiments/integrate_hadronic_process.py#L42).
The channel errors are added in quadrature:

\[
\delta\sigma_{\rm tot}
=
\sqrt{\sum_c \delta\sigma_c^2+\delta\sigma_{\rm tail}^2},
\]

with the tail error 0.00026 pb. The quoted **0.5931 pb** is this combined
statistical/quadrature error. It assumes the per-point estimates are
independent; it does not include model bias, grid systematics, PDF/scale
variation, or the native-vs-convolution normalization.

The native **10.32 pb** is MadGraph's VEGAS integration error from
`results.dat`, not a per-point NF error. It is a different estimator on a
different central value.

### 7.3 Error budget

| source | size | included in 0.5931 pb? |
|---|---:|---|
| per-point NF statistics + trapezoid | 0.5931 pb (4.3 × 10⁻⁵ relative) | yes |
| grid/quadrature, exact scan vs NF | 3.4 pb (0.025%) | no |
| grid coarsening, 60 → 30 points | 0.66% | no |
| partonic model vs MadEvent | 0.6% low, 0.3% high | no |
| tail | 0.00026 pb | yes, in quadrature |
| native-vs-convolution normalization | 1.50× (~4600 pb) | no |
| PDF/scale variation | not computed | no |

The statistical error is tiny compared with the normalization discrepancy.
The exact convolution difference of 3.4 pb (0.025%) is larger than the quoted
0.5931 pb, which is why the 0.5931 pb must not be presented as the total
uncertainty. The 1.50× native comparison is the dominant unresolved
systematic.

## 8. Reproduction

Local LHAPDF environment:

```bash
source /c/Users/laiji/test/lhapdf-local/env_lhapdf.sh
PY="C:/Users/laiji/.conda/envs/workspace/python.exe"
```

Regenerate the corrected plots:

```bash
"$PY" scripts/plot_nf_diagnostics.py
```

This writes `ess_summary.png/pdf`, `ess_summary.csv`, and the two
interpolation plots. The script prints "missing log" for the server-side
training logs if they are absent; the ESS summary and interpolation plots
still regenerate from the saved CSVs.

Run the exact-convolution audit:

```bash
"$PY" scripts/audit_hadronic_normalization.py
```

It writes `results/zj_hadronic_integral/normalization_audit.csv` and prints
the per-channel and total NF/exact ratios.

The full integration command is in `RUNBOOK_ZJ_HADRONIC.md` §5. The native
fixed-scale run-card settings are in §8 of that runbook.

## 9. Data inventory and caveats

- `results/zj_hadronic_{low,high}_<channel>_copula_balanced_gpu/`:
  `*_final_model.pt` (506 KB), `*_final_summary.csv`, `*_history.csv`.
- `results/zj_hadronic_integral/`: `summary.json`, `per_channel.csv`,
  `grid_*.csv`, `interpolation_{low,high}.csv`, `normalization_audit.csv`,
  `native_vs_nf.csv`, `ess_summary.csv`, `plots/`.
- `results/zj_hadronic_madevent_interpolation/<channel>/comparison.csv`:
  exact partonic scan.
- `madgraph_baselines/` (outside the ML repo): native Z+1 and Z+2 jet
  baselines; the 11668.4 pb run uses `etaj=5.0`.
- The 12 `.pt` checkpoints are **local and gitignored**. A fresh GitHub clone
  will not contain them; copy them or retrain.
- The `*_history.csv` files are empty and the training logs are not in this
  checkout. The final summaries, grid CSVs, and interpolation CSVs are
  sufficient for this report; the trajectory plots require the server logs.
  `scripts/plot_nf_diagnostics.py` now writes
  `training_history_{low,high}.csv` when those logs are supplied, so the
  time/KL/ESS series can be recovered from the server copy of
  `/storage/users/dlai/mango-linux/logs/`.
- The `norm` denominator/`0.59 pb` is statistical only; no PDF, scale, or
  native-reconciliation systematic is included.

## 10. Next steps

1. **Native PDF-matched run:** `p p > z j` with `pdlabel=lhapdf`,
   `lhaid=247000`, fixed scale 91.188 GeV, `ptj=20`, `etaj=2.5`,
   `maxjetflavor=4`. If it gives ≈13762 pb, the native internal PDF grid was
   the outlier; if it gives ≈9147 pb, the LHAPDF convolution/native setup
   needs further audit.
   An attempt on the local Windows build in
   `madgraph_baselines/z_1jet_noeta` rebuilt `Source/PDF` with the local
   LHAPDF 6.5.6 and was recognized as `Using LHAPDF v6.5.6 interface`, but
   the generated Fortran process aborted at runtime with
   `LHAPDF::ReadError: Could not find any pdfsets.index files` even with
   `LHAPDF_DATA_PATH` set and `pdfsets.index` present in both the share and
   local `lib/PDFsets` directories. The same static library works in a
   standalone C++ test and in the Python shim, so this is a Windows
   generated-MadEvent/LHAPDF runtime path issue. No native LHAPDF cross
   section was obtained; this test should be run on the Linux server or after
   fixing the Windows runtime path resolution.
2. **Per-subprocess comparison:** compare each native `P1_*` rate with the
   exact `comparison.csv` convolution. This localizes any channel-specific
   difference.
3. **Optional:** retrain the high-range models only if high-region training
   ESS is itself a deliverable. The current high models are usable for the
   integral; the integration ESS is high.
