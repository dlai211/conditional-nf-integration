# Hadronic `p p > z j` performance benchmark

Fixed scale 91.188 GeV, LHAPDF `NNPDF23_lo_as_0130_qed` (ID 247000, the
set behind MG5's `nn23lo1`), alpha_s = 0.130, pT,j > 20 GeV, |eta_j| < 2.5.
The default dynamic-scale MadEvent total is 9010 +/- 10 pb; the row below
is the fixed-scale run used consistently by both the native and NF sides.

| Method | Compute Time | Number of Samples | Cross Section (pb) | Error (pb) |
|---|---:|---:|---:|---:|
| MadGraph Native | 39.4 s | 1,360,036 | 9147.5 | 10.32 |
| NF Training | 4.30 h | 3,838,650,240 | 13765.5 | 0.5931 |
| NF Generation | 12.7 min | 72,000,000 | 13765.5 | 0.5931 |

Training campaign wall time with four concurrent 48-CPU jobs: 1.19 h.  The NF training samples count the
matrix-element evaluations at the weighted-cache builds, the KL batches
(30,000 points/step), the chi-square batches (16,384
points/step) and the final 100,000-sample ESS estimate.

The NF cross section is one physical result; the training row reports the
cost to produce the twelve models and the generation row reports the
amortised inference cost.  The native reference is the 9148 pb fixed-scale
run; the 9010 pb dynamic-scale run is quoted above for comparison.

| range | channel | KL steps | chi2 steps | caches | training samples | time |
|---|---|---:|---:|---:|---:|---:|
| high | gu_zu | 8800 | 3000 | 10 | 314,562,720 | 21.4 min |
| high | gd_zd | 8800 | 3600 | 10 | 324,393,120 | 23.8 min |
| high | gux_zux | 8800 | 3000 | 10 | 314,562,720 | 18.9 min |
| high | gdx_zdx | 8800 | 3300 | 10 | 319,477,920 | 20.6 min |
| high | uux_zg | 8800 | 3600 | 10 | 324,393,120 | 21.1 min |
| high | ddx_zg | 8800 | 3600 | 10 | 324,393,120 | 21.4 min |
| low | gu_zu | 8800 | 3600 | 10 | 324,393,120 | 21.9 min |
| low | gd_zd | 8800 | 3600 | 10 | 324,393,120 | 22.1 min |
| low | gux_zux | 8800 | 2700 | 10 | 309,647,520 | 22.3 min |
| low | gdx_zdx | 8800 | 2700 | 10 | 309,647,520 | 23.4 min |
| low | uux_zg | 8800 | 3600 | 10 | 324,393,120 | 19.6 min |
| low | ddx_zg | 8800 | 3600 | 10 | 324,393,120 | 21.2 min |

