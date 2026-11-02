# Z + jet diagnostic plots

## Current report figures

- `ess_summary.png` / `.pdf` — training/held-out ESS and integration-grid ESS,
  split by low and high `sqrt(s-hat)` range. Generated from the 12 saved
  `*_final_summary.csv` files and `grid_*.csv`; see
  `../ess_summary.csv`.
- `interpolation_low.png` / `.pdf` — six-channel summed partonic cross
  section, low range, with a lower NF/MadEvent ratio panel.
- `interpolation_high.png` / `.pdf` — high-range version.

## Legacy figures

- `loss_ess_low.png` / `.pdf` and `loss_ess_high.png` / `.pdf` are the
  original training-trajectory plots. Their data are correct, but the
  subtitles and line-style legends mixed channel identity with training stage
  and are misleading. The raw training logs are not archived in this checkout,
  so they could not be regenerated locally.

`scripts/plot_nf_diagnostics.py` now draws the trajectory plots with
**color = channel and line style = stage**, with separate stage and channel
legends, for a future run that has the server logs. Use `ess_summary.png` for
the current report.
