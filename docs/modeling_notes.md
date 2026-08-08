# Modeling Notes

## Baseline: Logistic Diffusion Curve

Fit per (country, mode) on training years (<=2021), evaluated on test years (2022-2025).

**Convergence:** 91/101 (country, mode) groups converged. The 10 failures were all due to
having fewer than 4 training data points (e.g. Nepal, whose EV boom only began in 2021-2022 --
consistent with EDA findings).

**Aggregate performance:**
- Mean MAPE: 135.3% -- heavily skewed by a small number of outlier countries
- Median MAPE: 38.6% -- a much more representative picture of typical performance
- Median RMSE: 4.8 percentage points

**Key limitation found: curve extrapolation fails on trend reversals.**

Costa Rica (2/3-wheelers) is the clearest example: training data (2014-2021) showed accelerating
growth (0.05% -> 5.5%), so the fitted logistic curve reasonably extrapolated continued growth
toward saturation. In reality, actual share *declined* throughout the test period (2022: 0.58% ->
2025: 0.11%), likely due to an incentive program ending -- a pattern also seen in Germany's Cars
data in EDA. The fitted curve predicted an 82% share by 2025; actual was 0.11%.

This is a structural limitation of pure time-based curve fitting, not a bug: the model has no way
to detect a trend reversal that occurs entirely within the held-out test period, since it only
ever sees `share(t)`, with no signal about *why* growth is happening or whether it will continue.

**This directly motivates testing gradient-boosted models next** (LightGBM / XGBoost / sklearn
GBM), which can use lag and year-over-year change features (`ev_sales_share_yoy_change`) to
potentially detect deceleration signals the logistic curve structurally cannot see.

**Always report both mean and median error when evaluating per-group models** -- the ~4x gap
between mean and median MAPE here shows how a handful of poorly-identified fits (sparse data,
parameters landing on optimizer bounds) can make the aggregate mean deeply misleading.
