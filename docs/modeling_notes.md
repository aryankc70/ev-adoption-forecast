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


## GBM Candidates: LightGBM, XGBoost, sklearn GradientBoostingRegressor

### Critical process note: initial leakage bug, caught and fixed

The first version of this comparison used lag features (`ev_sales_share_lag1`,
`ev_sales_share_yoy_change`) computed globally before the train/test split. This meant
a 2023 test row's "lag1" feature was the *real actual* 2022 value -- itself inside the
held-out test period, information that would not exist yet at genuine forecast time.
This produced misleadingly excellent results (mean MAPE ~10%, vs baseline's ~135%) that
did not reflect real forecasting ability.

**Fix:** rebuilt as recursive multi-step forecasting (`recursive_forecast.py`). Starting
from each country/mode's last known training-period state, the model predicts one year
at a time, and each step's own prediction becomes the next step's lag input -- exactly
matching real deployment, where future actuals are never available in advance.

### Results (recursive, leak-free), test years 2022-2025

| Model | Mean MAPE | Median MAPE | RMSE | MAE |
|---|---|---|---|---|
| Logistic baseline | 135.3% | 38.6% | 4.8 (median) | -- |
| LightGBM | 50.5% | 35.2% | 10.1 | 6.2 |
| XGBoost | 50.3% | 35.9% | 10.4 | 6.2 |
| sklearn GBM | 60.9% | 37.7% | 12.5 | 7.5 |

All three GBMs post a slightly better median MAPE than the baseline, but a notably worse
RMSE. Investigating individual worst-case errors explains why.

### Key limitation found: tree-based models cannot extrapolate beyond the training target range

Norway (Cars) is the clearest example: actual share continues climbing 90% -> 97%
through 2023-2025, but LightGBM's prediction plateaus around 45-49% and barely moves
year to year. This is a structural property of tree-based models, not a tuning issue:
each leaf outputs the average of training examples that landed there, so the model
cannot output a value higher than what it saw during training. Since Norway's training
window (<=2021) topped out around 86%, the model has no basis to predict 90%+.

This compounds under recursive forecasting: an underpredicted 2022 value becomes the
lag input for 2023, so the error doesn't just persist -- it stalls the whole trajectory,
producing the flat plateau observed.

**This is the opposite failure mode from the baseline.** The logistic curve extrapolates
freely (by construction, it can reach up to its fitted saturation value) but cannot
detect trend reversals (Costa Rica). The GBMs can react to short-term feature signals
but cannot extrapolate past observed ranges (Norway). Neither model family is
strictly better -- they fail on different, roughly opposite classes of countries.

### Conclusion

LightGBM is the best of the three GBM candidates on both MAPE and RMSE, and is
selected as the ML candidate to proceed with (Step 5.14+: hyperparameter tuning).
However, its current recursive-forecast weakness on fast-growing, near-saturation
countries is a known, documented limitation -- not silently ignored -- and a
candidate for future improvement (e.g. modeling growth *rate* rather than absolute
share level, or blending with the logistic baseline for countries already past takeoff).
