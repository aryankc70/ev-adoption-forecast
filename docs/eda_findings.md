# EDA Findings — EV Adoption Dataset

Source: IEA Global EV Outlook 2026, `data/raw/ev_clean.parquet` (see `load_manifest.json` for lineage).

## Data coverage

- **61 countries** have Cars EV sales share data (Historical).
- **44 countries** have 2/3-wheeler EV sales share data (Historical).
- All 44 two/three-wheeler countries are a strict subset of the 61 car countries — no country reports wheelers without also reporting cars.
- 17 countries report Cars data only, with no two/three-wheeler coverage: Bulgaria, Croatia, Cyprus, Czech Republic, Estonia, Hungary, Ireland, Jordan, Latvia, Lithuania, Luxembourg, Nepal, Romania, Seychelles, Slovakia, Slovenia, Uzbekistan.
- **Implication for modeling:** the two-wheeler model will need to run on a smaller country set (44) than the car model (61). This should be documented in the final model card, not treated as a bug.

## Adoption curve shape

- Car EV sales share broadly follows an **S-curve (logistic diffusion)** pattern, consistent with classic technology adoption theory (Bass diffusion).
- Norway is the clearest example: near-zero through 2013, steep acceleration 2014–2020, now approaching saturation (~97% in 2025).
- Countries reach their "takeoff point" at very different times — China's steep climb didn't begin until ~2020, nearly a decade after Norway's.

## Non-monotonic behavior (important — breaks the pure S-curve assumption)

Several countries show **real dips**, not noise, driven by policy changes:
- **Germany**: Cars EV sales share fell from ~31% (2022) to ~20% (2024), coinciding with the government ending its EV purchase subsidy at the end of 2023.
- **China**: 2/3-wheeler EV sales share fell from ~55% (2021) to ~43% (2022).

**Implication for modeling:** a pure smooth logistic/Bass curve cannot represent these dips. This motivates including policy/covariate features (e.g. subsidy status, a binary "incentive active" flag) in Phase 4 feature engineering, rather than relying on time-only curve fitting.

## Notable outlier: Nepal

Nepal's Cars EV sales share rose from ~10% (2021) to ~74% (2024) — one of the fastest adoption accelerations in the dataset, driven by import tax policy favoring EVs and cheap domestic hydropower. Nepal has no 2/3-wheeler data in this dataset (Cars-only country).

## Modeling implications

1. Fit per-country models rather than a single global curve — takeoff timing and curve steepness vary too much to pool naively.
2. Include policy/incentive covariates where available, since multiple countries show non-monotonic, policy-driven share changes that a pure time-based S-curve cannot capture.
3. Two-wheeler and Cars models must be trained/evaluated separately, with wheelers limited to the 44-country subset.

## Outlier / sanity scan

- No rows exceed 100% sales share (0 invalid rows found) — clean bound on the target variable.
- One large year-over-year jump found: Iceland 2/3-wheelers EV sales share rose from ~11% (2019) to 55% (2020).
  - Confirmed as a genuine small-market effect, not a data error: absolute EV 2/3-wheeler sales in Iceland were only 25 units (2019) → 300 units (2020) — a real but low-volume spike.
  - Iceland also has missing years (2013–2014) in this series.
- **Implication for modeling:** small-market countries can show high variance in share terms purely from small denominators. Consider either (a) weighting country loss by market size during training/evaluation, or (b) flagging low-volume countries (e.g. <1,000 annual units) separately rather than treating their curves with equal confidence to large markets like China or Germany.
