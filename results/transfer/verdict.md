# Transfer verdict: astlingen -> singapore

Written 2026-10-14T12:06:01 by `python -m sitetransfer verdict`. Rules: code_plan/SINGAPORE_TRANSFER_PLAN.md, sections 8.4 and 8.5. Bands: resampled at both sites.

**Recipe verdict: DOES NOT TRANSFER.** families B, C pass at home but not here.

| Family | v14 at astlingen | v14 at singapore |
| :--- | :---: | :---: |
| A | pass (3/3) | pass (2/3) |
| B | pass (4/7) | fail (3/7) |
| C | pass (3/6) | fail (1/6) |
| D (predicted, not required) | fail (0/6) | pass (4/6) |
| E_monthly | pass (1/1) | pass (1/1) |
| E_daily (predicted, not required) | pass (1/2) | fail (0/2) |

## Relative standing against the latent Gaussian AR (families B and C)

- astlingen: B v14 4 vs AR 5; C v14 3 vs AR 4
- singapore: B v14 3 vs AR 5; C v14 1 vs AR 4

## Window horizon (family D against the perfect 5h20 generator)

- astlingen long_storms_yr: v14 1.4; perfect 5h20 window 1.9-2.8; real 14.2
- astlingen rain_in_320min_pct: v14 24.6; perfect 5h20 window 25-30.6; real 64.7
- astlingen var_scaling_rmse: v14 0.171; perfect 5h20 window 0.13-0.153; real 0
- singapore long_storms_yr: v14 0.1; perfect 5h20 window 0.1-0.2; real 1.8
- singapore rain_in_320min_pct: v14 12.9; perfect 5h20 window 14.6-15.6; real 40.8
- singapore var_scaling_rmse: v14 0.145; perfect 5h20 window 0.0759-0.117; real 0
