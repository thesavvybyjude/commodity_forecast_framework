# Week 2 Simulation Report

**Period end:** 2004-12-23

## Summary Metrics
- Average RMSE: **4859.2349**
- Average MAE: **4859.2349**
- Average MAPE: **18.62%**

## Performance by Group
```
               rmse        mae     mape
group                                  
A_lstm    2994.4527  2994.4527  13.2225
B_lstm    7475.4201  7475.4201  25.7265
C_xgb     6772.0259  6772.0259  26.5531
D_xgb      282.2500   282.2500   1.0373
E_static  6772.0259  6772.0259  26.5531
```

## RMSE Trend
Chart saved to `C:\Users\thesa\Downloads\hi\reports\charts\week2_rmse_trend.png`

## Observations
- 2100 forecast evaluations logged through week 2.
- Best group (lowest RMSE): **D_xgb**
- Highest RMSE group: **B_lstm**

## Commentary
Continuous-learning groups (B, D) are compared against periodic retraining (A, C) and the static baseline (E). RMSE trends indicate whether daily adaptation outperforms weekly full retrains during this replay window.