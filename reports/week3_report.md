# Week 3 Simulation Report

**Period end:** 2004-12-29

## Summary Metrics
- Average RMSE: **4641.1052**
- Average MAE: **4641.1052**
- Average MAPE: **17.88%**

## Performance by Group
```
               rmse        mae     mape
group                                  
A_lstm    2979.0905  2979.0905  13.1746
B_lstm    6440.1467  6440.1467  22.2173
C_xgb     6766.4556  6766.4556  26.5476
D_xgb      253.3777   253.3777   0.9244
E_static  6766.4556  6766.4556  26.5476
```

## RMSE Trend
Chart saved to `C:\Users\thesa\Downloads\hi\reports\charts\week3_rmse_trend.png`

## Observations
- 2550 forecast evaluations logged through week 3.
- Best group (lowest RMSE): **D_xgb**
- Highest RMSE group: **C_xgb**

## Commentary
Continuous-learning groups (B, D) are compared against periodic retraining (A, C) and the static baseline (E). RMSE trends indicate whether daily adaptation outperforms weekly full retrains during this replay window.