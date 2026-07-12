# Week 1 Simulation Report

**Period end:** 2004-12-16

## Summary Metrics
- Average RMSE: **5270.5227**
- Average MAE: **5270.5227**
- Average MAPE: **20.24%**

## Performance by Group
```
               rmse        mae     mape
group                                  
A_lstm    3018.8725  3018.8725  13.2620
B_lstm    9357.1913  9357.1913  33.2711
C_xgb     6790.2615  6790.2615  26.5611
D_xgb      396.0268   396.0268   1.5203
E_static  6790.2615  6790.2615  26.5611
```

## RMSE Trend
Chart saved to `C:\Users\thesa\Downloads\hi\reports\charts\week1_rmse_trend.png`

## Observations
- 1050 forecast evaluations logged through week 1.
- Best group (lowest RMSE): **D_xgb**
- Highest RMSE group: **B_lstm**

## Commentary
Continuous-learning groups (B, D) are compared against periodic retraining (A, C) and the static baseline (E). RMSE trends indicate whether daily adaptation outperforms weekly full retrains during this replay window.