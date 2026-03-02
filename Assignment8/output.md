================================================================================
FETCHING STOCK DATA: AAPL
================================================================================
Data shape: (1258, 7)
Date range: 2019-01-02 00:00:00-05:00 to 2023-12-29 00:00:00-05:00
Columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

================================================================================
PREPROCESSING DATA
================================================================================
Missing values before: 0
Missing values after: 0

Final data shape: (1238, 15)
Date range: 2019-01-31 00:00:00-05:00 to 2023-12-29 00:00:00-05:00
Features created: ['Adj_Close', 'Daily_Return', 'Return_Lag1', 'MA_5', 'MA_20', 'Price_vs_MA5', 'Price_vs_MA20', 'Volatility_5']

================================================================================
TRAINING REGRESSION MODEL
================================================================================
Training samples: 990
Testing samples: 248

--- Model Performance ---
Train RMSE: 0.009751
Test RMSE: 0.006051
Train MAE: 0.007283
Test MAE: 0.004645

--- Feature Coefficients ---
Lag_Return: -0.563140
Price_vs_MA5: 1.091619
Price_vs_MA20: -0.077161
Volatility_5: 0.032134

================================================================================
COMPUTING FINANCIAL METRICS
================================================================================
Total return (test period): 0.5322
Annualized return: 0.5428

--- Sharpe Ratio ---
Mean daily return: 0.001801
Daily risk-free rate: 0.000079
Excess daily return: 0.001721
Std daily return: 0.012584
Sharpe Ratio (annualized): 2.1715

--- Sortino Ratio ---
Downside returns count: 108
Downside std: 0.007733
Sortino Ratio (annualized): 3.5335

================================================================================
CREATING VISUALIZATIONS
================================================================================
Saved: price_history.png
Saved: returns_analysis.png
Saved: prediction_analysis.png
Saved: metrics_summary.png

Saved: Assignment8/processed_stock_data.csv

================================================================================
ANALYSIS COMPLETE
================================================================================

--- Model Performance Summary ---
Train RMSE: 0.009751
Test RMSE: 0.006051
Train MAE: 0.007283
Test MAE: 0.004645

--- Financial Metrics Summary ---
Annual Return: 0.5428 (54.28%)
Sharpe Ratio: 2.1715
Sortino Ratio: 3.5335