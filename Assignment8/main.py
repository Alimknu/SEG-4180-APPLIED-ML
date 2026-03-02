import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
STOCK_SYMBOL = 'AAPL'  # Apple Inc.
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
TEST_SIZE = 0.2
RANDOM_STATE = 42
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# -------------------- Data Collection --------------------
def fetch_stock_data(symbol, start, end):
    print("=" * 80)
    print(f"FETCHING STOCK DATA: {symbol}")
    print("=" * 80)
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {data.columns.tolist()}")
        
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# -------------------- Data Preprocessing --------------------
def preprocess_stock_data(data):
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)
    
    # Make a copy to avoid warnings
    df = data.copy()
    
    # Check for missing values
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    # Forward fill missing values (for holidays/weekends)
    df = df.fillna(method='ffill')
    
    # Drop any remaining NaN rows
    df = df.dropna()
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after: {missing_after}")
    
    # Use adjusted close for returns (accounts for splits/dividends)
    df['Adj_Close'] = df['Close']  # yfinance already provides Adjusted Close as 'Close' when using auto_adjust
    
    # Calculate daily returns
    df['Daily_Return'] = df['Adj_Close'].pct_change()
    
    # Remove the first row which has NaN return
    df = df.dropna()
    
    # Create features for regression
    # Lagged returns (previous day's return)
    df['Return_Lag1'] = df['Daily_Return'].shift(1)
    
    # Moving averages
    df['MA_5'] = df['Adj_Close'].rolling(window=5).mean()
    df['MA_20'] = df['Adj_Close'].rolling(window=20).mean()
    
    # Price relative to moving average
    df['Price_vs_MA5'] = df['Adj_Close'] / df['MA_5'] - 1
    df['Price_vs_MA20'] = df['Adj_Close'] / df['MA_20'] - 1
    
    # Volatility (rolling std of returns)
    df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    
    # Drop rows with NaN from feature creation
    df = df.dropna()
    
    print(f"\nFinal data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Features created: {[col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]}")
    
    return df

# -------------------- Model Building --------------------
def prepare_features(df):
    # Features for prediction
    feature_cols = ['Return_Lag1', 'Price_vs_MA5', 'Price_vs_MA20', 'Volatility_5']
    
    X = df[feature_cols].values
    y = df['Daily_Return'].values  # Predicting next day's return (shifted in time split)
    
    # Note: For time series, we'll split chronologically, not randomly
    return X, y, feature_cols

def train_regression_model(X, y, test_size=TEST_SIZE):
    print("\n" + "=" * 80)
    print("TRAINING REGRESSION MODEL")
    print("=" * 80)
    
    # For time series, split chronologically (not randomly)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n--- Model Performance ---")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Feature importance (coefficients)
    feature_names = ['Lag_Return', 'Price_vs_MA5', 'Price_vs_MA20', 'Volatility_5']
    print(f"\n--- Feature Coefficients ---")
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.6f}")
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'coef': model.coef_,
        'intercept': model.intercept_
    }
    
    return model, X_train, X_test, y_train, y_test, metrics

# -------------------- Financial Metrics --------------------
def calculate_financial_metrics(returns, risk_free_rate=RISK_FREE_RATE):
    print("\n" + "=" * 80)
    print("COMPUTING FINANCIAL METRICS")
    print("=" * 80)
    
    # Convert to Series if needed
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    # 1. Annual Return
    # Assuming 252 trading days per year
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    print(f"Total return (test period): {total_return:.4f}")
    print(f"Annualized return: {annual_return:.4f}")
    
    # 2. Sharpe Ratio
    # Sharpe = (mean return - risk_free_rate) / std(returns)
    mean_daily_return = returns.mean()
    daily_risk_free = risk_free_rate / 252
    
    excess_daily_return = mean_daily_return - daily_risk_free
    std_daily_return = returns.std()
    
    # Annualize
    sharpe_ratio = (excess_daily_return / std_daily_return) * np.sqrt(252)
    
    print(f"\n--- Sharpe Ratio ---")
    print(f"Mean daily return: {mean_daily_return:.6f}")
    print(f"Daily risk-free rate: {daily_risk_free:.6f}")
    print(f"Excess daily return: {excess_daily_return:.6f}")
    print(f"Std daily return: {std_daily_return:.6f}")
    print(f"Sharpe Ratio (annualized): {sharpe_ratio:.4f}")
    
    # 3. Sortino Ratio
    # Sortino uses downside deviation instead of standard deviation
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    
    if downside_std > 0:
        sortino_ratio = (excess_daily_return / downside_std) * np.sqrt(252)
    else:
        sortino_ratio = np.inf
    
    print(f"\n--- Sortino Ratio ---")
    print(f"Downside returns count: {len(downside_returns)}")
    print(f"Downside std: {downside_std:.6f}")
    print(f"Sortino Ratio (annualized): {sortino_ratio:.4f}")
    
    metrics = {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'total_return': total_return,
        'mean_daily_return': mean_daily_return,
        'std_daily_return': std_daily_return,
        'downside_std': downside_std
    }
    
    return metrics

# -------------------- Visualizations --------------------
def create_visualizations(df, y_test, y_test_pred, model_metrics, fin_metrics, output_dir='Assignment8'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Stock Price History
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Adj_Close'], linewidth=1, color='#1f77b4')
    plt.title(f'{STOCK_SYMBOL} - Adjusted Close Price History', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_history.png'), dpi=300)
    print("Saved: price_history.png")
    plt.close()
    
    # 2. Daily Returns Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['Daily_Return'].dropna(), bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df['Daily_Return'], linewidth=0.5, color='#e74c3c')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Daily Returns Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_analysis.png'), dpi=300)
    print("Saved: returns_analysis.png")
    plt.close()
    
    # 3. Actual vs Predicted Returns (Test Set)
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=10, color='#3498db')
    plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', linewidth=1)
    plt.xlabel('Actual Return')
    plt.ylabel('Predicted Return')
    plt.title('Actual vs Predicted Returns', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Time series comparison (first 100 days)
    plt.subplot(2, 2, 2)
    days_to_plot = min(100, len(y_test))
    plt.plot(range(days_to_plot), y_test[:days_to_plot], 'b-', label='Actual', linewidth=1)
    plt.plot(range(days_to_plot), y_test_pred[:days_to_plot], 'r-', label='Predicted', linewidth=1, alpha=0.7)
    plt.xlabel('Trading Day')
    plt.ylabel('Return')
    plt.title('Actual vs Predicted (First 100 Days)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_test_pred
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residuals over time
    plt.subplot(2, 2, 4)
    plt.plot(range(len(residuals)), residuals, 'o', markersize=2, alpha=0.5, color='#e67e22')
    plt.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    plt.xlabel('Trading Day')
    plt.ylabel('Residual')
    plt.title('Residuals Over Time', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300)
    print("Saved: prediction_analysis.png")
    plt.close()
    
    # 4. Financial Metrics Summary
    plt.figure(figsize=(10, 6))
    metrics_names = ['Annual Return', 'Sharpe Ratio', 'Sortino Ratio']
    metrics_values = [
        fin_metrics['annual_return'],
        fin_metrics['sharpe_ratio'],
        fin_metrics['sortino_ratio']
    ]
    
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in metrics_values]
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.ylabel('Value')
    plt.title(f'{STOCK_SYMBOL} - Financial Metrics Summary', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * (1 if height > 0 else -1),
                f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300)
    print("Saved: metrics_summary.png")
    plt.close()

# -------------------- Main Execution --------------------
def main():
    """
    Main execution pipeline for Assignment 8.
    """
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # Step 1: Fetch data
    raw_data = fetch_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    if raw_data is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # Step 2: Preprocess data
    processed_data = preprocess_stock_data(raw_data)
    
    # Step 3: Prepare features for model
    X, y, feature_names = prepare_features(processed_data)
    
    # Step 4: Train regression model
    model, X_train, X_test, y_train, y_test, model_metrics = train_regression_model(X, y)
    
    # Step 5: Make predictions on test set
    y_test_pred = model.predict(X_test)
    
    # Step 6: Calculate financial metrics on actual test returns
    fin_metrics = calculate_financial_metrics(y_test)
    
    # Step 7: Create visualizations
    create_visualizations(processed_data, y_test, y_test_pred, model_metrics, fin_metrics)
    
    # Step 8: Save processed data and model info
    processed_data.to_csv('Assignment8/processed_stock_data.csv')
    print("\nSaved: Assignment8/processed_stock_data.csv")
    
    # Step 9: Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    print("\n--- Model Performance Summary ---")
    print(f"Train RMSE: {model_metrics['train_rmse']:.6f}")
    print(f"Test RMSE: {model_metrics['test_rmse']:.6f}")
    print(f"Train MAE: {model_metrics['train_mae']:.6f}")
    print(f"Test MAE: {model_metrics['test_mae']:.6f}")
    
    print("\n--- Financial Metrics Summary ---")
    print(f"Annual Return: {fin_metrics['annual_return']:.4f} ({fin_metrics['annual_return']*100:.2f}%)")
    print(f"Sharpe Ratio: {fin_metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {fin_metrics['sortino_ratio']:.4f}")
    
    print("\n--- Generated Files ---")
    print("  - Assignment8/price_history.png")
    print("  - Assignment8/returns_analysis.png")
    print("  - Assignment8/prediction_analysis.png")
    print("  - Assignment8/metrics_summary.png")
    print("  - Assignment8/processed_stock_data.csv")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()