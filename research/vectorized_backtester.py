# scripts/vectorized_backtester.py

import pandas as pd
import numpy as np

def calculate_sharpe_ratio(signals_df: pd.DataFrame, returns_df: pd.DataFrame, risk_free_rate: float) -> float:
    """
    Calculates the annualized Sharpe Ratio for a set of trading signals using a fast, vectorized approach.

    This function is optimized for speed and is intended for use in performance-critical
    applications like a Genetic Programming evolution loop.

    Args:
        signals_df (pd.DataFrame): DataFrame of trading signals (1 for Long, -1 for Short, 0 for Flat).
                                   Index is Date, columns are tickers.
        returns_df (pd.DataFrame): DataFrame of daily percentage returns for the same tickers and dates.
        risk_free_rate (float): The annual risk-free rate (e.g., 0.05 for 5%).

    Returns:
        float: The annualized Sharpe Ratio.
    """
    # 1. Alignment: Ensure dataframes are perfectly aligned by date and tickers.
    aligned_signals, aligned_returns = signals_df.align(returns_df, join='inner', axis=0)

    # 2. Prevent Lookahead Bias: Shift signals to ensure trades are based on prior information.
    # The signal from day T-1 determines the return for day T.
    trade_signals = aligned_signals.shift(1)

    # 3. Calculate Daily Portfolio Returns
    # Element-wise multiplication gives the return for each individual position
    daily_position_returns = trade_signals.multiply(aligned_returns)

    # Sum returns across all positions for each day to get the raw daily portfolio return
    raw_daily_portfolio_return = daily_position_returns.sum(axis=1)

    # Equal Weighting: Normalize returns by the number of active positions
    # Count active signals (long or short) for each day
    num_active_positions = trade_signals.abs().sum(axis=1)

    # Avoid division by zero on days with no trades
    safe_num_active = num_active_positions.replace(0, 1)

    # Calculate the equal-weighted daily portfolio returns
    daily_portfolio_returns = (raw_daily_portfolio_return / safe_num_active).fillna(0)

    # 4. Calculate Sharpe Ratio
    # Calculate standard deviation of the portfolio returns
    std_dev_daily_returns = daily_portfolio_returns.std()

    # If there's no volatility, the Sharpe Ratio is 0.
    if std_dev_daily_returns == 0:
        return 0.0

    # Calculate daily excess returns over the risk-free rate
    daily_risk_free_rate = risk_free_rate / 252
    daily_excess_returns = daily_portfolio_returns - daily_risk_free_rate
    mean_daily_excess_return = daily_excess_returns.mean()

    # Compute the annualized Sharpe Ratio
    sharpe_ratio = (mean_daily_excess_return / std_dev_daily_returns) * np.sqrt(252)

    return sharpe_ratio

# --- Testing Block ---
if __name__ == '__main__':
    print("--- Running Test for Vectorized Backtester ---")

    # Create sample data for testing
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=252))
    tickers = ['STOCK_A', 'STOCK_B']

    # Sample returns: STOCK_A gains 0.1% daily, STOCK_B loses 0.05% daily
    returns_data = np.array([np.full(252, 0.001), np.full(252, -0.0005)]).T
    test_returns_df = pd.DataFrame(returns_data, index=dates, columns=tickers)

    # Sample signals: Always long STOCK_A, always short STOCK_B
    signals_data = np.array([np.ones(252), np.full(252, -1)]).T
    test_signals_df = pd.DataFrame(signals_data, index=dates, columns=tickers)

    # Set risk-free rate
    rfr = 0.05

    # Calculate the Sharpe Ratio
    calculated_sharpe = calculate_sharpe_ratio(
        signals_df=test_signals_df,
        returns_df=test_returns_df,
        risk_free_rate=rfr
    )

    print(f"\nSample DataFrames created for {len(dates)} days and {len(tickers)} stocks.")
    print(f"Risk-Free Rate: {rfr*100:.2f}%")
    print(f"\nCalculated Annualized Sharpe Ratio: {calculated_sharpe:.4f}")

    # Manual verification for this simple case:
    # Daily return for STOCK_A position: 1 * 0.001 = 0.001
    # Daily return for STOCK_B position: -1 * -0.0005 = 0.0005
    # Raw daily portfolio return: 0.001 + 0.0005 = 0.0015
    # Number of positions is always 2.
    # Equal-weighted daily return: 0.0015 / 2 = 0.00075
    # Since the return is constant, the standard deviation is 0, so Sharpe should be 0.
    # Let's add some noise to returns for a better test.
    print("\n--- Running a second test with volatile returns ---")
    np.random.seed(42)
    volatile_returns = test_returns_df + np.random.normal(0, 0.01, test_returns_df.shape)
    calculated_sharpe_volatile = calculate_sharpe_ratio(
        signals_df=test_signals_df,
        returns_df=volatile_returns,
        risk_free_rate=rfr
    )
    print(f"Calculated Annualized Sharpe Ratio (Volatile): {calculated_sharpe_volatile:.4f}")