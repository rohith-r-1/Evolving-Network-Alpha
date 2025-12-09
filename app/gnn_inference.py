import pandas as pd
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "data", "artifacts")

GNN_RETURNS_PATH = os.path.join(ARTIFACT_DIR, "walk_forward_returns.csv")
BASE_RETURNS_PATH = os.path.join(ARTIFACT_DIR, "walk_forward_returns_NO_GNN.csv")


GNN_RET_COL = "Strategy"
BASE_RET_COL = "Strategy_No_GNN"

TRADING_DAYS_PER_YEAR = 252

def load_returns(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def compute_equity_curve(returns: pd.Series, initial_capital: float = 1.0):
    equity = (1 + returns).cumprod() * initial_capital
    return equity

def compute_cagr(equity: pd.Series):
    n_days = len(equity)
    if n_days == 0:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = n_days / TRADING_DAYS_PER_YEAR
    return total_return**(1 / years) - 1

def compute_sharpe(returns: pd.Series, rf: float = 0.0):
    if returns.std() == 0:
        return np.nan
    excess = returns - rf / TRADING_DAYS_PER_YEAR
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std()

def compute_max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return drawdown.min()

def get_backtest_summary():
    gnn_df = load_returns(GNN_RETURNS_PATH)
    base_df = load_returns(BASE_RETURNS_PATH)

    gnn_eq = compute_equity_curve(gnn_df[GNN_RET_COL])
    base_eq = compute_equity_curve(base_df[BASE_RET_COL])

    gnn_cagr = compute_cagr(gnn_eq)
    base_cagr = compute_cagr(base_eq)

    gnn_sharpe = compute_sharpe(gnn_df[GNN_RET_COL])
    base_sharpe = compute_sharpe(base_df[BASE_RET_COL])

    gnn_mdd = compute_max_drawdown(gnn_eq)
    base_mdd = compute_max_drawdown(base_eq)

    summary = {
        "start_date": str(gnn_df["Date"].min().date()),
        "end_date": str(gnn_df["Date"].max().date()),
        "gnn": {
            "CAGR": float(gnn_cagr),
            "Sharpe": float(gnn_sharpe),
            "MaxDrawdown": float(gnn_mdd),
        },
        "baseline": {
            "CAGR": float(base_cagr),
            "Sharpe": float(base_sharpe),
            "MaxDrawdown": float(base_mdd),
        },
    }
    return summary

def get_equity_curve():
    gnn_df = load_returns(GNN_RETURNS_PATH)
    base_df = load_returns(BASE_RETURNS_PATH)

    gnn_eq = compute_equity_curve(gnn_df[GNN_RET_COL])
    base_eq = compute_equity_curve(base_df[BASE_RET_COL])

    out = pd.DataFrame({
        "Date": gnn_df["Date"],
        "Equity_GNN": gnn_eq,
        "Equity_Baseline": base_eq,
    })
    return out

if __name__ == "__main__":
    summary = get_backtest_summary()
    print(summary)
 