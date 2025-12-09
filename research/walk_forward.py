"""
walk_forward.py

This script is the main orchestrator for the "hard reset" validation.
It imports the functions from engine.py and runs the full, point-in-time,
survivorship-free walk-forward backtest.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

# --- 1. Import the new engine ---
import engine

# --- 2. Import GP Primitives (needed to compile the champion strings) ---
from deap import base, creator, tools, gp
import functools
import random
import operator

# --- GP Primitive Definitions (must match engine.py) ---
# This block is not strictly necessary but good for reference
def protected_division(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where(np.abs(right) < 1e-6, 1.0, left / right)
        x[np.isinf(x)] = 1.0
        return x
def log_abs(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.abs(x) + 1e-6)
def rolling_mean_5(x):
    return pd.DataFrame(x).rolling(5, min_periods=1).mean().values
def rolling_mean_20(x):
    return pd.DataFrame(x).rolling(20, min_periods=1).mean().values
# --- End setup block ---

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 3. Walk-Forward Configuration ---
START_YEAR = 2010
FIRST_TEST_YEAR = 2013
FINAL_TEST_YEAR = 2024
LOOKBACK_YEARS = 3 # Train on 2010, 2011, 2012 -> Test on 2013

# File paths
PROCESSED_DIR = "data/processed"
SURVIVORSHIP_FILE = os.path.join("data", "survivorship", "weights.csv")

# --- 4. Main Walk-Forward Loop ---

def run_walk_forward():
    print("--- Starting Walk-Forward Validation (Hard Reset) ---")
    
    try:
        # Load all data *once*
        all_data_dfs, ticker_map = engine.load_all_data(PROCESSED_DIR)
        
        # Get the specific DFs we need
        all_price_data = all_data_dfs.pop("all_price_data")
        all_returns_data = all_data_dfs.pop("all_returns_data")
        # all_feature_dfs now *only* contains features (rsi, sma, etc.)
        
    except Exception as e:
        print(f"Failed to load base data: {e}")
        return

    all_oos_returns_list = []
    all_oos_signals_list = [] 
    all_champion_strategies = [] # <-- Your new line

    for year in range(FIRST_TEST_YEAR, FINAL_TEST_YEAR + 1):
        
        # --- 1. Define Dates ---
        train_end_date = pd.to_datetime(f"{year-1}-12-31")
        test_start_date = pd.to_datetime(f"{year}-01-01")
        test_end_date = pd.to_datetime(f"{year}-12-31")
        
        # This makes it a 3-year *rolling* window
        train_start_date = train_end_date - pd.DateOffset(years=LOOKBACK_YEARS) + pd.DateOffset(days=1)
        
        print(f"\n--- Processing Chunk: Train {train_start_date.year}-{train_end_date.year}, Test {year} ---")
        
        try:
            # --- 2. Get Universe ---
            valid_universe_base = engine.get_survivorship_free_universe(train_end_date, SURVIVORSHIP_FILE)
            valid_universe_ns = [ ticker_map[t] for t in valid_universe_base if t in ticker_map ]
            print(f"  [Engine] Universe for {train_end_date.date()} has {len(valid_universe_ns)} stocks.")

            # --- 3. Build TMFG ---
            tmfg_graph = engine.build_point_in_time_tmfg(
                all_price_data, valid_universe_ns, train_end_date, LOOKBACK_YEARS
            )
            
            # --- 4. Train GNN ---
            gnn_embeddings = engine.train_point_in_time_gnn(
                tmfg_graph, all_data_dfs, train_end_date, LOOKBACK_YEARS
            )
            
            # --- 5. Run GP Discovery ---
            champion_strategy_string, gp_periphery_universe = engine.run_gp_discovery(
                gnn_embeddings, all_data_dfs, all_returns_data,
                tmfg_graph, train_end_date, LOOKBACK_YEARS
            )
            
            if champion_strategy_string == "strategy_error":
                raise Exception("GP discovery failed.")
            
            # --- Your new feature (Correct) ---
            all_champion_strategies.append({"year": year, "strategy": champion_strategy_string})
            
            # --- 6. Run OOS Backtest ---
            # FIX: Pass all_returns_data to the engine function
            oos_returns, oos_signals = engine.run_oos_backtest(
                champion_strategy_string,
                all_data_dfs,      # The dictionary of all feature dataframes
                all_returns_data,  # <-- This was missing
                gnn_embeddings,    # The static embeddings we just trained
                gp_periphery_universe,
                test_start_date,
                test_end_date
            )
            
            all_oos_returns_list.append(oos_returns)
            all_oos_signals_list.append(oos_signals)
            print(f"--- Chunk {year} Complete ---")
        
        except Exception as e:
            print(f"!!! ERROR processing chunk for {year}: {e}")
            import traceback
            traceback.print_exc()
            date_range = pd.date_range(start=test_start_date, end=test_end_date, freq='B')
            all_oos_returns_list.append(pd.Series(0.0, index=date_range))
            
            # Fix: Use list() for columns
            all_oos_signals_list.append(pd.DataFrame(0.0, index=date_range, columns=list(ticker_map.values())))
            
    # --- 7. Analyze Final Results ---
    print("\n--- Walk-Forward Analysis Complete ---")
    
    final_oos_returns = pd.concat(all_oos_returns_list)
    final_oos_returns.name = "Strategy"
    
    final_oos_signals = pd.concat(all_oos_signals_list)
    final_champions_df = pd.DataFrame(all_champion_strategies)

    # --- Your new save-to-CSV block (Correct) ---
    final_oos_returns.to_csv("walk_forward_returns.csv")
    final_oos_signals.to_csv("walk_forward_signals.csv")
    final_champions_df.to_csv("walk_forward_champions.csv", index=False)
    print("Saved final returns, signals, and champion strategies to CSV.")
    # --- END ---

    
    # Calculate final, *defensible* metrics
    final_sharpe = engine.imported_calculate_sharpe(final_oos_returns.to_frame(), final_oos_returns.to_frame(), 0.0)
    final_mdd = engine.calculate_max_drawdown(final_oos_returns)
    
    cumulative_returns = (1 + final_oos_returns.fillna(0)).cumprod()
    
    # Calculate CAGR
    if cumulative_returns.empty:
        cagr = 0.0
    else:
        total_years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
        if total_years > 0:
            cagr = (cumulative_returns.iloc[-1] ** (1 / total_years)) - 1
        else:
            cagr = 0.0
    
    print("\n--- FINAL DEFENSIBLE METRICS (2013-2024) ---")
    print(f"  Annualized Return (CAGR): {cagr*100:.2f} %")
    print(f"  Annualized Sharpe Ratio:  {final_sharpe:.3f}")
    print(f"  Maximum Drawdown:         {final_mdd*100:.2f} %")
    
    # --- 8. Plot Final Equity Curve ---
    plt.figure(figsize=(12, 7))
    cumulative_returns.plot(lw=2)
    plt.title(f"Walk-Forward Equity Curve (GNN Strategy)\nSharpe: {final_sharpe:.3f} | CAGR: {cagr*100:.2f}% | MDD: {final_mdd*100:.2f}%")
    plt.ylabel("Cumulative Returns")
    plt.grid(True, linestyle='--')
    plt.savefig("walk_forward_equity.png")
    print("Saving final walk-forward equity curve to 'walk_forward_equity.png'...")

    print("--- Script Finished ---")

if __name__ == "__main__":
    run_walk_forward()