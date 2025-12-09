"""
walk_forward_NO_GNN.py

This is the "Ablation Study" orchestrator.
It runs the full walk-forward backtest using the engine_NO_GNN.py
script, which *excludes* GNN features from the GP.

This provides the "baseline" performance to compare against.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

# --- 1. Import the new engine ---
import engine_NO_GNN as engine # <-- CHANGED

# --- 2. Import GP Primitives (needed to compile the champion strings) ---
from deap import base, creator, tools, gp
import functools
import random
import operator

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
def delay_1(x):
    return pd.DataFrame(x).shift(1).bfill().values

# --- 3. Define Helper Functions ---

def get_compilation_toolbox(terminals_list):
    """
    Creates a DEAP toolbox with all primitives, used *only*
    for compiling the strategy string returned by the engine.
    
    --- THIS IS NOW A 1:1 MIRROR OF THE ENGINE'S PRIMITIVES ---
    """
    pset = gp.PrimitiveSet("MAIN", arity=len(terminals_list))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(terminals_list)})

    # 1. Arithmetic (from engine)
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_division, 2, name="pdiv") # Uses the function from top of file

    # 2. Logical Operators (from engine)
    def safe_gt(a, b): return (a > b).astype(int)
    def safe_lt(a, b): return (a < b).astype(int)
    def safe_and(a, b): return np.logical_and(a, b).astype(int)
    def safe_or(a, b): return np.logical_or(a, b).astype(int)
    def safe_not(a): return np.logical_not(a).astype(int)

    pset.addPrimitive(safe_gt, 2, name="gt")
    pset.addPrimitive(safe_lt, 2, name="lt")
    pset.addPrimitive(safe_and, 2, name="AND")
    pset.addPrimitive(safe_or, 2, name="OR")
    pset.addPrimitive(safe_not, 1, name="NOT")

    # 3. Rolling/Math Operators (from engine)
    pset.addPrimitive(rolling_mean_5, 1, name="sma_5")   # Uses the function from top of file
    pset.addPrimitive(rolling_mean_20, 1, name="sma_20_op") # Uses the function from top of file
    
    pset.addPrimitive(log_abs, 1, name="log_abs") # Uses the function from top of file
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")

    # 4. Ephemeral Constant
    pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))

    # 5. Create Toolbox
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    return toolbox


def run_out_of_sample_backtest(champion_string, toolbox, test_start_date, test_end_date,
                               all_feature_dfs, all_returns_data,
                               periphery_universe): # <-- gnn_embeddings_df removed
    """
    Runs a backtest of a single champion string on a single
    out-of-sample (OOS) period.
    --- NO GNN VERSION ---
    """
    print(f"  [Backtest] Running OOS test from {test_start_date.date()} to {test_end_date.date()}...")

    # --- 1. Prepare Data Panels for OOS period ---
    # Ensure periphery_universe is a list of strings and filter returns
    valid_periphery = [s for s in periphery_universe if s in all_returns_data.columns]
    oos_returns = all_returns_data.loc[test_start_date:test_end_date, valid_periphery]

    all_feature_panels = []

    # Add standard features
    for feature_name, df in all_feature_dfs.items():
        # Filter columns that exist in this feature's DataFrame AND are in our valid periphery
        valid_cols = [col for col in valid_periphery if col in df.columns]
        if not valid_cols:
            continue # Skip this feature if no periphery stocks have it
            
        panel_slice = df.loc[test_start_date:test_end_date, valid_cols]
        panel_slice.columns = pd.MultiIndex.from_product([panel_slice.columns, [feature_name]])
        all_feature_panels.append(panel_slice)

    # --- GNN FEATURES REMOVED ---

    if not all_feature_panels:
        print(f"  [Backtest] WARNING: No OOS feature data found for period. Skipping.")
        return pd.Series(0.0, index=pd.date_range(test_start_date, test_end_date))

    feature_panel = pd.concat(all_feature_panels, axis=1)

    oos_returns, feature_panel = oos_returns.align(feature_panel, join='inner', axis=0)

    if oos_returns.empty or feature_panel.empty:
        print(f"  [Backtest] WARNING: No aligned OOS data found for period. Skipping.")
        return pd.Series(0.0, index=pd.date_range(test_start_date, test_end_date))

    # --- 2. Compile and Run Strategy ---
    compile_terminals = feature_panel.columns.get_level_values(1).unique().tolist()

    # --- FIX 1 HERE ---
    # Check if we need to re-create the toolbox
    if 'compile' not in toolbox.__dict__ or set(compile_terminals) != set(toolbox.compile.keywords['pset'].arguments):
        toolbox = get_compilation_toolbox(compile_terminals)

    strategy_func = toolbox.compile(expr=champion_string)

    feature_args = {}
    # --- THIS IS THE CRITICAL FIX ---
    for feature_name in toolbox.compile.keywords['pset'].arguments:
        # Use .xs() to correctly slice the MultiIndex by level 1 (the feature name)
        feature_args[feature_name] = feature_panel.xs(feature_name, level=1, axis=1).values
    # --- END CRITICAL FIX ---
        
    try:
        raw_output = strategy_func(**feature_args)
        if isinstance(raw_output, (int, float)):
            raw_output = np.full(oos_returns.shape, raw_output)

        raw_output = raw_output.astype(float)
        signals = np.where(raw_output > 0.5, 1, 0) # Long-only signals
        
        # Get the correct stock names from the aligned feature panel's level 0 index
        stock_columns = feature_panel.xs(compile_terminals[0], level=1, axis=1).columns
        signals_df = pd.DataFrame(signals, index=oos_returns.index, columns=stock_columns)
        
        # Align signals with OOS returns
        signals_df, aligned_oos_returns = signals_df.align(oos_returns, join='right', axis=1)
        signals_df.fillna(0, inplace=True) # Fill NaNs for stocks not in feature panel

        oos_daily_returns = engine.calculate_portfolio_returns(signals_df, aligned_oos_returns)

    except Exception as e:
        print(f"  [Backtest] ERROR running OOS backtest: {e}")
        import traceback
        traceback.print_exc()
        oos_daily_returns = pd.Series(0.0, index=oos_returns.index)

    return oos_daily_returns

# --- 4. MAIN WALK-FORWARD EXECUTION ---
if __name__ == "__main__":
    
    print("--- Starting Walk-Forward Validation (Hard Reset) [NO GNN BASELINE] ---")
    
    # --- 1. Define Paths ---
    WEIGHTS_FILE_PATH = 'data/survivorship/weights.csv' # <-- The Figshare file
    PROCESSED_DIR = 'data/processed'
    PRICE_DATA_PATH = os.path.join(PROCESSED_DIR, 'close_50.csv')
    
    # --- 2. Define Walk-Forward Chunks ---
    walk_forward_chunks = [
        # Train: 2010-2012, Test: 2013
        ('2010-01-01', '2012-12-31', '2013-01-01', '2013-12-31'),
        # Train: 2010-2013, Test: 2014
        ('2010-01-01', '2013-12-31', '2014-01-01', '2014-12-31'),
        # Train: 2010-2014, Test: 2015
        ('2010-01-01', '2014-12-31', '2015-01-01', '2015-12-31'),
        # Train: 2010-2015, Test: 2016
        ('2010-01-01', '2015-12-31', '2016-01-01', '2016-12-31'),
        # Train: 2010-2016, Test: 2017
        ('2010-01-01', '2016-12-31', '2017-01-01', '2017-12-31'),
        # Train: 2010-2017, Test: 2018
        ('2010-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
        # Train: 2010-2018, Test: 2019
        ('2010-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
        # Train: 2010-2019, Test: 2020
        ('2010-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
        # Train: 2010-2020, Test: 2021
        ('2010-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
        # Train: 2010-2021, Test: 2022
        ('2010-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        # Train: 2010-2022, Test: 2023
        ('2010-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        # Train: 2010-2023, Test: 2024
        ('2010-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ]

    # --- 3. Load ALL Data (Once) ---
    print("Loading all base data files (once)...")
    all_price_data = pd.read_csv(PRICE_DATA_PATH, index_col='Date', parse_dates=True)
    all_returns_data = all_price_data.pct_change()

    all_price_data.columns = [c if ".NS" in c else f"{c}.NS" for c in all_price_data.columns]
    all_returns_data.columns = [c if ".NS" in c else f"{c}.NS" for c in all_returns_data.columns]

    all_feature_dfs = {
        'sma_50': pd.read_csv(os.path.join(PROCESSED_DIR, 'sma_50_50.csv'), index_col='Date', parse_dates=True),
        'sma_20': pd.read_csv(os.path.join(PROCESSED_DIR, 'sma_200_50.csv'), index_col='Date', parse_dates=True), # <-- FIX IS HERE
        'rsi_14': pd.read_csv(os.path.join(PROCESSED_DIR, 'rsi_14_50.csv'), index_col='Date', parse_dates=True),
        'atr_14': pd.read_csv(os.path.join(PROCESSED_DIR, 'atr_14_50.csv'), index_col='Date', parse_dates=True)
    }
    
    ns_tickers = all_price_data.columns
    ticker_map = { ticker.replace(".NS", ""): ticker for ticker in ns_tickers }
    
    for key in all_feature_dfs:
        all_feature_dfs[key].columns = [ticker_map.get(c, c) for c in all_feature_dfs[key].columns]

    # --- 4. Execute Walk-Forward Loop ---
    all_oos_returns_list = []
    # Create a dummy toolbox to start
    oos_toolbox = get_compilation_toolbox(list(all_feature_dfs.keys()))
    
    for chunk in walk_forward_chunks:
        train_start, train_end, test_start, test_end = chunk
        train_end_date = pd.to_datetime(train_end)
        
        print(f"\n--- Processing Chunk: Train {train_start[:4]}-{train_end[:4]}, Test {test_start[:4]} ---")
        
        try:
            # --- Ticker Mapping ---
            valid_universe_base = engine.get_survivorship_free_universe(train_end_date, WEIGHTS_FILE_PATH)
            valid_universe_ns = [ ticker_map[t] for t in valid_universe_base if t in ticker_map ]
            
            # --- 1. Build Point-in-Time TMFG ---
            tmfg_graph = engine.build_point_in_time_tmfg(
                all_price_data, valid_universe_ns, train_end_date, lookback_years=3
            )
            
            # --- 2. GNN TRAINING (SKIPPED) ---
            
            # --- 3. Run GP Strategy Discovery ---
            gp_lookback_yrs = int(train_end[:4]) - int(train_start[:4]) + 1
            
            champion_string, gp_periphery_universe = engine.run_gp_discovery(
                all_feature_dfs, all_returns_data, # gnn_embeddings removed
                tmfg_graph, train_end_date, lookback_years=gp_lookback_yrs
            )

            # --- 4. Run OOS Backtest ---
            # The terminals are just the standard features
            test_terminals = list(all_feature_dfs.keys())
            oos_toolbox = get_compilation_toolbox(test_terminals)
            
            periphery_nodes = [n for n in gp_periphery_universe] # No GNN index to check

            oos_returns = run_out_of_sample_backtest(
                champion_string, oos_toolbox, 
                pd.to_datetime(test_start), pd.to_datetime(test_end),
                all_feature_dfs, all_returns_data,
                periphery_nodes # gnn_embeddings removed
            )
            
            all_oos_returns_list.append(oos_returns)
            print(f"--- Chunk {test_start[:4]} Complete ---")

        except Exception as e:
            print(f"--- FAILED Chunk: Test {test_start[:4]} ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            date_range = pd.date_range(test_start, test_end)
            all_oos_returns_list.append(pd.Series(0.0, index=date_range))
            
    # --- 5. Analyze Final Results ---
    print("\n--- Walk-Forward Analysis Complete [NO GNN BASELINE] ---")
    
    final_oos_returns = pd.concat(all_oos_returns_list)
    final_oos_returns.name = "Strategy_No_GNN"
    
    # Save results to a separate file
    final_oos_returns.to_csv("walk_forward_returns_NO_GNN.csv")
    
    # Calculate final, *defensible* metrics
    final_sharpe = engine.imported_calculate_sharpe(final_oos_returns.to_frame(), final_oos_returns.to_frame(), 0.0)
    final_mdd = engine.calculate_max_drawdown(final_oos_returns)
    cumulative_returns = (1 + final_oos_returns).cumprod()
    
    total_years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    cagr = (cumulative_returns.iloc[-1] ** (1 / total_years)) - 1
    
    print("\n--- FINAL DEFENSIBLE METRICS [NO GNN BASELINE] (2013-2024) ---")
    print(f"  Annualized Return (CAGR): {cagr*100:.2f} %")
    print(f"  Annualized Sharpe Ratio:  {final_sharpe:.3f}")
    print(f"  Maximum Drawdown:         {final_mdd*100:.2f} %")
    
    # --- 6. Save Plot ---
    print("\nSaving final walk-forward equity curve to 'walk_forward_equity_NO_GNN.png'...")
    plt.figure(figsize=(12, 8))
    cumulative_returns.plot(title='Walk-Forward Equity Curve (Out-of-Sample) [NO GNN BASELINE]', logy=True)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (Log Scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('walk_forward_equity_NO_GNN.png')
    
    print("--- Script Finished ---")