"""
engine.py

This script contains the complete, refactored, point-in-time (causal)
pipeline for the Evolving Network-Aware Alpha project.

This version uses the correct TMFG.transform() API and
the correct data loading logic.
"""

# --- 1. CORE IMPORTS ---
import pandas as pd
import numpy as np
import networkx as nx
import os
import random
import operator
import pickle
import functools
import warnings

# --- 2. ML/AI IMPORTS ---

# TMFG (assumes TMFG.py is in the same directory)
try:
    from TMFG import TMFG
except ImportError:
    print("Warning: TMFG.py not found. TMFG functions will fail.")
    # Define a placeholder if not found so the script can be read
    class TMFG:
        def fit(self, *args): pass
        def transform(self, *args, **kwargs): return None, None, np.array([[]])

# GNN Imports
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling
from sklearn.preprocessing import StandardScaler

# GP Imports
from deap import base, creator, tools, gp, algorithms

# Local Backtester (Task 1)
try:
    from vectorized_backtester import calculate_sharpe_ratio as imported_calculate_sharpe
except ImportError:
    print("Warning: vectorized_backtester.py not found. GP functions will fail.")
    def imported_calculate_sharpe(*args, **kwargs): return 0.0

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 3. GP BACKTESTING HELPER FUNCTIONS ---

def calculate_portfolio_returns(signals_df, returns_df):
    """Helper function to get the daily portfolio returns series."""
    signals, returns = signals_df.align(returns_df, join='inner', axis=0)
    trade_signals = signals.shift(1) # Prevent lookahead bias
    
    portfolio_daily_returns = (trade_signals * returns).sum(axis=1)
    num_active_positions = trade_signals.abs().sum(axis=1)
    portfolio_daily_returns = portfolio_daily_returns / num_active_positions.replace(0, 1)
    portfolio_daily_returns.fillna(0, inplace=True)
    return portfolio_daily_returns

def calculate_max_drawdown(portfolio_returns):
    """Calculates the Maximum Drawdown from a returns series."""
    if portfolio_returns.empty or (portfolio_returns == 0).all():
        return 0.0
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return abs(max_drawdown) # Return the absolute value

# --- 4. CONFIGURATION ---
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SURVIVORSHIP_DIR = os.path.join(DATA_DIR, "survivorship")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SURVIVORSHIP_DIR, exist_ok=True)

# --- TMFG/GNN Settings ---
GNN_EMBEDDING_DIM = 16
GNN_EPOCHS = 200
GNN_LR = 0.01

# --- GP Settings ---
GP_POPULATION = 100
GP_GENERATIONS = 100 
GP_TOURNAMENT_SIZE = 3
GP_CXPB = 0.8  # Crossover probability
GP_MUTPB = 0.2 # Mutation probability

# --- 5. DATA LOADING FUNCTIONS ---

def load_all_data(processed_dir=PROCESSED_DIR):
    """
    Loads all required CSVs into a dictionary of DataFrames.
    This is run once at the start of the walk-forward script.
    """
    print("Loading all base data files (once)...")
    base_files = {
        "all_price_data": "close_50.csv", 
        "rsi_14": "rsi_14_50.csv",
        "sma_20": "sma_200_50.csv", # Using sma_200 as sma_20
        "sma_50": "sma_50_50.csv",
        "atr_14": "atr_14_50.csv"
    }
    
    data_dfs = {}
    try:
        for key, filename in base_files.items():
            file_path = os.path.join(processed_dir, filename)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Map columns to .NS for consistency
            df.columns = [c if ".NS" in c else f"{c}.NS" for c in df.columns]
            data_dfs[key] = df
        
        # Create all_returns_data from all_price_data
        data_dfs["all_returns_data"] = data_dfs["all_price_data"].pct_change()

        ns_tickers = data_dfs["all_returns_data"].columns
        ticker_map = { ticker.replace(".NS", ""): ticker for ticker in ns_tickers }
        
        return data_dfs, ticker_map

    except FileNotFoundError as e:
        print(f"Error: Missing base data file: {e.filename}")
        print("Please ensure your 'data/processed' folder has all the correct CSV files.")
        raise
    except Exception as e:
        print(f"Failed to load base data: {e}")
        raise

# --- 6. SURVIVORSHIP-FREE UNIVERSE FUNCTION ---

def get_survivorship_free_universe(end_date, weights_file_path):
    """
    Gets the list of Nifty 50 constituents for a *specific date*.
    """
    print(f"  [Engine] Getting universe for date: {end_date}")
    try:
        weights_df = pd.read_csv(weights_file_path, parse_dates=['DATE'], index_col='DATE')
    except FileNotFoundError:
        print(f"  [Engine] CRITICAL ERROR: Historical weights file not found at {weights_file_path}")
        raise
    
    snapshot_date = weights_df.loc[:end_date].index.max()
    if pd.isna(snapshot_date):
        raise ValueError(f"No historical constituent data found on or before {end_date}")
        
    latest_weights = weights_df.loc[snapshot_date]
    universe_tickers = latest_weights[latest_weights > 0].index.tolist()
    
    return universe_tickers

# --- 7. POINT-IN-TIME NETWORK (TMFG) ---

def build_point_in_time_tmfg(all_price_data_ns, valid_universe_ns, end_date, lookback_years):
    """
    Builds a TMFG graph using only data available up to end_date.
    """
    print("  [Engine] Building Point-in-Time TMFG...")
    
    start_date = end_date - pd.DateOffset(years=lookback_years)
    
    valid_cols = [col for col in valid_universe_ns if col in all_price_data_ns.columns]
    if not valid_cols:
        print("  [Engine] ERROR: No valid stocks in price data. Returning empty graph.")
        return nx.Graph()
        
    price_data_window_all_cols = all_price_data_ns[valid_cols]
    price_data_window = price_data_window_all_cols.loc[start_date:end_date]
    
    if price_data_window.empty:
        print("  [Engine] ERROR: No price data in the window. Returning empty graph.")
        return nx.Graph()

    returns_window = price_data_window.pct_change(fill_method=None)
    
    if returns_window.empty or returns_window.shape[1] < 2:
        print("  [Engine] ERROR: Not enough data to calculate correlation. Returning empty graph.")
        return nx.Graph()
        
    corr_matrix = returns_window.corr(method='spearman')
    
    tmfg_builder = TMFG()
    
    tmfg_builder.fit(corr_matrix, output="unweighted_sparse_W_matrix")
    cliques, separators, J = tmfg_builder.transform()
    
    J_df = pd.DataFrame(J, index=corr_matrix.index, columns=corr_matrix.columns)
    G = nx.from_pandas_adjacency(J_df)
    
    G.remove_nodes_from(list(nx.isolates(G)))
    
    print(f"  [Engine] TMFG built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

# --- 8. POINT-IN-TIME GNN ---

# GNN Model Definition
class GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        # FIX: Added sigmoid to ensure output is a probability [0, 1]
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

def train_point_in_time_gnn(tmfg_graph, all_feature_dfs, end_date, lookback_years):
    """
    Trains a Graph Autoencoder on the TMFG to generate node embeddings.
    """
    print("  [Engine] Training Point-in-Time GNN...")
    
    start_date = end_date - pd.DateOffset(years=lookback_years)
    
    graph_nodes = list(tmfg_graph.nodes())
    
    if not graph_nodes:
        print("  [Engine] ERROR: TMFG graph is empty. Cannot train GNN.")
        return pd.DataFrame()
        
    feature_list = []
    feature_keys = [k for k in all_feature_dfs.keys() if k not in ['all_price_data', 'all_returns_data']]
    
    for key in feature_keys:
        df = all_feature_dfs[key]
        valid_cols = [col for col in graph_nodes if col in df.columns]
        if not valid_cols:
            print(f"  [Engine] Warning: No valid columns for feature '{key}' in GNN.")
            continue
            
        feature_slice = df.loc[:end_date, valid_cols].iloc[-1]
        feature_slice.name = key
        feature_list.append(feature_slice)
        
    if not feature_list:
        print("  [Engine] ERROR: No valid feature data found. Cannot train GNN.")
        return pd.DataFrame()
        
    feature_matrix_df = pd.concat(feature_list, axis=1)
    feature_matrix_df = feature_matrix_df.reindex(graph_nodes).fillna(0.0)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix_df)
    
    data = from_networkx(tmfg_graph)
    data.x = torch.tensor(X, dtype=torch.float)
    
    if data.edge_index is None or data.edge_index.shape[1] == 0:
        print("  [Engine] Warning: TMFG had 0 edges. Using a non-graph GNN.")
        data.edge_index = torch.tensor([[], []], dtype=torch.long)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAE(in_channels=X.shape[1], out_channels=GNN_EMBEDDING_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=GNN_LR)
    data = data.to(device)

    for epoch in range(GNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        
        pos_loss = -torch.log(
            model.decode(z, data.edge_index) + 1e-15
        ).mean()
        
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
        neg_loss = -torch.log(
            1 - model.decode(z, neg_edge_index) + 1e-15
        ).mean()
        
        loss = pos_loss + neg_loss
        
        if torch.isnan(loss):
             print(f"  [Engine] GNN loss is NaN at epoch {epoch}. Stopping.")
             break
             
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index).cpu().numpy()
        
    embedding_df = pd.DataFrame(
        final_embeddings, 
        index=feature_matrix_df.index, 
        columns=[f"GNN_{i}" for i in range(GNN_EMBEDDING_DIM)]
    )
    
    print(f"  [Engine] GNN training complete. Embeddings shape: {embedding_df.shape}")
    return embedding_df

# --- 9. POINT-IN-TIME GENETIC PROGRAMMING ---

def create_gp_primitives(pset):
    """
    Creates the DEAP primitive set (pset) for the GP.
    """
    
    # 1. Arithmetic
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    
    def protected_division(left, right):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(np.abs(right) < 1e-6, 1.0, left / right)
            x[np.isinf(x)] = 1.0
            return x
    pset.addPrimitive(protected_division, 2, name="pdiv")
    
    # FIX: Wrap logical operators to return integers (0 or 1)
    def safe_gt(a, b):
        return (a > b).astype(int)
    def safe_lt(a, b):
        return (a < b).astype(int)
    def safe_and(a, b):
        return np.logical_and(a, b).astype(int)
    def safe_or(a, b):
        return np.logical_or(a, b).astype(int)
    def safe_not(a):
        return np.logical_not(a).astype(int)

    pset.addPrimitive(safe_gt, 2, name="gt")
    pset.addPrimitive(safe_lt, 2, name="lt")
    pset.addPrimitive(safe_and, 2, name="AND")
    pset.addPrimitive(safe_or, 2, name="OR")
    pset.addPrimitive(safe_not, 1, name="NOT")
    
    def rolling_mean_5(x):
        return pd.DataFrame(x).rolling(5, min_periods=1).mean().values
    def rolling_mean_20(x):
        return pd.DataFrame(x).rolling(20, min_periods=1).mean().values
    
    pset.addPrimitive(rolling_mean_5, 1, name="sma_5")
    pset.addPrimitive(rolling_mean_20, 1, name="sma_20_op")
    
    def log_abs(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(np.abs(x) + 1e-6)
    
    pset.addPrimitive(log_abs, 1, name="log_abs")
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    
    return pset

def create_gp_toolbox(pset, feature_panel, returns_panel):
    """
    Creates the DEAP toolbox for running the GP.
    """
    toolbox = base.Toolbox()
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    eval_func = functools.partial(
        evaluate_strategy,
        feature_panel=feature_panel,
        returns_panel=returns_panel,
        pset=pset
    )
    toolbox.register("evaluate", eval_func)
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    
    return toolbox

def evaluate_strategy(individual, feature_panel, returns_panel, pset):
    """
    The "Fitness Function" for the GP.
    """
    try:
        strategy_func = gp.compile(expr=individual, pset=pset)
        
        signals_np = strategy_func(**feature_panel) # This is a numpy array
        signals_np = (signals_np > 0).astype(int)
        
        # --- FIX: Convert to DataFrame for backtester ---
        sample_df = feature_panel[list(feature_panel.keys())[0]]
        signals_df = pd.DataFrame(signals_np, index=sample_df.index, columns=sample_df.columns)
        
        signals_df, returns_panel_aligned = signals_df.align(returns_panel, join='inner', axis=0)

        sharpe = imported_calculate_sharpe(signals_df, returns_panel_aligned, 0.0)
        num_trades = np.abs(np.diff(signals_np, axis=0)).sum()
        
        if not np.isfinite(sharpe):
            sharpe = -10.0
        if not np.isfinite(num_trades):
            num_trades = 1_000_000
            
        return sharpe, -num_trades

    except (OverflowError, FloatingPointError, ValueError) as e:
        return -10.0, 1_000_000

def run_gp_discovery(gnn_embeddings, all_feature_dfs, all_returns_data, tmfg_graph, end_date, lookback_years):
    """
    Runs the full GP discovery process on point-in-time data.
    """
    print("  [Engine] Starting GP Strategy Discovery...")
    
    start_date = end_date - pd.DateOffset(years=lookback_years)
    
    if tmfg_graph.number_of_nodes() == 0:
        print("  [Engine] ERROR: TMFG graph is empty. Cannot run GP.")
        return "strategy_error", []
        
    centrality = nx.degree_centrality(tmfg_graph)
    centrality_series = pd.Series(centrality, name="centrality").sort_values()
    
    periphery_cutoff = centrality_series.quantile(0.3)
    gp_periphery_universe = centrality_series[centrality_series <= periphery_cutoff].index.tolist()
    
    if len(gp_periphery_universe) == 0:
        print(f"Warning: No periphery stocks found for {end_date.date()}. Using all stocks.")
        gp_periphery_universe = list(tmfg_graph.nodes())
        
    print(f"  [Engine] GP Universe: {len(gp_periphery_universe)} periphery stocks.")

    if gnn_embeddings.empty:
        print("  [Engine] ERROR: GNN embeddings are empty. Cannot run GP.")
        return "strategy_error", []
        
    valid_stocks_ns = gnn_embeddings.index.tolist()
    
    feature_panel = {}
    
    # Add GNN features
    for col in gnn_embeddings.columns:
        stocks_to_use = [s for s in gp_periphery_universe if s in valid_stocks_ns]
        gnn_feat_series = gnn_embeddings.loc[stocks_to_use, col]
        dates = all_returns_data.loc[start_date:end_date].index
        
        feat_df = pd.DataFrame(0.0, index=dates, columns=valid_stocks_ns, dtype=float)
        
        feat_df.loc[:, gnn_feat_series.index] = gnn_feat_series.values
        feature_panel[col] = feat_df

    # Add standard features
    std_features_to_add = {
        "rsi_14": "rsi_14",
        "sma_20": "sma_20",
        "sma_50": "sma_50",
        "atr_14": "atr_14"
    }
    
    for key, df_name in std_features_to_add.items():
        feat_df_full = all_feature_dfs[df_name]
        stocks_to_use = [s for s in gp_periphery_universe if s in feat_df_full.columns]
        feat_df_slice = feat_df_full.loc[start_date:end_date, stocks_to_use]
        feature_panel[key] = feat_df_slice.fillna(0)
        
    # --- 3. Set up GP Primitives and Toolbox ---
    
    # Clean up creator
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
    
    # --- FIX: Correctly define pset with arguments ---
    all_feature_names = list(feature_panel.keys())
    pset = gp.PrimitiveSet("MAIN", arity=len(all_feature_names))
    arg_map = {f"ARG{i}": name for i, name in enumerate(all_feature_names)}
    pset.renameArguments(**arg_map)
    pset = create_gp_primitives(pset) # Call the cleaned function
    # --- END FIX ---
    
    # Get the returns panel for the backtester
    returns_panel_stocks = [s for s in gp_periphery_universe if s in all_returns_data.columns]
    returns_panel = all_returns_data.loc[start_date:end_date, returns_panel_stocks].fillna(0)

    toolbox = create_gp_toolbox(pset, feature_panel, returns_panel)

    # --- 4. Run the GP Algorithm ---
    print(f"  [Engine] Running GP for {GP_GENERATIONS} generations...")
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean, axis=0)
    stats_fit.register("max", np.max, axis=0)
    
    final_pop, logbook = algorithms.eaMuPlusLambda(
        population=toolbox.population(n=GP_POPULATION),
        toolbox=toolbox,
        mu=GP_POPULATION,
        lambda_=GP_POPULATION,
        cxpb=GP_CXPB,
        mutpb=GP_MUTPB,
        ngen=GP_GENERATIONS,
        stats=stats_fit,
        halloffame=None,
        verbose=False
    )
    
    # --- 5. Select the Champion ---
    best_strategy = tools.selBest(final_pop, k=1)[0]
    best_sharpe = best_strategy.fitness.values[0]
    
    print(f"  [Engine] GP Discovery complete. Champion Sharpe: {best_sharpe:.3f}")
    
    return str(best_strategy), gp_periphery_universe

# --- 10. OUT-OF-SAMPLE BACKTEST FUNCTION ---

def run_oos_backtest(strategy_string, all_data_dfs, all_returns_data, gnn_embeddings, 
                      gp_periphery_universe, start_date, end_date):
    """
    Runs a single, compiled strategy string on out-of-sample data.
    This function rebuilds the necessary OOS panels.
    """
    print(f"  [Backtest] Running OOS test from {start_date.date()} to {end_date.date()}...")
    
    try:
        # --- 1. Get OOS Data Slices ---
        # FIX: Use the 'all_returns_data' DataFrame passed directly to the function
        oos_returns_panel = all_returns_data.loc[start_date:end_date].fillna(0)

        # --- 2. Build OOS Feature Panel ---
        valid_stocks_ns = gnn_embeddings.index.tolist()
        oos_feature_panel = {}

        # Add GNN features (static)
        for col in gnn_embeddings.columns:
            stocks_to_use = [s for s in gp_periphery_universe if s in valid_stocks_ns]
            gnn_feat_series = gnn_embeddings.loc[stocks_to_use, col]
            dates = oos_returns_panel.index
            
            feat_df = pd.DataFrame(0.0, index=dates, columns=valid_stocks_ns, dtype=float)
            feat_df.loc[:, gnn_feat_series.index] = gnn_feat_series.values
            oos_feature_panel[col] = feat_df

        # Add standard features (dynamic)
        std_features_to_add = {
            "rsi_14": "rsi_14", "sma_20": "sma_20",
            "sma_50": "sma_50", "atr_14": "atr_14"
        }
        for key, df_name in std_features_to_add.items():
            feat_df_full = all_data_dfs[df_name]
            feat_df_slice = feat_df_full.loc[start_date:end_date, valid_stocks_ns]
            oos_feature_panel[key] = feat_df_slice.fillna(0)

        # --- 3. Set up OOS Toolbox ---
        all_feature_names = list(oos_feature_panel.keys())
        pset = gp.PrimitiveSet("MAIN", arity=len(all_feature_names))
        arg_map = {f"ARG{i}": name for i, name in enumerate(all_feature_names)}
        pset.renameArguments(**arg_map)
        pset = create_gp_primitives(pset)
        
        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=pset)
        
        # --- 4. Compile and Run Strategy ---
        strategy_func = toolbox.compile(expr=strategy_string)
        
        signals_np = strategy_func(**oos_feature_panel)
        signals_np = (signals_np > 0).astype(int)
        
        # Convert to DataFrame
        sample_df = oos_feature_panel[list(oos_feature_panel.keys())[0]]
        signals_df = pd.DataFrame(signals_np, index=sample_df.index, columns=sample_df.columns)

        # --- 5. Filter signals to periphery and calculate returns ---
        mask = pd.DataFrame(0, index=signals_df.index, columns=signals_df.columns)
        valid_periphery_stocks = [s for s in gp_periphery_universe if s in signals_df.columns]
        
        if not valid_periphery_stocks:
            print("  [Backtest] Warning: No valid periphery stocks in OOS period.")
            return pd.Series(0.0, index=oos_returns_panel.index), pd.DataFrame(0.0, index=oos_returns_panel.index, columns=oos_returns_panel.columns)
        
        mask[valid_periphery_stocks] = 1
        signals_df = signals_df * mask
        
        signals_df, oos_returns_panel = signals_df.align(oos_returns_panel, join='inner', axis=0)
        
        oos_portfolio_returns = calculate_portfolio_returns(signals_df, oos_returns_panel)
        
        return oos_portfolio_returns, signals_df.loc[oos_portfolio_returns.index]

    except Exception as e:
        print(f"  [Backtest] Error running OOS test: {e}")
        import traceback
        traceback.print_exc()
        dates = pd.date_range(start_date, end_date, freq='B')
        # FIX: Added 'columns' to the empty DataFrame constructor
        return pd.Series(0.0, index=dates), pd.DataFrame(0.0, index=dates, columns=["error"])

# --- 11. MAIN (for demonstration) ---

if __name__ == "__main__":
    """
    This main block demonstrates one "snapshot" of the engine,
    showing how it would be called by the walk_forward.py script.
    """
    
    print("--- Running Engine Demo (1 Snapshot) ---")
    
    # --- Config ---
    END_DATE = pd.to_datetime("2018-12-31")
    LOOKBACK_YRS = 3 # Use 3-year lookback for demo
    WEIGHTS_FILE_PATH = os.path.join(SURVIVORSHIP_DIR, "weights.csv")
    
    try:
        # --- Load Data ---
        all_data_dfs, ticker_map = load_all_data(PROCESSED_DIR)
        
        all_price_data = all_data_dfs.pop("all_price_data")
        all_returns_data = all_data_dfs.pop("all_returns_data")
        all_feature_dfs = all_data_dfs
        
        # --- Run Snapshot ---
        
        print("\nStep 1: Get Survivorship-Free Universe")
        valid_universe_base = get_survivorship_free_universe(END_DATE, WEIGHTS_FILE_PATH)
        valid_universe_ns = [ ticker_map[t] for t in valid_universe_base if t in ticker_map ]
        print(f"  [Engine] Universe for {END_DATE.date()} has {len(valid_universe_ns)} stocks.")
        
        print("\nStep 2: Build Point-in-Time TMFG")
        tmfg_graph = build_point_in_time_tmfg(
            all_price_data, valid_universe_ns, END_DATE, LOOKBACK_YRS
        )
        
        print("\nStep 3: Train Point-in-Time GNN")
        gnn_embeddings = train_point_in_time_gnn(
            tmfg_graph, all_feature_dfs, END_DATE, LOOKBACK_YRS
        )
        
        print("\nStep 4: Run GP Strategy Discovery")
        champion_strategy_string, _ = run_gp_discovery(
            gnn_embeddings, all_feature_dfs, all_returns_data,
            tmfg_graph, END_DATE, LOOKBACK_YRS
        )
        
        print("\n--- SNAPSHOT COMPLETE ---")
        print(f"Date: {END_DATE.date()}")
        print(f"Champion Strategy Found:\n{champion_strategy_string}")
        
    except FileNotFoundError as e:
        print(f"\n--- DEMO FAILED ---")
        print(f"Could not find data file: {e}")
        print("Please download the survivorship 'weights.csv' from the Figshare link.")
    
    except Exception as e:
        print(f"\n--- DEMO FAILED ---")
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()