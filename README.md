# Evolving Network-Aware Alpha Factory

### A Comparative Walk-Forward Analysis of GNNs vs. Technical Analysis in Algorithmic Trading

![Status](https://img.shields.io/badge/Status-Completed-green) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red) ![Validation](https://img.shields.io/badge/Validation-Walk--Forward-orange)

## Executive Summary and Scientific Finding

This project was designed to test a specific quantitative hypothesis: **Does modeling the stock market as a complex network (using Graph Neural Networks) provide a superior predictive edge over traditional technical analysis?**

Instead of a simple backtest, I engineered a robust **"Alpha Factory"**—an automated pipeline that builds market networks, learns node embeddings, and evolves trading strategies using Genetic Programming.

### The Result (The Negative Finding)

After a rigorous 12-year **Rolling Walk-Forward Validation** (2013–2024), the hypothesis was **falsified**.

The final stitched backtest (computed from `data/artifacts/*.csv` and served by `app/gnn_inference.py`) produced:

| Strategy                    | Sharpe Ratio | CAGR    | Max Drawdown |
|----------------------------|-------------:|--------:|-------------:|
| Baseline (Simple Features) | 0.97 (approx.) | 16.6% (approx.) | -34% (approx.) |
| GNN Strategy (Complex)     | 0.87 (approx.) | 16.8% (approx.) | -36% (approx.) |

**Conclusion:** In this setup (short 3-year rolling training windows, Nifty 50 universe, 2013–2024), **model simplicity (Baseline) is more robust than model complexity (GNN)**. The GNN factory was profitable, but it did not outperform the simpler baseline and showed higher drawdowns.

The key research lesson: in non-stationary, noisy environments, deep graph-based models can overfit to short-window structure, while simple technical indicators remain more stable.

---

## Methodology: The "Hard Reset"

Early versions of this project achieved unrealistically strong performance (Sharpe > 1.4). A rigorous audit revealed **survivorship bias** and **lookahead bias**.

The project was rebuilt via a deliberate "Hard Reset" to enforce strict scientific standards:

1. **Survivorship-free universe**  
   - The Nifty 50 universe is reconstructed dynamically using historical constituent weights (`weights.csv`) to include delisted companies and avoid survivorship bias.

2. **Causal, point-in-time feature engineering**  
   - Network: A **Triangulated Maximally Filtered Graph (TMFG)** is built using only past correlation data in the training window (no future information).
   - GNN features: A **Graph Autoencoder (GAE)** trains on the TMFG to generate 16-dimensional node embeddings per stock (GNN_0 … GNN_15).
   - Baseline features: Simple technical indicators (SMA, RSI, ATR) computed with proper point-in-time logic.

3. **Rolling walk-forward validation**  
   - 3-year training window → 1-year test window.
   - Example: Train 2010–2012 → Test 2013; then Train 2011–2013 → Test 2014; repeated to cover 2013–2024.
   - For each window, a fresh strategy is discovered by Genetic Programming and then frozen for out-of-sample trading.

---

## Repository Structure

The codebase is separated into a **Research Pipeline** (training/validation) and a lightweight **Inference API** (deployment of results).
```
Evolving-Network-Alpha/
│
├── app/ # Deployment / Inference code
│ ├── gnn_app.py # Flask REST API
│ └── gnn_inference.py # Logic adapter for serving backtest results
│
├── research/ # Core R&D pipelines
│ ├── engine.py # The GNN Factory (TMFG + GNN + GP)
│ ├── engine_NO_GNN.py # Baseline Factory (technical indicators only)
│ ├── walk_forward.py # 12-year rolling validator (GNN)
│ ├── walk_forward_NO_GNN.py # 12-year rolling validator (Baseline)
│ ├── TMFG.py # Network filtering algorithm (TMFG)
│ └── vectorized_backtester.py
│
├── data/
│ ├── artifacts/ # Validated results used by the app
│ │ ├── walk_forward_returns.csv
│ │ ├── walk_forward_returns_NO_GNN.csv
│ │ └── walk_forward_signals.csv
│ └── raw/ # Raw data (excluded via .gitignore)
│
└── notebooks/
└── analysis.ipynb # Statistical validation and robustness checks
```

---

## Key Analysis and Robustness Checks

The `notebooks/analysis.ipynb` notebook contains statistical and economic validation of the strategies.

1. **Equity curve comparison**  
   - Plots the cumulative equity curves of the GNN strategy vs the Baseline.
   - The Baseline curve is smoother and slightly superior in risk-adjusted terms, indicating that the GNN often overfits its short training windows.

2. **Statistical significance**  
   - The GNN strategy’s positive Sharpe (≈ 0.87) is statistically significant (p < 0.05) versus random chance, indicating it is a real, profitable signal.
   - However, the Baseline’s Sharpe is higher, meaning the simpler strategy offers better risk-adjusted performance in this experiment.

3. **Economic realism (turnover and costs)**  
   - Annual turnover: approximately 200% (the strategy is active, not buy-and-hold).
   - Cost sensitivity analysis shows that the strategy remains profitable under reasonable transaction cost assumptions.

---

## How to Run

### 1. Installation

Clone the repository and install dependencies:

pip install -r requirements.txt


### 2. Running the Inference API

To serve the validated equity curves and summary metrics as a JSON API:

python app/gnn_app.py


Available endpoints (default host: `http://127.0.0.1:5000`):

- `GET /gnn_backtest_summary`  
  Returns a JSON object with:
  - start_date, end_date
  - GNN: CAGR, Sharpe, MaxDrawdown
  - Baseline: CAGR, Sharpe, MaxDrawdown  
  (Computed from the CSVs in `data/artifacts/`.)

- `GET /gnn_equity_curve`  
  Returns a JSON time series of:
  - Date
  - Equity_GNN
  - Equity_Baseline

### 3. Reproducing the Research

To fully re-run the 12-year walk-forward validation, you will need the raw Nifty 50 historical data (not included in this repository).

Once raw data is prepared (similar to the original local setup), you can run:

Run the GNN-based Alpha Factory
python research/walk_forward.py

Run the Baseline (No-GNN) control group
python research/walk_forward_NO_GNN.py


These scripts regenerate the rolling-window experiments and overwrite the CSV artifacts in `data/artifacts/`.

---

## Tech Stack

- **Core**: Python, pandas, NumPy
- **Graph learning**: PyTorch Geometric, NetworkX, custom TMFG implementation
- **Evolutionary AI**: DEAP (NSGA-II Genetic Programming)
- **Backtesting and validation**: Custom vectorized backtester, rolling walk-forward
- **Deployment**: Flask (REST API)
- **Statistics**: `arch`, SciPy (bootstrap and significance testing), matplotlib/seaborn for visualization

---

## Future Work

While the GNN hypothesis was falsified in this specific setup, there are several natural extensions:

1. **Monte Carlo over GP runs**  
   - Repeat the Genetic Programming discovery process 30+ times per window to obtain a distribution of Sharpe ratios for both GNN and Baseline, accounting for GP stochasticity.

2. **Longer and regime-aware windows**  
   - Extend or adapt the training window (e.g., 5–7 years, or regime-clustered periods) so the GNN can learn more stable network topologies.

3. **Cross-universe validation**  
   - Apply the same framework to other universes such as the S&P 500 to test whether the findings generalize beyond Nifty 50.

4. **Live paper-trading**  
   - Deploy the pipeline to run daily on live data and track alpha decay and stability in real time.

---

## Author and License

- Author: Rohith R

