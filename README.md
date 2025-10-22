#Statistical Arbitrage Strategy (StatArb Backtester)

A quantitative trading backtester that identifies **cointegrated stock pairs** and simulates **statistical arbitrage (pairs trading)** strategies using **z-score signals**, **volatility-adjusted risk sizing**, and **Sharpe-based performance metrics**.

---

## 🧠 Overview

This project implements a **Statistical Arbitrage Backtester** — a Python framework designed to detect mean-reverting relationships between assets and simulate market-neutral trading strategies.

The system uses:
- **Cointegration tests** to find stable long-term price relationships.
- **Z-score entry/exit rules** for mean reversion signals.
- **Volatility-based position sizing** for dynamic risk control.
- **Performance metrics and visual dashboards** for strategy evaluation.

---

## ⚙️ Features

-  **Cointegration Detection:** Automatically finds statistically significant pairs using Engle-Granger tests.  
-  **Hedge Ratio Estimation:** Computes optimal hedge ratios via OLS regression.  
-  **Z-score Trading Logic:** Generates entry/exit points based on spread deviations.  
-  **Dynamic Risk Management:** Adjusts trade sizes according to volatility and target risk levels.  
-  **Performance Metrics:**  
  - Sharpe ratio  
  - Total return  
  - Max drawdown  
  - Win rate  
-  **S&P 500 Benchmarking:**  
  - Compares your strategy vs. the S&P 500  
  - Calculates annualized **alpha** (excess return)  
-  **Visual Dashboard:**  
  - Stock prices  
  - Spread & thresholds  
  - Z-score movement  
  - Equity curve (strategy vs. buy-and-hold vs. S&P 500)

---

##  Installation


## 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows

## 3. Install dependencies
pip install -r requirements.txt
 
# To run 
python stat_arb_backtester.py

