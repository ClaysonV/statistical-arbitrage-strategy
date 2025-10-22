import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import time, os

# ==========================
# CONFIGURATION
# ==========================
PAIRS = [
    ("KO", "PEP"),
    ("XOM", "CVX"),
    ("V", "MA"),
    ("JPM", "BAC"),
    ("AAPL", "MSFT")
]

START_DATE = "2022-01-01"
END_DATE = "2024-01-01"
ENTRY_Z = 1.5
EXIT_Z = 0.3
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000
RISK_MULTIPLIER = 5      # Base exposure
ROLLING_VOL_WINDOW = 20  # Days to calculate spread volatility
TARGET_VOL = 1.0         # Target “risk unit” per trade
REFRESH_DELAY = 0.02     # seconds for live terminal updates

# ==========================
# FUNCTION: BACKTEST PAIR
# ==========================
def backtest_pair(t1, t2, show_live=False):
    print(f"\n📊 Testing pair: {t1}/{t2}")

    data = yf.download([t1, t2], start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].dropna()
    if data.shape[0] < 100:
        print("❌ Not enough data.")
        return None

    # Cointegration
    score, pvalue, _ = coint(data[t1], data[t2])
    if pvalue > 0.05:
        print(f"⚠️ Not cointegrated (p={pvalue:.3f})")
        return None

    # Hedge ratio
    X = sm.add_constant(data[t2])
    model = sm.OLS(data[t1], X).fit()
    hedge_ratio = model.params[t2]

    # Spread + Z-score
    data["Spread"] = data[t1] - hedge_ratio * data[t2]
    data["Z"] = (data["Spread"] - data["Spread"].mean()) / data["Spread"].std()
    data["SpreadVol"] = data["Spread"].rolling(ROLLING_VOL_WINDOW).std().fillna(data["Spread"].std())

    # Strategy variables
    position = 0
    entry_price = 0
    cash = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    trade_markers = []
    pnl_list = []

    for i in range(1, len(data)):
        z = data["Z"].iloc[i]
        spread = data["Spread"].iloc[i]
        vol = data["SpreadVol"].iloc[i]
        date = data.index[i]

        # Dynamic position size
        position_size = TARGET_VOL / vol if vol > 0 else 1
        position_size *= RISK_MULTIPLIER

        # Entry
        if z < -ENTRY_Z and position == 0:
            position = 1 * position_size
            entry_price = spread
            trade_markers.append((i, spread, "LONG"))
            if show_live:
                print(f"🟢 [{date.date()}] LONG spread | Z={z:.2f} | PosSize={position_size:.2f}")

        elif z > ENTRY_Z and position == 0:
            position = -1 * position_size
            entry_price = spread
            trade_markers.append((i, spread, "SHORT"))
            if show_live:
                print(f"🔴 [{date.date()}] SHORT spread | Z={z:.2f} | PosSize={position_size:.2f}")

        # Exit
        elif abs(z) < EXIT_Z and position != 0:
            pnl = position * (spread - entry_price)
            cash += pnl - abs(pnl) * TRANSACTION_COST
            trade_markers.append((i, spread, "EXIT"))
            pnl_list.append(pnl - abs(pnl)*TRANSACTION_COST)
            if show_live:
                print(f"⚪ [{date.date()}] EXIT | Z={z:.2f} | PnL={pnl:.2f} | Equity=${cash:,.2f}")
            position = 0

        # Track equity
        unrealized = position * (spread - entry_price) if position != 0 else 0
        total_equity = cash + unrealized
        equity_curve.append(total_equity)

        # Live terminal update
        if show_live:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"📆 {date.date()} | Z={z:.2f} | Position={position}")
            print(f"💰 Equity: ${total_equity:,.2f}")
            print(f"Progress: {i}/{len(data)} ({(i/len(data))*100:.1f}%)")
            time.sleep(REFRESH_DELAY)

    # Ensure equity_curve matches data length
    eq_array = np.array(equity_curve)
    if len(eq_array) < len(data):
        eq_array = np.pad(eq_array, (0, len(data)-len(eq_array)), constant_values=INITIAL_CAPITAL)

    # Performance metrics
    returns = pd.Series(np.diff(eq_array) / eq_array[:-1])
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    total_return = (eq_array[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_drawdown = ((np.maximum.accumulate(eq_array) - eq_array)/np.maximum.accumulate(eq_array)).max()
    total_trades = len(pnl_list)
    win_rate = np.mean([1 if x>0 else 0 for x in pnl_list]) if pnl_list else 0

    if show_live:
        print(f"\n📈 Summary for {t1}/{t2}:")
        print(f"Sharpe: {sharpe:.2f} | Total Return: {total_return*100:.2f}%")
        print(f"Max Drawdown: {max_drawdown*100:.2f}% | Trades: {total_trades} | Win Rate: {win_rate*100:.1f}%")

    return {
        "pair": (t1, t2),
        "pvalue": pvalue,
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "equity_curve": eq_array,
        "dates": data.index,
        "spread": data["Spread"],
        "z": data["Z"],
        "trade_markers": trade_markers,
        "pnl_list": pnl_list,
        "hedge_ratio": hedge_ratio,
        "price_data": data[[t1, t2]]
    }

# ==========================
# TEST ALL PAIRS
# ==========================
results = []
for t1, t2 in PAIRS:
    r = backtest_pair(t1, t2)
    if r:
        results.append(r)

if not results:
    print("❌ No viable pairs found.")
    exit()

best_pair = max(results, key=lambda x: x["sharpe"])
t1, t2 = best_pair["pair"]
print(f"\n🏆 Best pair: {t1}/{t2} | Sharpe={best_pair['sharpe']:.2f} | Return={best_pair['total_return']*100:.2f}%")

# ==========================
# RUN LIVE SIMULATION
# ==========================
print("\n🔁 Running live simulation on best pair...")
time.sleep(2)
live_result = backtest_pair(t1, t2, show_live=True)

# ==========================
# VISUAL DASHBOARD
# ==========================
fig, axs = plt.subplots(4, 1, figsize=(14,12), sharex=True)

# --- Stock Prices ---
axs[0].plot(live_result["dates"], live_result["price_data"][t1], label=t1)
axs[0].plot(live_result["dates"], live_result["price_data"][t2], label=t2)
axs[0].set_ylabel("Price ($)")
axs[0].legend()
axs[0].set_title(f"{t1}/{t2} Statistical Arbitrage Strategy")

# --- Spread + Thresholds ---
axs[1].plot(live_result["dates"], live_result["spread"], label='Spread', color='purple')
axs[1].axhline(ENTRY_Z * live_result["spread"].std(), color='red', linestyle='--', label='Short Entry')
axs[1].axhline(-ENTRY_Z * live_result["spread"].std(), color='green', linestyle='--', label='Long Entry')
axs[1].axhline(EXIT_Z * live_result["spread"].std(), color='grey', linestyle=':', label='Exit Zone')
axs[1].axhline(-EXIT_Z * live_result["spread"].std(), color='grey', linestyle=':')
for idx, val, t in live_result["trade_markers"]:
    color = "green" if t=="LONG" else "red" if t=="SHORT" else "black"
    axs[1].scatter(live_result["dates"][idx], val, color=color, s=50)
axs[1].set_ylabel("Spread ($)")
axs[1].legend()

# --- Z-score ---
axs[2].plot(live_result["dates"], live_result["z"], label='Z-score', color='orange')
axs[2].axhline(ENTRY_Z, color='red', linestyle='--')
axs[2].axhline(-ENTRY_Z, color='green', linestyle='--')
axs[2].axhline(EXIT_Z, color='grey', linestyle=':')
axs[2].axhline(-EXIT_Z, color='grey', linestyle=':')
axs[2].set_ylabel("Z-score")
axs[2].legend()

# --- Equity Curve ---
axs[3].plot(live_result["dates"], live_result["equity_curve"], label="Strategy Equity", color="cyan")
for t in [t1, t2]:
    bh_returns = live_result["price_data"][t] / live_result["price_data"][t].iloc[0] * INITIAL_CAPITAL
    axs[3].plot(live_result["dates"], bh_returns, linestyle="--", label=f"{t} Buy & Hold")
axs[3].set_ylabel("Equity ($)")
axs[3].legend()
axs[3].set_xlabel("Date")

plt.tight_layout()
plt.show()
