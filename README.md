# Rokitcoder AI Trading Bot — Python

A **paper-trading-first** AI trading bot built with Python.

## What it does

- Downloads market data with `yfinance`
- Builds technical features: returns, moving averages, RSI, MACD-style momentum, volatility and volume change
- Trains a `RandomForestClassifier` to predict whether the next bar will rise
- Backtests the model with a simple long/flat strategy
- Produces a signal: BUY / HOLD / SELL
- Supports paper-trading state without placing real orders
- Keeps the broker/execution layer separate so real trading can be added later

> **Risk warning:** This project is educational software, not financial advice. The default mode is paper trading. Do not connect real funds until you have independently tested the strategy, execution, slippage, fees, data quality, risk controls and broker integration.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python bot.py --symbol SPY --period 2y --interval 1d --mode paper
```

For a different symbol:

```bash
python bot.py --symbol AAPL --period 2y --interval 1d --mode paper
```

## How the AI works

The model is trained on historical features and a binary target:

`target = 1` when the next closing price is higher than the current close.

The bot uses a chronological train/test split to avoid randomly mixing future observations into the training set.

The backtest assumes:

- Enter long when the model probability is above the buy threshold
- Exit/hold cash when probability is below the sell threshold
- No leverage
- No shorting
- Optional transaction cost in the backtest

This is deliberately simple so the code is easy to understand and extend.

## Project structure

```text
rokitcoder_ai_trading_bot/
├── bot.py
├── trading_ai.py
├── requirements.txt
├── config.example.json
├── README.md
└── .gitignore
```

## Important limitations

A high backtest score does not prove future profitability. Real trading introduces:

- slippage
- spreads
- commissions
- latency
- market gaps
- changing market regimes
- data errors
- model drift

For production use, add walk-forward validation, robust risk management, position sizing, logging, monitoring, broker-specific order handling, and a kill switch.

## Suggested Rokitcoder video

Title:

**I Built My Own AI Trading Bot With Python**

Video structure:

1. Show the finished bot and a paper-trading dashboard/result
2. Explain the problem
3. Download historical data
4. Engineer features
5. Train the AI
6. Backtest
7. Generate today's signal
8. Run paper trading
9. Explain what can go wrong
10. Improve the bot in Part 2
# Tradingaibot
