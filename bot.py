from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yfinance as yf

from trading_ai import (
    backtest,
    chronological_split,
    latest_signal,
    make_features,
    train_model,
)


def download_data(symbol: str, period: str, interval: str):
    print(f"Downloading {symbol} data: period={period}, interval={interval}")
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("No market data returned. Check the symbol, period and internet connection.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Rokitcoder AI Trading Bot")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--mode", choices=["paper"], default="paper")
    parser.add_argument("--buy-threshold", type=float, default=0.60)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--save-model", default="rokitcoder_model.pkl")
    args = parser.parse_args()

    raw = download_data(args.symbol, args.period, args.interval)
    df = make_features(raw)

    train_df, test_df = chronological_split(df)
    print(f"Rows: {len(df):,} | Train: {len(train_df):,} | Test: {len(test_df):,}")

    model = train_model(train_df)
    joblib.dump(model, args.save_model)

    result = backtest(
        df,
        model,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        transaction_cost=args.transaction_cost,
        initial_cash=args.initial_cash,
    )

    signal = latest_signal(df, model)

    print("\n=== BACKTEST ===")
    print(f"Final portfolio value: ${result.final_value:,.2f}")
    print(f"Strategy return:       {result.total_return:.2%}")
    print(f"Buy & hold return:     {result.buy_and_hold_return:.2%}")
    print(f"Max drawdown:           {result.max_drawdown:.2%}")
    print(f"Test accuracy:          {result.test_accuracy:.2%}")
    print(f"Test precision:         {result.test_precision:.2%}")

    print("\n=== LATEST PAPER SIGNAL ===")
    print(f"Symbol:                 {args.symbol.upper()}")
    print(f"Close:                  ${signal['close']:,.2f}")
    print(f"AI probability UP:      {signal['probability_up']:.2%}")
    print(f"Signal:                 {signal['signal']}")

    print("\nPaper-trading mode only. No real order has been sent.")


if __name__ == "__main__":
    main()
