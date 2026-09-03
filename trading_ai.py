from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score


FEATURES = [
    "return_1",
    "return_5",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "rsi_14",
    "momentum_10",
    "volatility_10",
    "volume_change",
]


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    final_value: float
    total_return: float
    buy_and_hold_return: float
    max_drawdown: float
    test_accuracy: float
    test_precision: float


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    df["return_1"] = close.pct_change(1)
    df["return_5"] = close.pct_change(5)

    for n in (10, 20, 50):
        sma = close.rolling(n).mean()
        df[f"sma_{n}_ratio"] = close / sma - 1

    df["rsi_14"] = rsi(close, 14) / 100.0
    df["momentum_10"] = close / close.shift(10) - 1
    df["volatility_10"] = close.pct_change().rolling(10).std()
    df["volume_change"] = volume.pct_change()

    # Next-bar direction is the prediction target.
    df["target"] = (close.shift(-1) > close).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"])
    return model


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_fraction)
    if cut < 100 or len(df) - cut < 30:
        raise ValueError("Not enough data for a reliable train/test split.")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def backtest(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.45,
    transaction_cost: float = 0.001,
    initial_cash: float = 10_000,
) -> BacktestResult:
    train_df, test_df = chronological_split(df)

    probs = model.predict_proba(test_df[FEATURES])[:, 1]
    predictions = (probs >= 0.50).astype(int)

    accuracy = accuracy_score(test_df["target"], predictions)
    precision = precision_score(test_df["target"], predictions, zero_division=0)

    cash = initial_cash
    shares = 0.0
    previous_position = 0
    equity = []

    closes = test_df["Close"].astype(float).to_numpy()

    for i, price in enumerate(closes):
        p_up = float(probs[i])

        if p_up >= buy_threshold:
            desired_position = 1
        elif p_up <= sell_threshold:
            desired_position = 0
        else:
            desired_position = previous_position

        if desired_position != previous_position:
            if desired_position == 1 and shares == 0:
                # Invest all available cash.
                fee = cash * transaction_cost
                investable = cash - fee
                shares = investable / price
                cash = 0.0
            elif desired_position == 0 and shares > 0:
                gross = shares * price
                fee = gross * transaction_cost
                cash = gross - fee
                shares = 0.0

        equity.append(cash + shares * price)
        previous_position = desired_position

    equity_curve = pd.Series(equity, index=test_df.index, name="equity")
    final_value = float(equity_curve.iloc[-1])
    total_return = final_value / initial_cash - 1

    first_price = float(closes[0])
    buy_hold = initial_cash * (closes[-1] / first_price)
    buy_and_hold_return = buy_hold / initial_cash - 1

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min())

    return BacktestResult(
        equity_curve=equity_curve,
        final_value=final_value,
        total_return=total_return,
        buy_and_hold_return=buy_and_hold_return,
        max_drawdown=max_drawdown,
        test_accuracy=float(accuracy),
        test_precision=float(precision),
    )


def latest_signal(df: pd.DataFrame, model: RandomForestClassifier) -> Dict[str, float | str]:
    latest = df.iloc[[-1]]
    probability_up = float(model.predict_proba(latest[FEATURES])[0, 1])

    if probability_up >= 0.60:
        signal = "BUY"
    elif probability_up <= 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "probability_up": probability_up,
        "close": float(latest["Close"].iloc[0]),
    }
