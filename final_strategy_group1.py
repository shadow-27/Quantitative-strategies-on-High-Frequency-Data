import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from functions.position_VB import positionVB


quarters = [
    "2023_Q1",
    "2023_Q3",
    "2023_Q4",
    "2024_Q2",
    "2024_Q4",
    "2025_Q1",
    "2025_Q2",
]


def mySR(x, scale=252):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return np.sqrt(scale) * mu / sd


def calmar_from_daily_pnl(pnl_daily, scale=252):
    pnl = pd.Series(pnl_daily).astype(float).fillna(0.0)
    equity = pnl.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = -drawdown.min()  # positive number
    ann_ret = pnl.mean() * scale
    if max_dd == 0:
        return np.nan
    return ann_ret / max_dd


# Final selected Group 1 strategy: Volatility Breakout 2.2 (fixed parameters)
# NQ: MOM (signalEMA=20, slowEMA=60, volat_sd=30, m=2.0)
# SP: MR  (signalEMA=90, slowEMA=180, volat_sd=60, m=1.0)

contracts = {
    "NQ": {"ptval": 20, "tcost": 12, "signalEMA": 20, "slowEMA": 60, "volat_sd": 30, "m": 2.0, "side": "mom"},
    "SP": {"ptval": 50, "tcost": 12, "signalEMA": 90, "slowEMA": 180, "volat_sd": 60, "m": 1.0, "side": "mr"},
}


summary_rows = []

OUTPUT_DIR = Path("outputs") / "group1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for quarter in quarters:
    data1 = pd.read_parquet(f"data/data1_{quarter}.parquet")
    data1["datetime"] = pd.to_datetime(data1["datetime"])
    data1 = data1.set_index("datetime").sort_index()

    # Assumption 1: exclude first/last 10 minutes from calculations (set to NaN)
    data1.loc[data1.between_time("9:31", "9:40").index] = np.nan
    data1.loc[data1.between_time("15:51", "16:00").index] = np.nan

    # Assumption 2: create pos_flat (1 = must be flat)
    pos_flat = np.zeros(len(data1), dtype=float)
    pos_flat[data1.index.time <= pd.to_datetime("9:55").time()] = 1
    pos_flat[data1.index.time >= pd.to_datetime("15:40").time()] = 1

    for sym in ["NQ", "SP"]:
        p = data1[sym]
        cfg = contracts[sym]

        signalEMA = int(cfg["signalEMA"])
        slowEMA = int(cfg["slowEMA"])
        volat_sd = int(cfg["volat_sd"])
        m = float(cfg["m"])

        sEMA = p.ewm(span=signalEMA, min_periods=max(1, signalEMA // 2), adjust=False).mean()
        lEMA = p.ewm(span=slowEMA, min_periods=max(1, slowEMA // 2), adjust=False).mean()
        vSD = p.rolling(window=volat_sd, min_periods=max(1, volat_sd // 2)).std()

        sEMA[p.isna()] = np.nan
        lEMA[p.isna()] = np.nan
        vSD[p.isna()] = np.nan

        upper = lEMA + m * vSD
        lower = lEMA - m * vSD

        # Positions from the lab helper (uses t-1 signal/bands internally)
        pos = np.asarray(
            positionVB(
                signal=sEMA.to_numpy(),
                lower=lower.to_numpy(),
                upper=upper.to_numpy(),
                pos_flat=pos_flat,
                strategy=cfg["side"],
            ),
            dtype=float,
        )
        pos = np.nan_to_num(pos, nan=0.0)

        # Transactions = absolute changes in position
        ntrans = np.abs(np.diff(pos, prepend=0.0))

        # Minute price change
        dpx = p.diff().fillna(0.0).to_numpy()

        # PnL in USD
        pnl_gross = pos * dpx * cfg["ptval"]
        pnl_net = pnl_gross - ntrans * cfg["tcost"]

        # ---- Per-contract daily aggregation + metrics ----
        idx = data1.index
        pnl_gross_d_sym = pd.Series(pnl_gross, index=idx).groupby(idx.date).sum()
        pnl_net_d_sym = pd.Series(pnl_net, index=idx).groupby(idx.date).sum()
        ntrans_d_sym = pd.Series(ntrans, index=idx).groupby(idx.date).sum()

        gross_SR = mySR(pnl_gross_d_sym, scale=252)
        net_SR = mySR(pnl_net_d_sym, scale=252)
        gross_PnL = float(pnl_gross_d_sym.sum())
        net_PnL = float(pnl_net_d_sym.sum())
        gross_CR = calmar_from_daily_pnl(pnl_gross_d_sym)
        net_CR = calmar_from_daily_pnl(pnl_net_d_sym)
        av_daily_ntrans = float(ntrans_d_sym.mean())
        stat = (net_SR - 0.5) * np.maximum(0, np.log(np.abs(net_PnL / 1000)))

        summary_rows.append(
            {
                "quarter": quarter,
                "sym": sym,
                "gross_SR": gross_SR,
                "net_SR": net_SR,
                "gross_PnL": gross_PnL,
                "net_PnL": net_PnL,
                "gross_CR": gross_CR,
                "net_CR": net_CR,
                "av_daily_ntrans": av_daily_ntrans,
                "stat": stat,
            }
        )

        # Equity line (daily cumulative) for each contract
        plt.figure(figsize=(12, 6))
        plt.plot(pnl_gross_d_sym.fillna(0).cumsum(), label="Gross PnL", color="blue")
        plt.plot(pnl_net_d_sym.fillna(0).cumsum(), label="Net PnL", color="red")
        plt.title(f"Group 1 Volatility Breakout 2.2 ({sym}): Cumulative PnL ({quarter})")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / f"data1_{quarter}_{sym}.png", dpi=300, bbox_inches="tight")
        plt.close()


summary_data1_all_quarters = pd.DataFrame(summary_rows)
summary_data1_all_quarters.to_csv(OUTPUT_DIR / "summary_data1_all_quarters.csv", index=False)