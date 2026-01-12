import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input/Output Path
CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR / "data"
OUTPUT_PATH = CURRENT_DIR / "outputs" / "group2"

QUARTERS = [
    '2023_Q1', '2023_Q3', '2023_Q4',
    '2024_Q2', '2024_Q4',
    '2025_Q1', '2025_Q2'
]

# ASSET SPECIFICATIONS
ASSETS_CONFIG = {
    'XAU': {'cost': 15.0, 'point_value': 100.0},
    'XAG': {'cost': 10.0, 'point_value': 5000.0}
}

# WINNING PARAMETERS
PARAMS_XAU = {'window': 288, 'vol_threshold': 0.5, 'session': 'full'}
PARAMS_XAG = {'window': 144, 'vol_threshold': 0.05, 'session': 'us_only'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def mySR(x, scale=252):
    """Annualized Sharpe Ratio"""
    std = np.nanstd(x)
    if std == 0: return 0
    return np.sqrt(scale) * np.nanmean(x) / std

def calculate_calmar(pnl_series, annual_factor=1.0):
    """Calmar Ratio"""
    cum_pnl = pnl_series.cumsum()
    max_dd = (cum_pnl - cum_pnl.expanding().max()).min()
    if max_dd == 0: return 0
    return (pnl_series.sum() / annual_factor) / abs(max_dd)

def get_stats_dict(quarter, sym, daily_gross, daily_net, daily_trades, days_total):
    """Calculates row stats for a specific asset"""
    gross_SR = mySR(daily_gross)
    net_SR = mySR(daily_net)

    gross_PnL = daily_gross.sum()
    net_PnL = daily_net.sum()

    frac_year = max(days_total / 365.25, 0.2)

    gross_CR = calculate_calmar(daily_gross, frac_year)
    net_CR = calculate_calmar(daily_net, frac_year)

    av_daily_ntrans = daily_trades.mean()

    try:
        # Avoid log(0) error
        val = abs(net_PnL/1000)
        log_val = np.log(val) if val > 0 else 0
        stat = (net_SR - 0.5) * max(0, log_val)
    except:
        stat = 0

    return {
        'quarter': quarter,
        'sym': sym,
        'gross_SR': gross_SR,
        'net_SR': net_SR,
        'gross_PnL': gross_PnL,
        'net_PnL': net_PnL,
        'gross_CR': gross_CR,
        'net_CR': net_CR,
        'av_daily_ntrans': av_daily_ntrans,
        'stat': stat
    }

def run_breakout_engine(df, asset_conf, params):
    """Applies Volatility Breakout with Session Filter & Strict Exit"""
    close = df['close'].values
    times = df.index.time
    t_vals = np.array([t.hour * 100 + t.minute for t in times])

    position = np.zeros(len(df))
    current_pos = 0

    session_mode = params.get('session', 'full')
    entry_start = 1300 if session_mode == 'us_only' else 0
    entry_end = 1630 if session_mode == 'us_only' else 1650

    window = params['window']
    vol_th = params['vol_threshold']
    vol_win = 20

    roll_max = df['close'].rolling(window).max().shift(1).values
    roll_min = df['close'].rolling(window).min().shift(1).values
    vol = df['close'].diff().abs().rolling(vol_win).mean().values

    for i in range(1, len(df)):
        t = t_vals[i]

        # 1. STRICT FORCED EXIT (16:50 CET)
        if 1650 <= t < 1810:
            current_pos = 0; position[i] = 0; continue

        # 2. SESSION ENTRY FILTER
        can_enter = True
        if session_mode == 'us_only':
            if not (entry_start <= t < entry_end): can_enter = False

        # 3. STRATEGY LOGIC
        if current_pos == 0:
            if can_enter and vol[i] > vol_th:
                if close[i] > roll_max[i]: current_pos = 1
                elif close[i] < roll_min[i]: current_pos = -1
        elif current_pos == 1:
            if close[i] < roll_min[i]: current_pos = -1
        elif current_pos == -1:
            if close[i] > roll_max[i]: current_pos = 1

        position[i] = current_pos

    pos_series = pd.Series(position, index=df.index)
    price_change = df['close'].diff()
    gross_pnl = pos_series.shift(1).fillna(0) * price_change * asset_conf['point_value']
    trades = pos_series.diff().abs().fillna(0)
    net_pnl = gross_pnl - (trades * asset_conf['cost'])

    return gross_pnl, net_pnl, trades

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================

summary_rows = []

print(f"Running Group 2 Strategy Engine...")
print(f"Saving outputs to: {OUTPUT_PATH}")

for quarter in QUARTERS:
    # 1. Load Data
    f_name = f"data2_{quarter}.parquet"
    file_path = os.path.join(DATA_PATH, f_name)

    if not os.path.exists(file_path):
        print(f"Skipping {quarter}, file not found.")
        continue

    df = pd.read_parquet(file_path)

    # 2. Timezone Fix
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Europe/Berlin')

    # 3. Extract Assets
    cols = df.columns
    col_xau = [c for c in cols if 'XAU' in c]
    col_xag = [c for c in cols if 'XAG' in c]

    if not col_xau or not col_xag: continue

    df_xau = df[[col_xau[0]]].copy(); df_xau.columns = ['close']; df_xau.dropna(inplace=True)
    df_xag = df[[col_xag[0]]].copy(); df_xag.columns = ['close']; df_xag.dropna(inplace=True)

    # 4. Run Strategy Logic
    g_xau, n_xau, t_xau = run_breakout_engine(df_xau, ASSETS_CONFIG['XAU'], PARAMS_XAU)
    g_xag, n_xag, t_xag = run_breakout_engine(df_xag, ASSETS_CONFIG['XAG'], PARAMS_XAG)

    days_total = (df.index[-1] - df.index[0]).days

    # 5. XAU STATS & PLOT
    d_g_xau = g_xau.resample('D').sum()
    d_n_xau = n_xau.resample('D').sum()
    d_t_xau = t_xau.resample('D').sum()

    # Append row for XAU
    summary_rows.append(get_stats_dict(quarter, 'XAU', d_g_xau, d_n_xau, d_t_xau, days_total))

    # Save XAU Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(d_g_xau.fillna(0)), label='Gross PnL', color='blue')
    plt.plot(np.cumsum(d_n_xau.fillna(0)), label='Net PnL', color='red')
    plt.title(f'Gold (XAU) Cumulative PnL - {quarter}')
    plt.legend(); plt.grid(axis='x')
    plt.savefig(os.path.join(OUTPUT_PATH, f"data2_XAU_{quarter}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 6. XAG STATS & PLOT
    d_g_xag = g_xag.resample('D').sum()
    d_n_xag = n_xag.resample('D').sum()
    d_t_xag = t_xag.resample('D').sum()

    # Append row for XAG
    summary_rows.append(get_stats_dict(quarter, 'XAG', d_g_xag, d_n_xag, d_t_xag, days_total))

    # Save XAG Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(d_g_xag.fillna(0)), label='Gross PnL', color='blue')
    plt.plot(np.cumsum(d_n_xag.fillna(0)), label='Net PnL', color='red')
    plt.title(f'Silver (XAG) Cumulative PnL - {quarter}')
    plt.legend(); plt.grid(axis='x')
    plt.savefig(os.path.join(OUTPUT_PATH, f"data2_XAG_{quarter}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Clean up
    del df, df_xau, df_xag, g_xau, g_xag, n_xau, n_xag

# 7. CREATE FINAL DATAFRAME & SAVE
summary_df = pd.DataFrame(summary_rows)
# Reorder columns to match requested format exactly
cols = ['quarter', 'sym', 'gross_SR', 'net_SR', 'gross_PnL', 'net_PnL', 'gross_CR', 'net_CR', 'av_daily_ntrans', 'stat']
summary_df = summary_df[cols]

csv_path = os.path.join(OUTPUT_PATH, 'summary_data2_all_quarters.csv')
summary_df.to_csv(csv_path, index=False)

print(f"Processing complete.")
print(f"Table saved to: {csv_path}")
print(f"Plots saved to: {OUTPUT_PATH}")
print("\nPreview of Table:")
print(summary_df.head(4).to_string(index=False))