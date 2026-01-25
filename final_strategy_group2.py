import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
CURRENT_DIR = Path(__file__).parent

# Define Paths
IS_DIR = CURRENT_DIR / "data"
OOS_DIR = CURRENT_DIR / "data" / "data_oos"
OUTPUT_DIR = CURRENT_DIR / "outputs" / "group2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ALL 12 QUARTERS (IS + OOS)
QUARTERS = [
    '2023_Q1', '2023_Q2', '2023_Q3', '2023_Q4',
    '2024_Q1', '2024_Q2', '2024_Q3', '2024_Q4',
    '2025_Q1', '2025_Q2', '2025_Q3', '2025_Q4'
]

# ASSET SPECIFICATIONS
ASSETS_CONFIG = {
    'XAU': {'cost': 15.0, 'point_value': 100.0},
    'XAG': {'cost': 10.0, 'point_value': 5000.0}
}

# WINNING PARAMETERS (Fixed)
PARAMS_XAU = {'window': 288, 'vol_threshold': 0.5, 'session': 'full'}
PARAMS_XAG = {'window': 144, 'vol_threshold': 0.05, 'session': 'us_only'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_file_path(quarter):
    """Finds the parquet file in either IS or OOS folder"""
    filename = f"data2_{quarter}.parquet"
    
    # Check In-Sample Folder
    p1 = IS_DIR / filename
    if p1.exists(): return p1
    
    # Check Out-of-Sample Folder
    p2 = OOS_DIR / filename
    if p2.exists(): return p2
    
    return None

def load_all_data():
    """Loads ALL files from both folders to ensure indicator continuity"""
    all_files = []
    
    # Grab from IS
    if IS_DIR.exists():
        all_files.extend(list(IS_DIR.glob("data2_*.parquet")))
    
    # Grab from OOS
    if OOS_DIR.exists():
        all_files.extend(list(OOS_DIR.glob("data2_*.parquet")))
        
    df_list = []
    for f in all_files:
        try:
            q_df = pd.read_parquet(f)
            if 'datetime' in q_df.columns:
                q_df['datetime'] = pd.to_datetime(q_df['datetime'])
                q_df.set_index('datetime', inplace=True)
            
            # UTC -> CET
            if q_df.index.tz is None: q_df.index = q_df.index.tz_localize('UTC')
            q_df.index = q_df.index.tz_convert('Europe/Berlin')
            df_list.append(q_df)
        except: pass
    
    if not df_list: return pd.DataFrame()
    
    # Concat and Drop Duplicates (in case of overlapping files)
    return pd.concat(df_list).sort_index().drop_duplicates()

def mySR(x, scale=252):
    std = np.nanstd(x)
    if std == 0: return np.nan
    return np.sqrt(scale) * np.nanmean(x) / std

def calculate_calmar(pnl_series, annual_factor=1.0):
    cum_pnl = pnl_series.cumsum()
    max_dd = (cum_pnl - cum_pnl.expanding().max()).min()
    if max_dd == 0: return np.nan
    return (pnl_series.sum() / annual_factor) / abs(max_dd)

def get_stats_dict(quarter, sym, daily_gross, daily_net, daily_trades, days_total):
    gross_SR = mySR(daily_gross)
    net_SR = mySR(daily_net)
    gross_PnL = daily_gross.sum()
    net_PnL = daily_net.sum()
    
    frac_year = max(days_total / 365.25, 0.2)
    gross_CR = calculate_calmar(daily_gross, frac_year)
    net_CR = calculate_calmar(daily_net, frac_year)
    av_daily_ntrans = daily_trades.mean()
    
    try:
        val = abs(net_PnL/1000)
        log_val = np.log(val) if val > 0 else 0
        stat = (net_SR - 0.5) * max(0, log_val)
    except: stat = 0

    return {
        'quarter': quarter, 'sym': sym,
        'gross_SR': gross_SR, 'net_SR': net_SR,
        'gross_PnL': gross_PnL, 'net_PnL': net_PnL,
        'gross_CR': gross_CR, 'net_CR': net_CR,
        'av_daily_ntrans': av_daily_ntrans, 'stat': stat
    }

def run_breakout_engine(df_wide, asset_name, asset_conf, params):
    # Prepare Series
    col = [c for c in df_wide.columns if asset_name in c]
    if not col: return None, None, None
    
    df = df_wide[[col[0]]].copy()
    df.columns = ['close']
    df.dropna(inplace=True) 
    
    close = df['close'].values
    times = df.index.time
    t_vals = np.array([t.hour * 100 + t.minute for t in times])
    
    position = np.zeros(len(df))
    current_pos = 0
    
    # Params
    session_mode = params.get('session', 'full')
    entry_start = 1300 if session_mode == 'us_only' else 0
    entry_end = 1630 if session_mode == 'us_only' else 1650
    window = params['window']
    vol_th = params['vol_threshold']
    vol_win = 20

    # Indicators
    roll_max = df['close'].rolling(window).max().shift(1).values
    roll_min = df['close'].rolling(window).min().shift(1).values
    vol = df['close'].diff().abs().rolling(vol_win).mean().values
    
    for i in range(1, len(df)):
        t = t_vals[i]
        
        # 1. Strict Exit (16:50 CET)
        if 1650 <= t < 1810:
            current_pos = 0; position[i] = 0; continue
            
        # 2. Session Filter
        can_enter = True
        if session_mode == 'us_only':
            if not (entry_start <= t < entry_end): can_enter = False
        
        # 3. Logic
        if current_pos == 0:
            if can_enter and vol[i] > vol_th:
                if close[i] > roll_max[i]: current_pos = 1
                elif close[i] < roll_min[i]: current_pos = -1
        elif current_pos == 1:
            if close[i] < roll_min[i]: current_pos = -1
        elif current_pos == -1:
            if close[i] > roll_max[i]: current_pos = 1
            
        position[i] = current_pos

    # PnL
    pos_series = pd.Series(position, index=df.index)
    gross_pnl = pos_series.shift(1).fillna(0) * df['close'].diff() * asset_conf['point_value']
    trades = pos_series.diff().abs().fillna(0)
    net_pnl = gross_pnl - (trades * asset_conf['cost'])
    
    return gross_pnl, net_pnl, trades

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

print("1. Loading ALL Data (IS + OOS)...")
df_all = load_all_data()

if not df_all.empty:
    print(f"   Loaded data range: {df_all.index[0]} to {df_all.index[-1]}")
    print("2. Running Strategies on Full History...")
    
    # Run Strategies on continuous data
    xau_gross, xau_net, xau_trades = run_breakout_engine(df_all, 'XAU', ASSETS_CONFIG['XAU'], PARAMS_XAU)
    xag_gross, xag_net, xag_trades = run_breakout_engine(df_all, 'XAG', ASSETS_CONFIG['XAG'], PARAMS_XAG)
    
    # Resample to Daily
    xau_d_gross = xau_gross.resample('D').sum(); xau_d_net = xau_net.resample('D').sum()
    xau_d_trades = xau_trades.resample('D').sum()
    
    xag_d_gross = xag_gross.resample('D').sum(); xag_d_net = xag_net.resample('D').sum()
    xag_d_trades = xag_trades.resample('D').sum()
    
    summary_rows = []
    
    print(f"3. Slicing Quarters and Saving to {OUTPUT_DIR}...")
    
    # We need specific start/end dates for each quarter to slice accurately
    # Using the file content itself to determine the exact range for that quarter
    
    for quarter in QUARTERS:
        file_path = get_file_path(quarter)
        if not file_path:
            print(f"   [Missing] {quarter} not found in IS or OOS folders.")
            continue
            
        # Load just to get date range
        q_df = pd.read_parquet(file_path)
        if 'datetime' in q_df.columns: q_df['datetime'] = pd.to_datetime(q_df['datetime']); q_df.set_index('datetime', inplace=True)
        if q_df.index.tz is None: q_df.index = q_df.index.tz_localize('UTC')
        q_df.index = q_df.index.tz_convert('Europe/Berlin')
        
        q_start = q_df.index[0]
        q_end = q_df.index[-1]
        days = (q_end - q_start).days
        
        # Slice XAU
        s_xau_g = xau_d_gross.loc[q_start:q_end]
        s_xau_n = xau_d_net.loc[q_start:q_end]
        s_xau_t = xau_d_trades.loc[q_start:q_end]
        
        # Slice XAG
        s_xag_g = xag_d_gross.loc[q_start:q_end]
        s_xag_n = xag_d_net.loc[q_start:q_end]
        s_xag_t = xag_d_trades.loc[q_start:q_end]
        
        # Append Stats
        if len(s_xau_g) > 0:
            summary_rows.append(get_stats_dict(quarter, 'XAU', s_xau_g, s_xau_n, s_xau_t, days))
        if len(s_xag_g) > 0:
            summary_rows.append(get_stats_dict(quarter, 'XAG', s_xag_g, s_xag_n, s_xag_t, days))
            
        # Calculate Combined Portfolio for PNG
        # Align by index for this specific quarter slice
        comb_gross = s_xau_g.add(s_xag_g, fill_value=0)
        comb_net = s_xau_n.add(s_xag_n, fill_value=0)
        
        # Plot Combined
        plt.figure(figsize=(12, 6))
        plt.plot(comb_gross.cumsum(), label='Gross PnL', color='blue', linewidth=1.5)
        plt.plot(comb_net.cumsum(), label='Net PnL', color='red', linewidth=1.5)
        
        net_val = comb_net.sum()
        plt.title(f"XAU+XAG: {quarter}\nNet PnL: ${net_val:,.0f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel("Date")
        plt.ylabel("PnL (USD)")
        
        plt.savefig(OUTPUT_DIR / f"data2_combined_{quarter}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Save Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        cols = ['quarter', 'sym', 'gross_SR', 'net_SR', 'gross_PnL', 'net_PnL', 'gross_CR', 'net_CR', 'av_daily_ntrans', 'stat']
        summary_df = summary_df[cols]
        summary_df.to_csv(OUTPUT_DIR / 'summary_data2_all_quarters.csv', index=False)
        print("CSV saved.")
        
    print("Group 2 Analysis Complete.")
else:
    print("CRITICAL: No data loaded. Check paths.")