"""
Credit Spread Signal Robustness Test

Tests the stability of credit spread signals across different parameter choices:
- MA smoothing periods
- Z-score lookback windows
- ROC calculation periods

Goal: Determine if findings are robust or curve-fitted
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT SPREAD ROBUSTNESS TEST")
print("=" * 80)
print("\nDownloading data...")

# Download data
start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

iei = yf.download('IEI', start=start_date, end=end_date, progress=False)['Adj Close']
hyg = yf.download('HYG', start=start_date, end=end_date, progress=False)['Adj Close']
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

data = pd.DataFrame({
    'IEI': iei,
    'HYG': hyg,
    'SPY_Close': spy['Adj Close'],
    'SPY_Open': spy['Open']
})

data = data.dropna()
data['Spread'] = data['IEI'] / data['HYG']

# Forward returns
data['Fwd3D'] = data['SPY_Close'].pct_change(3).shift(-3) * 100
data['Fwd5D'] = data['SPY_Close'].pct_change(5).shift(-5) * 100
data['Fwd10D'] = data['SPY_Close'].pct_change(10).shift(-10) * 100

print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Total days: {len(data)}\n")

# ============================================================================
# PARAMETER SPACE - FIXED to avoid KeyError
# ============================================================================

SPREAD_MA_PERIODS = [5, 10, 20, 30, 50]  # Added 30 and 50
Z_LOOKBACK_PERIODS = [60, 90, 120, 180, 252, 360]
MA_SLOW_PERIODS = [30, 50, 100]  # Removed 20 (too close to fast)
ROC_PERIODS = [1, 2, 3, 5, 10]

print("=" * 80)
print("PARAMETER SPACE")
print("=" * 80)
print(f"Spread MA Periods: {SPREAD_MA_PERIODS}")
print(f"Z-Score Lookbacks: {Z_LOOKBACK_PERIODS}")
print(f"MA Slow Periods: {MA_SLOW_PERIODS}")
print(f"ROC Periods: {ROC_PERIODS}")
print()

# ============================================================================
# CALCULATE ALL INDICATORS
# ============================================================================

print("Calculating spread moving averages...")
for ma_period in SPREAD_MA_PERIODS:
    data[f'SpreadMA{ma_period}'] = data['Spread'].rolling(window=ma_period).mean()

print("Calculating Z-scores across all lookback periods...")
for ma_period in SPREAD_MA_PERIODS:
    for z_period in Z_LOOKBACK_PERIODS:
        mean = data[f'SpreadMA{ma_period}'].rolling(window=z_period).mean()
        std = data[f'SpreadMA{ma_period}'].rolling(window=z_period).std()
        data[f'Z_MA{ma_period}_LB{z_period}'] = (data[f'SpreadMA{ma_period}'] - mean) / std

print("Calculating ROC across all periods...")
for roc_period in ROC_PERIODS:
    data[f'ROC{roc_period}D'] = data['Spread'].pct_change(roc_period) * 100

print("Calculating MA crossovers...")
for slow_period in MA_SLOW_PERIODS:
    # Use SpreadMA10 as fast for all crossovers
    data[f'MA10_Above_MA{slow_period}'] = (data['SpreadMA10'] > data[f'SpreadMA{slow_period}']).astype(int)

print("\nTotal indicators calculated:", len([c for c in data.columns if c.startswith(('Z_', 'ROC', 'MA10_'))]))

# ============================================================================
# TEST ROBUSTNESS OF TOP SIGNALS
# ============================================================================

results = []

print("\n" + "=" * 80)
print("TESTING SIGNAL ROBUSTNESS")
print("=" * 80)

# ============================================================================
# TEST 1: Z-Score Zone [0.0 to 0.25] - Vary MA and Lookback
# ============================================================================

print("\n[1/4] Testing Z-Score [0.0-0.25] across parameters...")

for ma_period in SPREAD_MA_PERIODS:
    for z_period in Z_LOOKBACK_PERIODS:
        z_col = f'Z_MA{ma_period}_LB{z_period}'

        mask = (data[z_col] >= 0.0) & (data[z_col] < 0.25)

        if mask.sum() >= 30:  # Min 30 occurrences
            for fwd in ['Fwd3D', 'Fwd5D', 'Fwd10D']:
                fwd_returns = data.loc[mask, fwd].dropna()

                if len(fwd_returns) >= 30:
                    mean_ret = fwd_returns.mean()
                    std_ret = fwd_returns.std()
                    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0
                    win_rate = (fwd_returns > 0).mean() * 100

                    results.append({
                        'Signal': f'ZScore[0-0.25]',
                        'MA_Period': ma_period,
                        'Z_Lookback': z_period,
                        'ROC_Period': None,
                        'Hold': fwd.replace('Fwd', '').replace('D', ''),
                        'Mean_Return': mean_ret,
                        'Sharpe': sharpe,
                        'Win_Rate': win_rate,
                        'N': len(fwd_returns)
                    })

# ============================================================================
# TEST 2: Extreme Wide [1.0-1.5] → Large Tightening - Vary ROC Period
# ============================================================================

print("[2/4] Testing Extreme Wide → Tightening across parameters...")

for ma_period in SPREAD_MA_PERIODS:
    for z_period in Z_LOOKBACK_PERIODS:
        for roc_period in ROC_PERIODS:
            z_col = f'Z_MA{ma_period}_LB{z_period}'
            roc_col = f'ROC{roc_period}D'

            mask = (data[z_col] >= 1.0) & (data[z_col] < 1.5) & (data[roc_col] < -0.3)

            if mask.sum() >= 10:  # Lower threshold for rare signal
                for fwd in ['Fwd3D', 'Fwd5D', 'Fwd10D']:
                    fwd_returns = data.loc[mask, fwd].dropna()

                    if len(fwd_returns) >= 10:
                        mean_ret = fwd_returns.mean()
                        std_ret = fwd_returns.std()
                        sharpe = (mean_ret / std_ret) if std_ret > 0 else 0
                        win_rate = (fwd_returns > 0).mean() * 100

                        results.append({
                            'Signal': f'ExtremeWide→Tight',
                            'MA_Period': ma_period,
                            'Z_Lookback': z_period,
                            'ROC_Period': roc_period,
                            'Hold': fwd.replace('Fwd', '').replace('D', ''),
                            'Mean_Return': mean_ret,
                            'Sharpe': sharpe,
                            'Win_Rate': win_rate,
                            'N': len(fwd_returns)
                        })

# ============================================================================
# TEST 3: Strong ROC Momentum [1.0% to 2.0%] - Vary ROC Period
# ============================================================================

print("[3/4] Testing Strong ROC Momentum across parameters...")

for roc_period in ROC_PERIODS:
    roc_col = f'ROC{roc_period}D'

    mask = (data[roc_col] >= 1.0) & (data[roc_col] < 2.0)

    if mask.sum() >= 20:
        for fwd in ['Fwd3D', 'Fwd5D', 'Fwd10D']:
            fwd_returns = data.loc[mask, fwd].dropna()

            if len(fwd_returns) >= 20:
                mean_ret = fwd_returns.mean()
                std_ret = fwd_returns.std()
                sharpe = (mean_ret / std_ret) if std_ret > 0 else 0
                win_rate = (fwd_returns > 0).mean() * 100

                results.append({
                    'Signal': f'StrongROC[1.0-2.0]',
                    'MA_Period': None,
                    'Z_Lookback': None,
                    'ROC_Period': roc_period,
                    'Hold': fwd.replace('Fwd', '').replace('D', ''),
                    'Mean_Return': mean_ret,
                    'Sharpe': sharpe,
                    'Win_Rate': win_rate,
                    'N': len(fwd_returns)
                })

# ============================================================================
# TEST 4: MA Crossover + Tight Spreads - Vary MA and Z Lookback
# ============================================================================

print("[4/4] Testing MA Crossover + Tight across parameters...")

for slow_period in MA_SLOW_PERIODS:
    cross_col = f'MA10_Above_MA{slow_period}'

    for ma_period in [10]:  # Only test MA10 for Z-score
        for z_period in Z_LOOKBACK_PERIODS:
            z_col = f'Z_MA{ma_period}_LB{z_period}'

            # Fresh bullish cross (within last 5 days) + tight spreads
            data['Fresh_Cross'] = (data[cross_col] == 1) & (data[cross_col].shift(1) == 0)
            data['Cross_Age'] = 0
            for i in range(1, 6):
                data['Cross_Age'] += data['Fresh_Cross'].shift(i)

            mask = (data['Cross_Age'] > 0) & (data[z_col] < 0)

            if mask.sum() >= 20:
                for fwd in ['Fwd5D', 'Fwd10D']:
                    fwd_returns = data.loc[mask, fwd].dropna()

                    if len(fwd_returns) >= 20:
                        mean_ret = fwd_returns.mean()
                        std_ret = fwd_returns.std()
                        sharpe = (mean_ret / std_ret) if std_ret > 0 else 0
                        win_rate = (fwd_returns > 0).mean() * 100

                        results.append({
                            'Signal': f'MACross+Tight',
                            'MA_Period': slow_period,
                            'Z_Lookback': z_period,
                            'ROC_Period': None,
                            'Hold': fwd.replace('Fwd', '').replace('D', ''),
                            'Mean_Return': mean_ret,
                            'Sharpe': sharpe,
                            'Win_Rate': win_rate,
                            'N': len(fwd_returns)
                        })

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("ROBUSTNESS ANALYSIS RESULTS")
print("=" * 80)
print(f"\nTotal parameter combinations tested: {len(df_results)}")

# Group by signal type
for signal_name in df_results['Signal'].unique():
    signal_data = df_results[df_results['Signal'] == signal_name]

    print("\n" + "=" * 80)
    print(f"SIGNAL: {signal_name}")
    print("=" * 80)

    # Overall statistics
    print(f"\nParameter combinations tested: {len(signal_data)}")
    print(f"\nSharpe Ratio Distribution:")
    print(f"  Mean:   {signal_data['Sharpe'].mean():.2f}")
    print(f"  Median: {signal_data['Sharpe'].median():.2f}")
    print(f"  Std:    {signal_data['Sharpe'].std():.2f}")
    print(f"  Min:    {signal_data['Sharpe'].min():.2f}")
    print(f"  Max:    {signal_data['Sharpe'].max():.2f}")

    # Positive Sharpe rate
    pct_positive = (signal_data['Sharpe'] > 0).mean() * 100
    pct_good = (signal_data['Sharpe'] > 1.5).mean() * 100
    pct_excellent = (signal_data['Sharpe'] > 3.0).mean() * 100

    print(f"\nRobustness Metrics:")
    print(f"  % Positive Sharpe (>0):    {pct_positive:.1f}%")
    print(f"  % Good Sharpe (>1.5):      {pct_good:.1f}%")
    print(f"  % Excellent Sharpe (>3.0): {pct_excellent:.1f}%")

    # Top 10 parameter combinations
    print(f"\nTop 10 Parameter Combinations:")
    top10 = signal_data.nlargest(10, 'Sharpe')[['MA_Period', 'Z_Lookback', 'ROC_Period', 'Hold', 'Mean_Return', 'Sharpe', 'Win_Rate', 'N']]
    print(top10.to_string(index=False))

    # Stability by holding period
    print(f"\nPerformance by Holding Period:")
    for hold in sorted(signal_data['Hold'].unique()):
        hold_data = signal_data[signal_data['Hold'] == hold]
        print(f"  {hold}D: Mean Sharpe = {hold_data['Sharpe'].mean():.2f}, "
              f"Median = {hold_data['Sharpe'].median():.2f}, "
              f"% >1.5 = {(hold_data['Sharpe'] > 1.5).mean()*100:.1f}%")

# ============================================================================
# VERDICT ON ROBUSTNESS
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS VERDICT")
print("=" * 80)

for signal_name in df_results['Signal'].unique():
    signal_data = df_results[df_results['Signal'] == signal_name]

    mean_sharpe = signal_data['Sharpe'].mean()
    median_sharpe = signal_data['Sharpe'].median()
    pct_good = (signal_data['Sharpe'] > 1.5).mean() * 100
    std_sharpe = signal_data['Sharpe'].std()

    print(f"\n{signal_name}:")
    print(f"  Mean Sharpe: {mean_sharpe:.2f}")
    print(f"  Median Sharpe: {median_sharpe:.2f}")
    print(f"  Std Dev: {std_sharpe:.2f}")
    print(f"  % Good (>1.5): {pct_good:.1f}%")

    # Verdict
    if median_sharpe > 2.0 and pct_good > 60:
        verdict = "✅ HIGHLY ROBUST - Works across most parameters"
    elif median_sharpe > 1.0 and pct_good > 40:
        verdict = "✅ ROBUST - Works with proper parameter selection"
    elif median_sharpe > 0.5 and pct_good > 20:
        verdict = "⚠️ MODERATELY ROBUST - Requires careful parameter tuning"
    else:
        verdict = "❌ NOT ROBUST - Likely curve-fitted"

    print(f"  Verdict: {verdict}")

# ============================================================================
# EXPORT RESULTS
# ============================================================================

output_file = 'credit_spread_robustness_results.csv'
df_results.to_csv(output_file, index=False)
print(f"\n✅ Full results exported to: {output_file}")

print("\n" + "=" * 80)
print("ROBUSTNESS TEST COMPLETE")
print("=" * 80)
