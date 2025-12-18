"""
Credit Spread Expanded Analysis

Test the core hypothesis: "Narrowing credit spreads = Bullish for SPY"

This expanded test explores:
1. Simple directional signals (spread tightening/widening)
2. Longer holding periods (1D to 60D)
3. Different measurement methods (ROC, direction, momentum)
4. Regime-based analysis (only certain spread levels)
5. Continuous exposure strategies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT SPREAD EXPANDED ANALYSIS")
print("Testing: Does spread narrowing predict bullish SPY returns?")
print("=" * 80)

# Download data
start_date = '2007-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

print("\nDownloading data...")
raw = yf.download(['IEI', 'HYG', 'SPY'], start=start_date, end=end_date, progress=False)

data = pd.DataFrame({
    'IEI': raw[('Close', 'IEI')],
    'HYG': raw[('Close', 'HYG')],
    'SPY': raw[('Close', 'SPY')]
})

data = data.dropna()
data['Spread'] = data['IEI'] / data['HYG']

# Calculate forward returns for various periods
print("Calculating forward returns...")
for days in [1, 2, 3, 5, 10, 20, 30, 60]:
    data[f'Fwd{days}D'] = data['SPY'].pct_change(days).shift(-days) * 100

print(f"✅ Data: {data.index[0].date()} to {data.index[-1].date()} ({len(data)} days)\n")

# ============================================================================
# TEST 1: Simple Spread Direction
# ============================================================================

print("=" * 80)
print("TEST 1: SIMPLE SPREAD DIRECTION")
print("Does spread direction alone predict returns?")
print("=" * 80)

results_direction = []

for lookback in [1, 2, 3, 5, 10, 20]:
    # Calculate spread change
    spread_change = data['Spread'].pct_change(lookback)

    # Narrowing = spread decreasing (negative change)
    narrowing = spread_change < 0
    widening = spread_change > 0

    print(f"\n{lookback}D Spread Change:")
    print(f"  Narrowing periods: {narrowing.sum()}")
    print(f"  Widening periods: {widening.sum()}")

    for hold in [1, 2, 3, 5, 10, 20, 30, 60]:
        fwd_col = f'Fwd{hold}D'

        narrow_rets = data.loc[narrowing, fwd_col].dropna()
        wide_rets = data.loc[widening, fwd_col].dropna()

        if len(narrow_rets) >= 50 and len(wide_rets) >= 50:
            # Calculate statistics
            narrow_mean = narrow_rets.mean()
            narrow_sharpe = narrow_mean / narrow_rets.std() if narrow_rets.std() > 0 else 0
            narrow_win = (narrow_rets > 0).mean() * 100

            wide_mean = wide_rets.mean()
            wide_sharpe = wide_mean / wide_rets.std() if wide_rets.std() > 0 else 0
            wide_win = (wide_rets > 0).mean() * 100

            # Calculate edge (narrowing vs widening)
            edge = narrow_mean - wide_mean

            results_direction.append({
                'Lookback': lookback,
                'Hold': hold,
                'Narrow_Ret': narrow_mean,
                'Narrow_Sharpe': narrow_sharpe,
                'Narrow_Win': narrow_win,
                'Narrow_N': len(narrow_rets),
                'Wide_Ret': wide_mean,
                'Wide_Sharpe': wide_sharpe,
                'Wide_Win': wide_win,
                'Wide_N': len(wide_rets),
                'Edge': edge
            })

df_dir = pd.DataFrame(results_direction)

print("\n" + "=" * 80)
print("BEST EDGES (Narrowing vs Widening)")
print("=" * 80)
top_edges = df_dir.nlargest(10, 'Edge')
for _, r in top_edges.iterrows():
    print(f"Lookback={r['Lookback']}D, Hold={r['Hold']}D: Edge={r['Edge']:.3f}% (Narrow: {r['Narrow_Ret']:.2f}%, Wide: {r['Wide_Ret']:.2f}%)")

# ============================================================================
# TEST 2: Spread Momentum (Consecutive Narrowing)
# ============================================================================

print("\n" + "=" * 80)
print("TEST 2: SPREAD MOMENTUM")
print("Does consecutive narrowing strengthen the signal?")
print("=" * 80)

results_momentum = []

# Calculate daily spread change
data['Spread_Change'] = data['Spread'].pct_change()
data['Is_Narrowing'] = (data['Spread_Change'] < 0).astype(int)

# Count consecutive narrowing days
data['Consec_Narrow'] = 0
for i in range(1, 11):
    data['Consec_Narrow'] += data['Is_Narrowing'].shift(i)

for consec in [1, 2, 3, 5]:
    mask = data['Consec_Narrow'] >= consec

    if mask.sum() >= 50:
        print(f"\n{consec}+ Consecutive Narrowing Days: {mask.sum()} occurrences")

        for hold in [5, 10, 20, 30, 60]:
            fwd_col = f'Fwd{hold}D'
            rets = data.loc[mask, fwd_col].dropna()

            if len(rets) >= 30:
                results_momentum.append({
                    'Consecutive': consec,
                    'Hold': hold,
                    'Mean_Ret': rets.mean(),
                    'Sharpe': rets.mean() / rets.std() if rets.std() > 0 else 0,
                    'Win_Rate': (rets > 0).mean() * 100,
                    'N': len(rets)
                })

df_mom = pd.DataFrame(results_momentum)
if len(df_mom) > 0:
    print("\nBest Momentum Signals:")
    top_mom = df_mom.nlargest(5, 'Sharpe')
    for _, r in top_mom.iterrows():
        print(f"  {r['Consecutive']}+ consec days, {r['Hold']}D hold: Sharpe={r['Sharpe']:.2f}, Ret={r['Mean_Ret']:.2f}%, Win={r['Win_Rate']:.0f}%")

# ============================================================================
# TEST 3: Regime-Based (Only During Elevated Spreads)
# ============================================================================

print("\n" + "=" * 80)
print("TEST 3: REGIME-BASED NARROWING")
print("Does narrowing work better when spreads are elevated?")
print("=" * 80)

results_regime = []

# Calculate spread percentile
data['Spread_Pctl'] = data['Spread'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)

# Define regimes
regimes = [
    ('Extremely Tight', 0, 20),
    ('Tight', 20, 40),
    ('Normal', 40, 60),
    ('Wide', 60, 80),
    ('Extremely Wide', 80, 100)
]

for regime_name, pctl_low, pctl_high in regimes:
    in_regime = (data['Spread_Pctl'] >= pctl_low) & (data['Spread_Pctl'] < pctl_high)

    # Check for narrowing within this regime
    for lookback in [1, 3, 5]:
        spread_change = data['Spread'].pct_change(lookback)
        narrowing_in_regime = in_regime & (spread_change < 0)

        if narrowing_in_regime.sum() >= 30:
            for hold in [10, 20, 30, 60]:
                fwd_col = f'Fwd{hold}D'
                rets = data.loc[narrowing_in_regime, fwd_col].dropna()

                if len(rets) >= 30:
                    results_regime.append({
                        'Regime': regime_name,
                        'Lookback': lookback,
                        'Hold': hold,
                        'Mean_Ret': rets.mean(),
                        'Sharpe': rets.mean() / rets.std() if rets.std() > 0 else 0,
                        'Win_Rate': (rets > 0).mean() * 100,
                        'N': len(rets)
                    })

df_regime = pd.DataFrame(results_regime)
if len(df_regime) > 0:
    print("\nBest Regime-Based Signals:")
    top_regime = df_regime.nlargest(10, 'Sharpe')
    for _, r in top_regime.iterrows():
        print(f"  {r['Regime']}, {r['Lookback']}D→{r['Hold']}D: Sharpe={r['Sharpe']:.2f}, Ret={r['Mean_Ret']:.2f}%, Win={r['Win_Rate']:.0f}%")

# ============================================================================
# TEST 4: Magnitude of Narrowing
# ============================================================================

print("\n" + "=" * 80)
print("TEST 4: MAGNITUDE OF NARROWING")
print("Do larger narrowing moves predict stronger returns?")
print("=" * 80)

results_magnitude = []

for lookback in [1, 3, 5, 10]:
    spread_roc = data['Spread'].pct_change(lookback) * 100

    # Define magnitude buckets (negative = narrowing)
    buckets = [
        ('Large Narrowing', -100, -2.0),
        ('Moderate Narrowing', -2.0, -0.5),
        ('Small Narrowing', -0.5, 0),
        ('Small Widening', 0, 0.5),
        ('Moderate Widening', 0.5, 2.0),
        ('Large Widening', 2.0, 100)
    ]

    for bucket_name, low, high in buckets:
        mask = (spread_roc >= low) & (spread_roc < high)

        if mask.sum() >= 30:
            for hold in [10, 20, 30, 60]:
                fwd_col = f'Fwd{hold}D'
                rets = data.loc[mask, fwd_col].dropna()

                if len(rets) >= 30:
                    results_magnitude.append({
                        'Lookback': lookback,
                        'Bucket': bucket_name,
                        'Hold': hold,
                        'Mean_Ret': rets.mean(),
                        'Sharpe': rets.mean() / rets.std() if rets.std() > 0 else 0,
                        'Win_Rate': (rets > 0).mean() * 100,
                        'N': len(rets)
                    })

df_mag = pd.DataFrame(results_magnitude)
if len(df_mag) > 0:
    print("\nBest Magnitude Signals:")
    # Focus on narrowing signals
    narrowing = df_mag[df_mag['Bucket'].str.contains('Narrowing')]
    if len(narrowing) > 0:
        top_mag = narrowing.nlargest(10, 'Sharpe')
        for _, r in top_mag.iterrows():
            print(f"  {r['Bucket']}, {r['Lookback']}D→{r['Hold']}D: Sharpe={r['Sharpe']:.2f}, Ret={r['Mean_Ret']:.2f}%, Win={r['Win_Rate']:.0f}%")

# ============================================================================
# TEST 5: Continuous Exposure Strategy
# ============================================================================

print("\n" + "=" * 80)
print("TEST 5: CONTINUOUS EXPOSURE")
print("Allocate to SPY based on spread narrowing strength")
print("=" * 80)

# Calculate 5-day spread ROC
data['Spread_ROC5'] = data['Spread'].pct_change(5) * 100

# Define allocation tiers (negative ROC = narrowing = bullish)
def get_allocation(roc):
    if pd.isna(roc):
        return 0
    elif roc < -2.0:  # Large narrowing
        return 150  # 150% long
    elif roc < -0.5:  # Moderate narrowing
        return 100  # 100% long
    elif roc < 0:     # Small narrowing
        return 75   # 75% long
    elif roc < 0.5:   # Small widening
        return 50   # 50% long
    elif roc < 2.0:   # Moderate widening
        return 25   # 25% long
    else:             # Large widening
        return 0    # 0% (cash)

data['Allocation'] = data['Spread_ROC5'].apply(get_allocation)
data['SPY_Daily_Ret'] = data['SPY'].pct_change()
data['Strategy_Daily_Ret'] = data['Allocation'].shift(1) / 100 * data['SPY_Daily_Ret']

# Calculate cumulative returns
data['SPY_Cumulative'] = (1 + data['SPY_Daily_Ret']).cumprod()
data['Strategy_Cumulative'] = (1 + data['Strategy_Daily_Ret']).cumprod()

# Performance metrics
strategy_rets = data['Strategy_Daily_Ret'].dropna()
spy_rets = data['SPY_Daily_Ret'].dropna()

print(f"\nContinuous Exposure Strategy (vs Buy & Hold):")
print(f"  Strategy Return: {(data['Strategy_Cumulative'].iloc[-1] - 1) * 100:.1f}%")
print(f"  SPY Return: {(data['SPY_Cumulative'].iloc[-1] - 1) * 100:.1f}%")
print(f"  Strategy Sharpe: {strategy_rets.mean() / strategy_rets.std() * np.sqrt(252):.2f}")
print(f"  SPY Sharpe: {spy_rets.mean() / spy_rets.std() * np.sqrt(252):.2f}")
print(f"  Strategy Volatility: {strategy_rets.std() * np.sqrt(252) * 100:.1f}%")
print(f"  SPY Volatility: {spy_rets.std() * np.sqrt(252) * 100:.1f}%")

# ============================================================================
# SUMMARY & EXPORT
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

df_dir.to_csv('spread_direction_test.csv', index=False)
print("✅ spread_direction_test.csv")

if len(df_mom) > 0:
    df_mom.to_csv('spread_momentum_test.csv', index=False)
    print("✅ spread_momentum_test.csv")

if len(df_regime) > 0:
    df_regime.to_csv('spread_regime_test.csv', index=False)
    print("✅ spread_regime_test.csv")

if len(df_mag) > 0:
    df_mag.to_csv('spread_magnitude_test.csv', index=False)
    print("✅ spread_magnitude_test.csv")

# Save strategy equity curve
equity_data = data[['SPY_Cumulative', 'Strategy_Cumulative']].dropna()
equity_data.to_csv('spread_continuous_strategy.csv')
print("✅ spread_continuous_strategy.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
