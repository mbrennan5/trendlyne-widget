"""
============================================================================
GAP + WIDE RANGE DEEP DIVE ANALYSIS
============================================================================
Tests specific combinations and conditions:
1. Gap + WR combinations (WR2, WR4, WR7)
2. Gap direction vs trend alignment (5 SMA)
3. GapStat thresholds
4. Gap size buckets
5. Gap with trend vs against trend

Focused on finding the BEST gap + momentum setups.

Copy this entire code block into Google Colab and run it.
============================================================================
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = '/content/drive/MyDrive/StockData'
RESULTS_FILE = '/content/drive/MyDrive/backtest_results/daytype_classification_results.csv'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

print("="*80)
print("GAP + WIDE RANGE DEEP DIVE ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load Day Type Results
# ============================================================================
print("\nSTEP 1: Loading day type classification results...")
daytype_df = pd.read_csv(RESULTS_FILE)
daytype_df['Date'] = pd.to_datetime(daytype_df['Date'])
daytype_df = daytype_df.sort_values(['Symbol', 'Date'])

# Create binary target
daytype_df['IsDirectional'] = daytype_df['DayType'].str.contains('Directional', na=False).astype(int)

print(f"✓ Loaded {len(daytype_df):,} classified days for {daytype_df['Symbol'].nunique()} symbols")

all_symbols = sorted(daytype_df['Symbol'].unique())

# ============================================================================
# DEFINE SYMBOL GROUPS
# ============================================================================
SYMBOL_GROUPS = {
    'High Beta Tech': ['NVDA', 'TSLA', 'AMD', 'PLTR', 'SNOW', 'NET', 'DDOG',
                       'SMCI', 'MSTR', 'RIVN', 'ABNB', 'UBER'],
    'Volatility ETFs': ['SOXL', 'TQQQ', 'TSLL'],
    'Major Indices': ['SPY', 'QQQ', 'IWM', 'DIA'],
    'Semiconductor': ['SMH', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'MU', 'MRVL'],
}

# Filter to available symbols
SYMBOL_GROUPS_FILTERED = {}
for group_name, symbols in SYMBOL_GROUPS.items():
    available = [s for s in symbols if s in all_symbols]
    if available:
        SYMBOL_GROUPS_FILTERED[group_name] = available

print("\nSymbol Groups:")
for i, (group_name, symbols) in enumerate(SYMBOL_GROUPS_FILTERED.items(), 1):
    print(f"  {i}. {group_name}: {', '.join(symbols)}")

# ============================================================================
# USER INPUT
# ============================================================================
group_input = input("\nEnter group number: ").strip()
group_idx = int(group_input) - 1
group_name = list(SYMBOL_GROUPS_FILTERED.keys())[group_idx]
SYMBOLS = SYMBOL_GROUPS_FILTERED[group_name]
group_label = group_name.replace(' ', '_')

print(f"\n✓ Selected: {group_name}")
print(f"  Symbols: {', '.join(SYMBOLS)}")

# Filter daytype data
daytype_df = daytype_df[daytype_df['Symbol'].isin(SYMBOLS)].copy()
print(f"  {len(daytype_df):,} classified days")

# ============================================================================
# STEP 2: Load Raw Price Data
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading raw price data...")
print("="*80)

def load_symbol_daily_data(symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """Load all year files for a symbol and create daily OHLCV bars"""

    pattern = f"{symbol}_30Min_*.csv"
    files = list(Path(DATA_DIR).glob(pattern))

    if not files:
        return pd.DataFrame()

    # Extract years and filter
    year_files = []
    for file_path in files:
        parts = file_path.stem.split('_')
        year = None
        for part in parts:
            if part.isdigit() and len(part) == 4 and 2000 <= int(part) <= 2100:
                year = int(part)
                break
        if year and (start_year is None or start_year <= year <= (end_year or 9999)):
            year_files.append((year, file_path))

    if not year_files:
        return pd.DataFrame()

    # Load and concatenate
    all_data = []
    for year, file_path in sorted(year_files):
        df = pd.read_csv(file_path)
        if 't' in df.columns:
            df['datetime'] = pd.to_datetime(df['t'], utc=True)
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)

        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('datetime')
    combined['Date'] = combined['datetime'].dt.date

    # Create daily OHLCV
    daily = combined.groupby('Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).reset_index()

    daily['Date'] = pd.to_datetime(daily['Date'])
    daily['Symbol'] = symbol

    return daily

# Year range
start_year_input = input("Start YEAR (e.g., 2020, or press Enter for all): ").strip()
START_YEAR = None if start_year_input == '' else int(start_year_input)
end_year_input = input("End YEAR (e.g., 2025, or press Enter for all): ").strip()
END_YEAR = None if end_year_input == '' else int(end_year_input)

print(f"\nLoading price data for {len(SYMBOLS)} symbols...")

all_daily_data = []
for symbol in SYMBOLS:
    print(f"  Loading {symbol}...", end=' ')
    daily = load_symbol_daily_data(symbol, START_YEAR, END_YEAR)
    if not daily.empty:
        print(f"✓ {len(daily)} days")
        all_daily_data.append(daily)
    else:
        print("✗ No data")

price_df = pd.concat(all_daily_data, ignore_index=True)
price_df = price_df.sort_values(['Symbol', 'Date'])

print(f"\n✓ Loaded {len(price_df):,} daily bars")

# ============================================================================
# STEP 3: Calculate Indicators with Trend Analysis
# ============================================================================
print("\nSTEP 3: Calculating gap, range, and trend indicators...")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators including trend direction"""

    df = df.copy().sort_values('Date')

    # Price and Range
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100
    df['PrevClose'] = df['Close'].shift(1)

    # Short-term trend (5-day SMA)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['Price_Above_5SMA'] = (df['Close'] > df['SMA_5']).astype(int)
    df['Trend_5D'] = np.where(df['Close'] > df['SMA_5'], 'Up', 'Down')

    # Wide Range indicators
    df['WR2'] = (df['Range'] == df['Range'].rolling(2).max()).astype(int)
    df['WR4'] = (df['Range'] == df['Range'].rolling(4).max()).astype(int)
    df['WR7'] = (df['Range'] == df['Range'].rolling(7).max()).astype(int)

    # Gap calculations
    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100
    df['Gap_Direction'] = np.where(df['Gap'] > 0, 'Up', np.where(df['Gap'] < 0, 'Down', 'Flat'))

    # Gap size categories
    df['Gap_Size'] = 'No_Gap'
    df.loc[df['Gap_Pct'].abs() > 0.5, 'Gap_Size'] = 'Small'
    df.loc[df['Gap_Pct'].abs() > 1.0, 'Gap_Size'] = 'Medium'
    df.loc[df['Gap_Pct'].abs() > 1.5, 'Gap_Size'] = 'Large'
    df.loc[df['Gap_Pct'].abs() > 2.5, 'Gap_Size'] = 'XLarge'

    # Binary gap indicators
    df['Has_Gap'] = (df['Gap_Pct'].abs() > 0.5).astype(int)
    df['Gap_Up'] = (df['Gap_Pct'] > 0.5).astype(int)
    df['Gap_Down'] = (df['Gap_Pct'] < -0.5).astype(int)
    df['Large_Gap'] = (df['Gap_Pct'].abs() > 1.5).astype(int)
    df['XLarge_Gap'] = (df['Gap_Pct'].abs() > 2.5).astype(int)

    # Gap vs Trend alignment
    df['Gap_With_Trend'] = 0
    df['Gap_Against_Trend'] = 0

    df.loc[(df['Gap'] > 0) & (df['Trend_5D'] == 'Up'), 'Gap_With_Trend'] = 1
    df.loc[(df['Gap'] < 0) & (df['Trend_5D'] == 'Down'), 'Gap_With_Trend'] = 1

    df.loc[(df['Gap'] > 0) & (df['Trend_5D'] == 'Down'), 'Gap_Against_Trend'] = 1
    df.loc[(df['Gap'] < 0) & (df['Trend_5D'] == 'Up'), 'Gap_Against_Trend'] = 1

    # GapStat calculation
    df['Gap_Pct_TOS'] = (df['Gap'] / df['PrevClose']) * 10
    df['Gap_Pct_TOS_Up'] = df['Gap_Pct_TOS'].where(df['Gap_Pct_TOS'] > 0, np.nan)
    df['Gap_Pct_TOS_Down'] = df['Gap_Pct_TOS'].where(df['Gap_Pct_TOS'] < 0, np.nan)

    lookback = 250
    df['AvgGapUp'] = df['Gap_Pct_TOS_Up'].rolling(lookback, min_periods=1).mean()
    df['AvgGapDown'] = df['Gap_Pct_TOS_Down'].rolling(lookback, min_periods=1).mean()
    df['StdevGapUp'] = df['Gap_Pct_TOS_Up'].rolling(lookback, min_periods=1).std()
    df['StdevGapDown'] = df['Gap_Pct_TOS_Down'].rolling(lookback, min_periods=1).std()

    df['NormalizedGapUp'] = df['AvgGapUp'].fillna(0.1) + df['StdevGapUp'].fillna(0)
    df['NormalizedGapDown'] = df['AvgGapDown'].fillna(-0.1) + df['StdevGapDown'].fillna(0)

    df['NormFactor'] = np.where(df['Gap'] > 0,
                                  df['NormalizedGapUp'],
                                  df['NormalizedGapDown'].abs())

    df['GapStat'] = np.where((df['NormFactor'].isna()) | (df['NormFactor'] == 0),
                              0,
                              df['Gap_Pct_TOS'] / df['NormFactor'])

    # GapStat thresholds
    df['GapStat_Above_1'] = (df['GapStat'].abs() > 1.0).astype(int)
    df['GapStat_Above_1_5'] = (df['GapStat'].abs() > 1.5).astype(int)
    df['GapStat_Above_2'] = (df['GapStat'].abs() > 2.0).astype(int)
    df['GapStat_Above_2_5'] = (df['GapStat'].abs() > 2.5).astype(int)

    return df

# Calculate indicators per symbol
indicator_data = []
for symbol in SYMBOLS:
    symbol_price = price_df[price_df['Symbol'] == symbol].copy()
    if not symbol_price.empty:
        symbol_indicators = calculate_indicators(symbol_price)
        indicator_data.append(symbol_indicators)

indicators_df = pd.concat(indicator_data, ignore_index=True)

print(f"✓ Calculated indicators for {len(indicators_df):,} daily bars")

# ============================================================================
# STEP 4: Merge with Day Types
# ============================================================================
print("\nSTEP 4: Creating predictive dataset...")

merged_df = indicators_df.merge(
    daytype_df[['Symbol', 'Date', 'DayType', 'IsDirectional']],
    on=['Symbol', 'Date'],
    how='inner'
)

# Create next-day target
merged_df = merged_df.sort_values(['Symbol', 'Date'])
merged_df['Next_IsDirectional'] = merged_df.groupby('Symbol')['IsDirectional'].shift(-1)
predictive_df = merged_df.dropna(subset=['Next_IsDirectional']).copy()

print(f"✓ Created predictive dataset with {len(predictive_df):,} samples")

baseline = predictive_df['Next_IsDirectional'].mean() * 100
print(f"  Baseline directional rate: {baseline:.1f}%")

# ============================================================================
# STEP 5: Gap + WR Combination Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 5: GAP + WIDE RANGE COMBINATIONS")
print("="*80)

def analyze_combo(df: pd.DataFrame, condition_name: str, condition_mask: pd.Series) -> Dict:
    """Analyze a specific condition"""

    subset = df[condition_mask]

    if len(subset) < 10:
        return None

    dir_rate = subset['Next_IsDirectional'].mean() * 100
    baseline = df['Next_IsDirectional'].mean() * 100
    edge = dir_rate - baseline

    return {
        'Condition': condition_name,
        'Count': len(subset),
        'Directional%': dir_rate,
        'Baseline%': baseline,
        'Edge': edge,
        'Abs_Edge': abs(edge)
    }

results = []

print("\n--- GAP SIZE + WR COMBINATIONS ---")
print("-" * 80)

# Test different gap sizes with WR indicators
gap_wr_combos = [
    # Small gaps
    ('Small Gap + WR2', (predictive_df['Gap_Size'] == 'Small') & (predictive_df['WR2'] == 1)),
    ('Small Gap + WR4', (predictive_df['Gap_Size'] == 'Small') & (predictive_df['WR4'] == 1)),
    ('Small Gap + WR7', (predictive_df['Gap_Size'] == 'Small') & (predictive_df['WR7'] == 1)),

    # Medium gaps
    ('Medium Gap + WR2', (predictive_df['Gap_Size'] == 'Medium') & (predictive_df['WR2'] == 1)),
    ('Medium Gap + WR4', (predictive_df['Gap_Size'] == 'Medium') & (predictive_df['WR4'] == 1)),
    ('Medium Gap + WR7', (predictive_df['Gap_Size'] == 'Medium') & (predictive_df['WR7'] == 1)),

    # Large gaps
    ('Large Gap + WR2', (predictive_df['Gap_Size'] == 'Large') & (predictive_df['WR2'] == 1)),
    ('Large Gap + WR4', (predictive_df['Gap_Size'] == 'Large') & (predictive_df['WR4'] == 1)),
    ('Large Gap + WR7', (predictive_df['Gap_Size'] == 'Large') & (predictive_df['WR7'] == 1)),

    # XLarge gaps
    ('XLarge Gap + WR2', (predictive_df['Gap_Size'] == 'XLarge') & (predictive_df['WR2'] == 1)),
    ('XLarge Gap + WR4', (predictive_df['Gap_Size'] == 'XLarge') & (predictive_df['WR4'] == 1)),
    ('XLarge Gap + WR7', (predictive_df['Gap_Size'] == 'XLarge') & (predictive_df['WR7'] == 1)),
]

for name, mask in gap_wr_combos:
    result = analyze_combo(predictive_df, name, mask)
    if result:
        results.append(result)
        print(f"  {result['Condition']:25s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:4,})")

print("\n--- GAPSTAT THRESHOLD + WR COMBINATIONS ---")
print("-" * 80)

gapstat_wr_combos = [
    ('GapStat>1.0 + WR2', (predictive_df['GapStat_Above_1'] == 1) & (predictive_df['WR2'] == 1)),
    ('GapStat>1.0 + WR4', (predictive_df['GapStat_Above_1'] == 1) & (predictive_df['WR4'] == 1)),
    ('GapStat>1.0 + WR7', (predictive_df['GapStat_Above_1'] == 1) & (predictive_df['WR7'] == 1)),

    ('GapStat>1.5 + WR2', (predictive_df['GapStat_Above_1_5'] == 1) & (predictive_df['WR2'] == 1)),
    ('GapStat>1.5 + WR4', (predictive_df['GapStat_Above_1_5'] == 1) & (predictive_df['WR4'] == 1)),
    ('GapStat>1.5 + WR7', (predictive_df['GapStat_Above_1_5'] == 1) & (predictive_df['WR7'] == 1)),

    ('GapStat>2.0 + WR2', (predictive_df['GapStat_Above_2'] == 1) & (predictive_df['WR2'] == 1)),
    ('GapStat>2.0 + WR4', (predictive_df['GapStat_Above_2'] == 1) & (predictive_df['WR4'] == 1)),
    ('GapStat>2.0 + WR7', (predictive_df['GapStat_Above_2'] == 1) & (predictive_df['WR7'] == 1)),

    ('GapStat>2.5 + WR2', (predictive_df['GapStat_Above_2_5'] == 1) & (predictive_df['WR2'] == 1)),
    ('GapStat>2.5 + WR4', (predictive_df['GapStat_Above_2_5'] == 1) & (predictive_df['WR4'] == 1)),
    ('GapStat>2.5 + WR7', (predictive_df['GapStat_Above_2_5'] == 1) & (predictive_df['WR7'] == 1)),
]

for name, mask in gapstat_wr_combos:
    result = analyze_combo(predictive_df, name, mask)
    if result:
        results.append(result)
        print(f"  {result['Condition']:25s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:4,})")

# ============================================================================
# STEP 6: Gap Direction vs Trend Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 6: GAP DIRECTION VS TREND ALIGNMENT")
print("="*80)

print("\n--- GAP WITH TREND vs AGAINST TREND ---")
print("-" * 80)

gap_trend_combos = [
    # Gap with trend
    ('Gap With Trend + WR2', (predictive_df['Gap_With_Trend'] == 1) & (predictive_df['WR2'] == 1)),
    ('Gap With Trend + WR4', (predictive_df['Gap_With_Trend'] == 1) & (predictive_df['WR4'] == 1)),
    ('Gap With Trend + WR7', (predictive_df['Gap_With_Trend'] == 1) & (predictive_df['WR7'] == 1)),
    ('Gap With Trend Only', (predictive_df['Gap_With_Trend'] == 1)),

    # Gap against trend
    ('Gap Against Trend + WR2', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['WR2'] == 1)),
    ('Gap Against Trend + WR4', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['WR4'] == 1)),
    ('Gap Against Trend + WR7', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['WR7'] == 1)),
    ('Gap Against Trend Only', (predictive_df['Gap_Against_Trend'] == 1)),
]

for name, mask in gap_trend_combos:
    result = analyze_combo(predictive_df, name, mask)
    if result:
        results.append(result)
        print(f"  {result['Condition']:30s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:4,})")

print("\n--- LARGE GAP WITH/AGAINST TREND ---")
print("-" * 80)

large_gap_trend = [
    ('Large Gap With Trend', (predictive_df['Large_Gap'] == 1) & (predictive_df['Gap_With_Trend'] == 1)),
    ('Large Gap Against Trend', (predictive_df['Large_Gap'] == 1) & (predictive_df['Gap_Against_Trend'] == 1)),
    ('Large Gap With Trend + WR4', (predictive_df['Large_Gap'] == 1) & (predictive_df['Gap_With_Trend'] == 1) & (predictive_df['WR4'] == 1)),
    ('Large Gap Against Trend + WR4', (predictive_df['Large_Gap'] == 1) & (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['WR4'] == 1)),
]

for name, mask in large_gap_trend:
    result = analyze_combo(predictive_df, name, mask)
    if result:
        results.append(result)
        print(f"  {result['Condition']:30s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:4,})")

print("\n--- GAP DIRECTION SPECIFIC (Up vs Down) ---")
print("-" * 80)

gap_direction_combos = [
    ('Gap Up + WR2', (predictive_df['Gap_Up'] == 1) & (predictive_df['WR2'] == 1)),
    ('Gap Up + WR4', (predictive_df['Gap_Up'] == 1) & (predictive_df['WR4'] == 1)),
    ('Gap Up + WR7', (predictive_df['Gap_Up'] == 1) & (predictive_df['WR7'] == 1)),

    ('Gap Down + WR2', (predictive_df['Gap_Down'] == 1) & (predictive_df['WR2'] == 1)),
    ('Gap Down + WR4', (predictive_df['Gap_Down'] == 1) & (predictive_df['WR4'] == 1)),
    ('Gap Down + WR7', (predictive_df['Gap_Down'] == 1) & (predictive_df['WR7'] == 1)),
]

for name, mask in gap_direction_combos:
    result = analyze_combo(predictive_df, name, mask)
    if result:
        results.append(result)
        print(f"  {result['Condition']:25s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:4,})")

# ============================================================================
# STEP 7: Rankings & Summary
# ============================================================================
print("\n" + "="*80)
print("STEP 7: TOP 20 COMBINATIONS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Abs_Edge', ascending=False)

print("\nRanked by Absolute Edge:")
print("-" * 80)
print(results_df[['Condition', 'Edge', 'Directional%', 'Count']].head(20).to_string(index=False))

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("="*80)

output_file = os.path.join(OUTPUT_DIR, f'gap_wr_analysis_{group_label}.csv')
results_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Gap + Wide Range Analysis - {group_label}', fontsize=16, fontweight='bold')

# 1. Top combinations
ax1 = axes[0, 0]
top_15 = results_df.head(15).sort_values('Edge')
colors = ['green' if x > 0 else 'red' for x in top_15['Edge']]
ax1.barh(range(len(top_15)), top_15['Edge'], color=colors)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['Condition'], fontsize=8)
ax1.set_xlabel('Edge (%)')
ax1.set_title('Top 15 Gap + WR Combinations')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 2. Gap size comparison
ax2 = axes[0, 1]
gap_size_results = results_df[results_df['Condition'].str.contains('Gap \+')]
if len(gap_size_results) > 0:
    gap_sizes = gap_size_results.groupby(gap_size_results['Condition'].str.extract(r'(\w+ Gap)', expand=False))['Edge'].mean().sort_values()
    ax2.barh(range(len(gap_sizes)), gap_sizes.values, color='steelblue')
    ax2.set_yticks(range(len(gap_sizes)))
    ax2.set_yticklabels(gap_sizes.index)
    ax2.set_xlabel('Average Edge (%)')
    ax2.set_title('Average Edge by Gap Size')
    ax2.grid(axis='x', alpha=0.3)

# 3. With vs Against Trend
ax3 = axes[1, 0]
trend_align = results_df[results_df['Condition'].str.contains('Trend')]
if len(trend_align) > 0:
    trend_avg = trend_align.groupby(trend_align['Condition'].str.contains('With'))['Edge'].mean()
    labels = ['Against Trend', 'With Trend']
    colors = ['red' if x < 0 else 'green' for x in trend_avg.values]
    ax3.bar(range(len(trend_avg)), trend_avg.values, color=colors)
    ax3.set_xticks(range(len(trend_avg)))
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Average Edge (%)')
    ax3.set_title('Gap With Trend vs Against Trend')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='y', alpha=0.3)

# 4. GapStat threshold comparison
ax4 = axes[1, 1]
gapstat_results = results_df[results_df['Condition'].str.contains('GapStat>')]
if len(gapstat_results) > 0:
    top_gapstat = gapstat_results.head(10).sort_values('Edge')
    colors = ['green' if x > 0 else 'red' for x in top_gapstat['Edge']]
    ax4.barh(range(len(top_gapstat)), top_gapstat['Edge'], color=colors)
    ax4.set_yticks(range(len(top_gapstat)))
    ax4.set_yticklabels(top_gapstat['Condition'], fontsize=8)
    ax4.set_xlabel('Edge (%)')
    ax4.set_title('Top GapStat + WR Combinations')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

viz_file = os.path.join(OUTPUT_DIR, f'gap_wr_analysis_{group_label}.png')
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved charts to: {viz_file}")
plt.show()

# Download files
try:
    from google.colab import files
    print("\nDownloading files...")
    files.download(output_file)
    files.download(viz_file)
    print("✓ Files downloaded!")
except:
    print("\nNot in Colab - files saved to Drive only")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings:")
print("- Best gap size + WR combination")
print("- Optimal GapStat threshold")
print("- Gap with trend vs against trend performance")
print("- Gap direction (up/down) effects")
print("="*80)
