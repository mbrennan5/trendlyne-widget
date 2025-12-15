"""
============================================================================
GAP + WR + GDT DAY# ANALYSIS
============================================================================
Overlays GDT Day# momentum regime indicator with gap + WR combinations.

GDT Day# Logic:
- Tracks 2-day ROC momentum regime changes
- Buy Days: 1, 2, 2.1, 2.2, 2.3 (days after bullish flip)
- Sell Days: 3, 4, 4.1, 4.2, 4.3 (days after bearish flip)

Tests which gap + WR combinations work best on each GDT day type.

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
print("GAP + WR + GDT DAY# ANALYSIS")
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
# STEP 3: Calculate Indicators + GDT Day#
# ============================================================================
print("\nSTEP 3: Calculating gap, range, and GDT Day# indicators...")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators including GDT Day# logic"""

    df = df.copy().sort_values('Date')

    # Price and Range
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100
    df['PrevClose'] = df['Close'].shift(1)

    # Wide Range indicators
    df['WR2'] = (df['Range'] == df['Range'].rolling(2).max()).astype(int)
    df['WR4'] = (df['Range'] == df['Range'].rolling(4).max()).astype(int)
    df['WR7'] = (df['Range'] == df['Range'].rolling(7).max()).astype(int)

    # Gap calculations
    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100

    # Gap size categories
    df['Gap_Size'] = 'No_Gap'
    df.loc[df['Gap_Pct'].abs() > 0.5, 'Gap_Size'] = 'Small'
    df.loc[df['Gap_Pct'].abs() > 1.5, 'Gap_Size'] = 'Large'

    df['Large_Gap'] = (df['Gap_Pct'].abs() > 1.5).astype(int)

    # Short-term trend (5-day SMA)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['Trend_5D'] = np.where(df['Close'] > df['SMA_5'], 'Up', 'Down')

    # Gap vs Trend alignment
    df['Gap_With_Trend'] = 0
    df['Gap_Against_Trend'] = 0

    df.loc[(df['Gap'] > 0) & (df['Trend_5D'] == 'Up'), 'Gap_With_Trend'] = 1
    df.loc[(df['Gap'] < 0) & (df['Trend_5D'] == 'Down'), 'Gap_With_Trend'] = 1

    df.loc[(df['Gap'] > 0) & (df['Trend_5D'] == 'Down'), 'Gap_Against_Trend'] = 1
    df.loc[(df['Gap'] < 0) & (df['Trend_5D'] == 'Up'), 'Gap_Against_Trend'] = 1

    # ========================================================================
    # GDT DAY# LOGIC (from ThinkScript)
    # ========================================================================

    # ROC = close[0] - close[2]
    df['ROC'] = df['Close'] - df['Close'].shift(2)

    # ROCprev = close[1] - close[3]
    df['ROCprev'] = df['Close'].shift(1) - df['Close'].shift(3)

    # ROClevel = ROCprev + close[2]
    df['ROClevel'] = df['ROCprev'] + df['Close'].shift(2)

    # State logic
    df['State'] = 0
    df.loc[df['Close'] > df['ROClevel'], 'State'] = 1
    df.loc[df['Close'] <= df['ROClevel'], 'State'] = -1

    # Buy/Sell day detection
    df['BuyDay'] = ((df['State'].shift(1) < 0) & (df['State'] > 0)).astype(int)
    df['SellDay'] = ((df['State'].shift(1) > 0) & (df['State'] < 0)).astype(int)

    # GDT Day# classification
    df['GDT_DayType'] = None

    # Buy cycle days
    df.loc[df['BuyDay'] == 1, 'GDT_DayType'] = 'Buy_Day1'

    # Day 2 after buy
    mask = (df['BuyDay'].shift(1) == 1) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day2'

    # Day 3 after buy
    mask = (df['BuyDay'].shift(2) == 1) & (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day3'

    # Day 4 after buy
    mask = (df['BuyDay'].shift(3) == 1) & (df['BuyDay'].shift(2) == 0) & (df['SellDay'].shift(2) == 0) & \
           (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day4'

    # Sell cycle days
    df.loc[df['SellDay'] == 1, 'GDT_DayType'] = 'Sell_Day1'

    # Day 2 after sell
    mask = (df['SellDay'].shift(1) == 1) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Sell_Day2'

    # Day 3 after sell
    mask = (df['SellDay'].shift(2) == 1) & (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & \
           (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Sell_Day3'

    # Day 4 after sell
    mask = (df['SellDay'].shift(3) == 1) & (df['BuyDay'].shift(2) == 0) & (df['SellDay'].shift(2) == 0) & \
           (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Sell_Day4'

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

# Show GDT Day distribution
print("\nGDT Day# Distribution:")
gdt_counts = predictive_df['GDT_DayType'].value_counts().sort_index()
for day_type, count in gdt_counts.items():
    if day_type:
        pct = count / len(predictive_df) * 100
        print(f"  {day_type:15s}: {count:5,} ({pct:4.1f}%)")

# ============================================================================
# STEP 5: Analyze Gap + WR by GDT Day#
# ============================================================================
print("\n" + "="*80)
print("STEP 5: GAP + WR PERFORMANCE BY GDT DAY#")
print("="*80)

def analyze_combo_by_gdt(df: pd.DataFrame, condition_name: str, condition_mask: pd.Series, gdt_day: str) -> Dict:
    """Analyze a specific condition on a specific GDT day"""

    subset = df[condition_mask & (df['GDT_DayType'] == gdt_day)]

    if len(subset) < 10:
        return None

    dir_rate = subset['Next_IsDirectional'].mean() * 100
    baseline = df[df['GDT_DayType'] == gdt_day]['Next_IsDirectional'].mean() * 100
    edge = dir_rate - baseline

    return {
        'Condition': condition_name,
        'GDT_Day': gdt_day,
        'Count': len(subset),
        'Directional%': dir_rate,
        'Baseline%': baseline,
        'Edge': edge,
        'Abs_Edge': abs(edge)
    }

# Top combinations to test
top_combos = [
    ('Gap Against Trend + WR4', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['WR4'] == 1)),
    ('Gap Against Trend + WR2', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['WR2'] == 1)),
    ('Gap With Trend + WR4', (predictive_df['Gap_With_Trend'] == 1) & (predictive_df['WR4'] == 1)),
    ('Large Gap + WR4', (predictive_df['Large_Gap'] == 1) & (predictive_df['WR4'] == 1)),
    ('WR4 Only', (predictive_df['WR4'] == 1)),
    ('WR2 Only', (predictive_df['WR2'] == 1)),
]

results = []

# Test each combination on each GDT day
gdt_days = ['Buy_Day1', 'Buy_Day2', 'Buy_Day3', 'Buy_Day4',
            'Sell_Day1', 'Sell_Day2', 'Sell_Day3', 'Sell_Day4']

for combo_name, combo_mask in top_combos:
    print(f"\n{combo_name}:")
    print("-" * 80)

    for gdt_day in gdt_days:
        result = analyze_combo_by_gdt(predictive_df, combo_name, combo_mask, gdt_day)

        if result:
            results.append(result)
            print(f"  {result['GDT_Day']:15s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:4,})")

# ============================================================================
# STEP 6: Rankings & Insights
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TOP EDGES BY GDT DAY#")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Abs_Edge', ascending=False)

print("\nTop 20 Strongest Edges (Any Combo + GDT Day):")
print("-" * 80)
print(results_df[['Condition', 'GDT_Day', 'Edge', 'Directional%', 'Count']].head(20).to_string(index=False))

# Best GDT days for each combo
print("\n" + "="*80)
print("BEST GDT DAY FOR EACH COMBINATION")
print("="*80)

for combo_name, _ in top_combos:
    combo_results = results_df[results_df['Condition'] == combo_name].sort_values('Edge', ascending=False)
    if len(combo_results) > 0:
        best = combo_results.iloc[0]
        print(f"{combo_name:30s}: {best['GDT_Day']:15s} | {best['Edge']:+6.2f}% edge (n={best['Count']:,})")

# ============================================================================
# STEP 7: Save Results & Visualize
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

output_file = os.path.join(OUTPUT_DIR, f'gap_wr_gdt_analysis_{group_label}.csv')
results_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Gap + WR + GDT Day# Analysis - {group_label}', fontsize=16, fontweight='bold')

# 1. Top edges overall
ax1 = axes[0, 0]
top_15 = results_df.head(15).sort_values('Edge')
colors = ['green' if x > 0 else 'red' for x in top_15['Edge']]
ax1.barh(range(len(top_15)), top_15['Edge'], color=colors)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([f"{row.Condition[:20]} ({row.GDT_Day})" for _, row in top_15.iterrows()], fontsize=7)
ax1.set_xlabel('Edge (%)')
ax1.set_title('Top 15 Edges (Combo + GDT Day)')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 2. Gap Against Trend + WR4 by GDT Day
ax2 = axes[0, 1]
gat_wr4 = results_df[results_df['Condition'] == 'Gap Against Trend + WR4'].sort_values('GDT_Day')
if len(gat_wr4) > 0:
    colors = ['green' if x > 0 else 'red' for x in gat_wr4['Edge']]
    ax2.barh(range(len(gat_wr4)), gat_wr4['Edge'], color=colors)
    ax2.set_yticks(range(len(gat_wr4)))
    ax2.set_yticklabels(gat_wr4['GDT_Day'])
    ax2.set_xlabel('Edge (%)')
    ax2.set_title('Gap Against Trend + WR4 by GDT Day')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

# 3. Buy Days vs Sell Days
ax3 = axes[1, 0]
results_df['Cycle'] = results_df['GDT_Day'].str.split('_').str[0]
cycle_avg = results_df.groupby('Cycle')['Edge'].mean().sort_values()
colors = ['green' if x > 0 else 'red' for x in cycle_avg.values]
ax3.bar(range(len(cycle_avg)), cycle_avg.values, color=colors)
ax3.set_xticks(range(len(cycle_avg)))
ax3.set_xticklabels(cycle_avg.index)
ax3.set_ylabel('Average Edge (%)')
ax3.set_title('Average Edge: Buy Cycle vs Sell Cycle')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.grid(axis='y', alpha=0.3)

# 4. WR4 Only by GDT Day
ax4 = axes[1, 1]
wr4_only = results_df[results_df['Condition'] == 'WR4 Only'].sort_values('GDT_Day')
if len(wr4_only) > 0:
    colors = ['green' if x > 0 else 'red' for x in wr4_only['Edge']]
    ax4.barh(range(len(wr4_only)), wr4_only['Edge'], color=colors)
    ax4.set_yticks(range(len(wr4_only)))
    ax4.set_yticklabels(wr4_only['GDT_Day'])
    ax4.set_xlabel('Edge (%)')
    ax4.set_title('WR4 Only by GDT Day')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

viz_file = os.path.join(OUTPUT_DIR, f'gap_wr_gdt_analysis_{group_label}.png')
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
print("\nKey Insights:")
print("- Which GDT days show strongest gap + WR edges?")
print("- Do counter-trend gaps work better on buy days or sell days?")
print("- Does WR4 performance change across GDT cycle?")
print("="*80)
