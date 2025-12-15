"""
============================================================================
GDT DAY# + GAPSTAT + GAP DIRECTION ANALYSIS
============================================================================
Analyzes gap behavior across GDT momentum cycle days.

Tests:
1. GDT Day# (Buy_Day1-4, Sell_Day1-4)
2. GapStat size (>1.0, >1.5, >2.0, >2.5)
3. Gap direction vs 5-day SMA trend (with trend vs against trend)
4. Gap size buckets (small, medium, large, xlarge)

No WR indicators - pure gap behavior analysis.

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
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = '/content/drive/MyDrive/StockData'
RESULTS_FILE = '/content/drive/MyDrive/backtest_results/daytype_classification_results.csv'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

print("="*80)
print("GDT DAY# + GAPSTAT + GAP DIRECTION ANALYSIS")
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
# STEP 3: Calculate GDT Day# and Gap Indicators
# ============================================================================
print("\nSTEP 3: Calculating GDT Day# and gap indicators...")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate GDT Day# logic and gap indicators"""

    df = df.copy().sort_values('Date')

    # Basic price data
    df['PrevClose'] = df['Close'].shift(1)
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100

    # Range vs usual (ADR comparison)
    df['ADR_20'] = df['Range_Pct'].rolling(20).mean()
    df['Range_vs_ADR'] = df['Range_Pct'] / df['ADR_20']

    # Categorize range size
    df['Range_Size'] = 'Normal'
    df.loc[df['Range_vs_ADR'] < 0.7, 'Range_Size'] = 'Narrow'
    df.loc[df['Range_vs_ADR'] > 1.3, 'Range_Size'] = 'Wide'
    df.loc[df['Range_vs_ADR'] > 1.6, 'Range_Size'] = 'Very_Wide'

    # Closing Range % (where close is within the day's range)
    # 0% = close at low, 100% = close at high
    df['Close_Range_Pct'] = ((df['Close'] - df['Low']) / df['Range']) * 100
    df['Close_Range_Pct'] = df['Close_Range_Pct'].fillna(50)  # Handle zero-range days

    # Categorize closing position
    df['Close_Position'] = 'Middle'
    df.loc[df['Close_Range_Pct'] <= 25, 'Close_Position'] = 'Bottom_Quarter'
    df.loc[df['Close_Range_Pct'] >= 75, 'Close_Position'] = 'Top_Quarter'
    df.loc[(df['Close_Range_Pct'] > 25) & (df['Close_Range_Pct'] < 40), 'Close_Position'] = 'Lower_Middle'
    df.loc[(df['Close_Range_Pct'] > 60) & (df['Close_Range_Pct'] < 75), 'Close_Position'] = 'Upper_Middle'

    # Binary indicators
    df['Close_Top_Half'] = (df['Close_Range_Pct'] >= 50).astype(int)
    df['Close_Top_Quarter'] = (df['Close_Range_Pct'] >= 75).astype(int)
    df['Close_Bottom_Quarter'] = (df['Close_Range_Pct'] <= 25).astype(int)

    # Short-term trend (5-day SMA)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['Trend_5D'] = np.where(df['Close'] > df['SMA_5'], 'Up', 'Down')

    # Gap calculations
    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100

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

    # Positive vs Negative GapStat
    df['GapStat_Positive'] = (df['GapStat'] > 1.0).astype(int)
    df['GapStat_Negative'] = (df['GapStat'] < -1.0).astype(int)
    df['GapStat_Extreme_Positive'] = (df['GapStat'] > 2.0).astype(int)
    df['GapStat_Extreme_Negative'] = (df['GapStat'] < -2.0).astype(int)

    # ========================================================================
    # GDT DAY# LOGIC
    # ========================================================================

    # ROC = close[0] - close[2]
    df['ROC'] = df['Close'] - df['Close'].shift(2)
    df['ROCprev'] = df['Close'].shift(1) - df['Close'].shift(3)
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

    mask = (df['BuyDay'].shift(1) == 1) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day2'

    mask = (df['BuyDay'].shift(2) == 1) & (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day3'

    mask = (df['BuyDay'].shift(3) == 1) & (df['BuyDay'].shift(2) == 0) & (df['SellDay'].shift(2) == 0) & \
           (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day4'

    # Sell cycle days
    df.loc[df['SellDay'] == 1, 'GDT_DayType'] = 'Sell_Day1'

    mask = (df['SellDay'].shift(1) == 1) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Sell_Day2'

    mask = (df['SellDay'].shift(2) == 1) & (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & \
           (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Sell_Day3'

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
print(f"  Overall baseline: {baseline:.1f}%")

# Show GDT Day distribution
print("\nGDT Day# Distribution:")
gdt_counts = predictive_df['GDT_DayType'].value_counts().sort_index()
for day_type, count in gdt_counts.items():
    if day_type:
        pct = count / len(predictive_df) * 100
        baseline_gdt = predictive_df[predictive_df['GDT_DayType'] == day_type]['Next_IsDirectional'].mean() * 100
        print(f"  {day_type:15s}: {count:5,} ({pct:4.1f}%) | Baseline: {baseline_gdt:5.1f}%")

# ============================================================================
# STEP 5: Analyze Gap Behavior by GDT Day#
# ============================================================================
print("\n" + "="*80)
print("STEP 5: GAP ANALYSIS BY GDT DAY#")
print("="*80)

def analyze_gap_by_gdt(df: pd.DataFrame, condition_name: str, condition_mask: pd.Series, gdt_day: str) -> Dict:
    """Analyze a gap condition on a specific GDT day"""

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

# Gap conditions to test
gap_conditions = [
    # Gap with trend vs against trend
    ('Gap With Trend', (predictive_df['Gap_With_Trend'] == 1)),
    ('Gap Against Trend', (predictive_df['Gap_Against_Trend'] == 1)),

    # Gap size
    ('Small Gap', (predictive_df['Gap_Size'] == 'Small')),
    ('Medium Gap', (predictive_df['Gap_Size'] == 'Medium')),
    ('Large Gap', (predictive_df['Gap_Size'] == 'Large')),
    ('XLarge Gap', (predictive_df['Gap_Size'] == 'XLarge')),

    # GapStat thresholds
    ('GapStat > 1.0', (predictive_df['GapStat_Above_1'] == 1)),
    ('GapStat > 1.5', (predictive_df['GapStat_Above_1_5'] == 1)),
    ('GapStat > 2.0', (predictive_df['GapStat_Above_2'] == 1)),
    ('GapStat > 2.5', (predictive_df['GapStat_Above_2_5'] == 1)),

    # GapStat direction
    ('GapStat Positive (>1)', (predictive_df['GapStat_Positive'] == 1)),
    ('GapStat Negative (<-1)', (predictive_df['GapStat_Negative'] == 1)),
    ('GapStat Extreme Pos (>2)', (predictive_df['GapStat_Extreme_Positive'] == 1)),
    ('GapStat Extreme Neg (<-2)', (predictive_df['GapStat_Extreme_Negative'] == 1)),

    # Gap direction
    ('Gap Up', (predictive_df['Gap_Up'] == 1)),
    ('Gap Down', (predictive_df['Gap_Down'] == 1)),

    # Range size vs usual
    ('Narrow Range Day', (predictive_df['Range_Size'] == 'Narrow')),
    ('Wide Range Day', (predictive_df['Range_Size'] == 'Wide')),
    ('Very Wide Range Day', (predictive_df['Range_Size'] == 'Very_Wide')),

    # Closing position
    ('Close Top Quarter', (predictive_df['Close_Position'] == 'Top_Quarter')),
    ('Close Bottom Quarter', (predictive_df['Close_Position'] == 'Bottom_Quarter')),
    ('Close Upper Middle', (predictive_df['Close_Position'] == 'Upper_Middle')),
    ('Close Lower Middle', (predictive_df['Close_Position'] == 'Lower_Middle')),
    ('Close Top Half', (predictive_df['Close_Top_Half'] == 1)),

    # Combinations
    ('Gap Against + Wide Range', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['Range_Size'].isin(['Wide', 'Very_Wide']))),
    ('Gap Against + Close Top Quarter', (predictive_df['Gap_Against_Trend'] == 1) & (predictive_df['Close_Top_Quarter'] == 1)),
    ('Gap With + Wide Range', (predictive_df['Gap_With_Trend'] == 1) & (predictive_df['Range_Size'].isin(['Wide', 'Very_Wide']))),
    ('GapStat >2 + Wide Range', (predictive_df['GapStat_Above_2'] == 1) & (predictive_df['Range_Size'].isin(['Wide', 'Very_Wide']))),
    ('GapStat >2 + Close Top Quarter', (predictive_df['GapStat_Above_2'] == 1) & (predictive_df['Close_Top_Quarter'] == 1)),
]

results = []

# Test each GDT day
gdt_days = ['Buy_Day1', 'Buy_Day2', 'Buy_Day3', 'Buy_Day4',
            'Sell_Day1', 'Sell_Day2', 'Sell_Day3', 'Sell_Day4']

print("\nTesting gap conditions across GDT days...")
for gdt_day in gdt_days:
    print(f"\n{gdt_day}:")
    print("-" * 80)

    for condition_name, condition_mask in gap_conditions:
        result = analyze_gap_by_gdt(predictive_df, condition_name, condition_mask, gdt_day)

        if result:
            results.append(result)
            print(f"  {result['Condition']:30s}: {result['Edge']:+6.2f}% edge | {result['Directional%']:5.1f}% (n={result['Count']:5,})")

# ============================================================================
# STEP 6: Rankings & Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TOP FINDINGS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Abs_Edge', ascending=False)

print("\nTop 25 Strongest Edges (Gap Condition + GDT Day):")
print("-" * 80)
print(results_df[['Condition', 'GDT_Day', 'Edge', 'Directional%', 'Count']].head(25).to_string(index=False))

# Best condition for each GDT day
print("\n" + "="*80)
print("BEST GAP CONDITION FOR EACH GDT DAY")
print("="*80)

for gdt_day in gdt_days:
    day_results = results_df[results_df['GDT_Day'] == gdt_day].sort_values('Edge', ascending=False)
    if len(day_results) > 0:
        best = day_results.iloc[0]
        print(f"{gdt_day:15s}: {best['Condition']:30s} | {best['Edge']:+6.2f}% edge (n={best['Count']:,})")

# Best GDT day for each condition
print("\n" + "="*80)
print("BEST GDT DAY FOR EACH GAP CONDITION")
print("="*80)

for condition_name, _ in gap_conditions:
    cond_results = results_df[results_df['Condition'] == condition_name].sort_values('Edge', ascending=False)
    if len(cond_results) > 0:
        best = cond_results.iloc[0]
        print(f"{condition_name:30s}: {best['GDT_Day']:15s} | {best['Edge']:+6.2f}% edge (n={best['Count']:,})")

# ============================================================================
# STEP 7: Save & Visualize
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

output_file = os.path.join(OUTPUT_DIR, f'gdt_gapstat_analysis_{group_label}.csv')
results_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'GDT Day# × GapStat × Gap Direction - {group_label}', fontsize=16, fontweight='bold')

# 1. Top edges
ax1 = axes[0, 0]
top_20 = results_df.head(20).sort_values('Edge')
colors = ['green' if x > 0 else 'red' for x in top_20['Edge']]
ax1.barh(range(len(top_20)), top_20['Edge'], color=colors)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels([f"{row.Condition[:18]} ({row.GDT_Day})" for _, row in top_20.iterrows()], fontsize=7)
ax1.set_xlabel('Edge (%)')
ax1.set_title('Top 20 Edges (Gap Condition + GDT Day)')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 2. Gap Against Trend across GDT days
ax2 = axes[0, 1]
gat = results_df[results_df['Condition'] == 'Gap Against Trend'].sort_values('GDT_Day')
if len(gat) > 0:
    colors = ['green' if x > 0 else 'red' for x in gat['Edge']]
    ax2.barh(range(len(gat)), gat['Edge'], color=colors)
    ax2.set_yticks(range(len(gat)))
    ax2.set_yticklabels(gat['GDT_Day'])
    ax2.set_xlabel('Edge (%)')
    ax2.set_title('Gap Against Trend by GDT Day')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

# 3. GapStat > 2.0 across GDT days
ax3 = axes[1, 0]
gs2 = results_df[results_df['Condition'] == 'GapStat > 2.0'].sort_values('GDT_Day')
if len(gs2) > 0:
    colors = ['green' if x > 0 else 'red' for x in gs2['Edge']]
    ax3.barh(range(len(gs2)), gs2['Edge'], color=colors)
    ax3.set_yticks(range(len(gs2)))
    ax3.set_yticklabels(gs2['GDT_Day'])
    ax3.set_xlabel('Edge (%)')
    ax3.set_title('GapStat > 2.0 by GDT Day')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)

# 4. Buy Days vs Sell Days average
ax4 = axes[1, 1]
results_df['Cycle'] = results_df['GDT_Day'].str.split('_').str[0]
cycle_avg = results_df.groupby('Cycle')['Edge'].mean().sort_values()
colors = ['green' if x > 0 else 'red' for x in cycle_avg.values]
ax4.bar(range(len(cycle_avg)), cycle_avg.values, color=colors)
ax4.set_xticks(range(len(cycle_avg)))
ax4.set_xticklabels(cycle_avg.index)
ax4.set_ylabel('Average Edge (%)')
ax4.set_title('Average Gap Edge: Buy vs Sell Cycle')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

viz_file = os.path.join(OUTPUT_DIR, f'gdt_gapstat_analysis_{group_label}.png')
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
print("- Which GDT days favor gap with trend vs against trend?")
print("- What GapStat threshold works best on each GDT day?")
print("- Which gap sizes predict directional days on which GDT days?")
print("="*80)
