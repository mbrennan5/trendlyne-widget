"""
============================================================================
TECHNICAL ANALYSIS PREDICTOR - Predict Directional Days
============================================================================
Analyzes if technical indicators can predict next-day directional moves.

Tests indicators like:
- NR4, NR7 (Narrow Range days)
- ADR% Z-Score (volatility contraction)
- Bollinger Band Width
- ATR Contraction
- Volume patterns
- Gap size
- Consecutive range days
- Previous day type patterns

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
print("TECHNICAL ANALYSIS PREDICTOR FOR DIRECTIONAL DAYS")
print("="*80)

# ============================================================================
# STEP 1: Load Day Type Results
# ============================================================================
print("\nSTEP 1: Loading day type classification results...")
daytype_df = pd.read_csv(RESULTS_FILE)
daytype_df['Date'] = pd.to_datetime(daytype_df['Date'])
daytype_df = daytype_df.sort_values(['Symbol', 'Date'])

# Create binary target: 1 if Directional (DNP or DWP), 0 otherwise
daytype_df['IsDirectional'] = daytype_df['DayType'].str.contains('Directional', na=False).astype(int)
daytype_df['IsDNP'] = (daytype_df['DayType'] == 'DNP (Directional No Pullbacks)').astype(int)
daytype_df['IsRange'] = (daytype_df['DayType'] == 'RANGE DAY').astype(int)

print(f"✓ Loaded {len(daytype_df):,} classified days for {daytype_df['Symbol'].nunique()} symbols")

# Show available symbols
all_symbols = sorted(daytype_df['Symbol'].unique())
print(f"\nAvailable symbols ({len(all_symbols)}):")
print(f"  {', '.join(all_symbols)}")

# ============================================================================
# USER INPUT: Symbol Selection
# ============================================================================
print("\n" + "="*80)
print("SYMBOL SELECTION")
print("="*80)
symbols_input = input("Enter symbols to test (comma-separated, or 'ALL'): ")
SYMBOLS = None if symbols_input.upper().strip() == 'ALL' else [s.strip().upper() for s in symbols_input.split(',')]

if SYMBOLS:
    # Filter daytype data to selected symbols
    daytype_df = daytype_df[daytype_df['Symbol'].isin(SYMBOLS)].copy()
    print(f"\n✓ Filtered to {len(SYMBOLS)} symbol(s): {', '.join(SYMBOLS)}")
    print(f"  {len(daytype_df):,} classified days")
else:
    SYMBOLS = all_symbols
    print(f"\n✓ Using ALL {len(SYMBOLS)} symbols")

# ============================================================================
# STEP 2: Load Raw Price Data and Calculate Daily OHLCV
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading raw price data and calculating daily OHLCV...")
print("="*80)

def load_symbol_daily_data(symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """Load all year files for a symbol and create daily OHLCV bars"""

    # Find all files for this symbol
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

    # Load and concatenate all files
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

# User input for year range
print("\n" + "-"*80)
print("YEAR RANGE SELECTION")
print("-"*80)
start_year_input = input("Start YEAR for price data (e.g., 2020, or press Enter for all): ")
START_YEAR = None if start_year_input.strip() == '' else int(start_year_input.strip())
end_year_input = input("End YEAR for price data (e.g., 2025, or press Enter for all): ")
END_YEAR = None if end_year_input.strip() == '' else int(end_year_input.strip())

print(f"\nLoading price data for {len(SYMBOLS)} symbol(s)...")

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
# STEP 3: Calculate Technical Indicators
# ============================================================================
print("\nSTEP 3: Calculating technical indicators...")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for a symbol's data"""

    df = df.copy().sort_values('Date')

    # Daily Range
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100

    # True Range (includes gaps)
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'PrevClose']].max(axis=1) - df[['Low', 'PrevClose']].min(axis=1)
    df['TR_Pct'] = (df['TR'] / df['Close']) * 100

    # Narrow Range indicators (NR2, NR3, NR4, NR7)
    df['NR2'] = (df['Range'] == df['Range'].rolling(2).min()).astype(int)
    df['NR3'] = (df['Range'] == df['Range'].rolling(3).min()).astype(int)
    df['NR4'] = (df['Range'] == df['Range'].rolling(4).min()).astype(int)
    df['NR7'] = (df['Range'] == df['Range'].rolling(7).min()).astype(int)

    # Wide Range indicators (opposite of NR - range is LARGEST in period)
    df['WR2'] = (df['Range'] == df['Range'].rolling(2).max()).astype(int)
    df['WR3'] = (df['Range'] == df['Range'].rolling(3).max()).astype(int)
    df['WR4'] = (df['Range'] == df['Range'].rolling(4).max()).astype(int)
    df['WR7'] = (df['Range'] == df['Range'].rolling(7).max()).astype(int)

    # ADR (Average Daily Range) - 20 day
    df['ADR_20'] = df['Range_Pct'].rolling(20).mean()
    df['ADR_StdDev'] = df['Range_Pct'].rolling(20).std()

    # ADR Z-Score (current range vs average)
    df['ADR_ZScore'] = (df['Range_Pct'] - df['ADR_20']) / df['ADR_StdDev']

    # Volatility Contraction (range < 0.7 * average range)
    df['Vol_Contraction'] = (df['Range_Pct'] < (df['ADR_20'] * 0.7)).astype(int)

    # Volatility Expansion (range > 1.3 * average range) - MOMENTUM
    df['Vol_Expansion'] = (df['Range_Pct'] > (df['ADR_20'] * 1.3)).astype(int)

    # ATR (Average True Range) - 14 day
    df['ATR_14'] = df['TR_Pct'].rolling(14).mean()
    df['ATR_StdDev'] = df['TR_Pct'].rolling(14).std()
    df['ATR_ZScore'] = (df['TR_Pct'] - df['ATR_14']) / df['ATR_StdDev']

    # Bollinger Band Width (20-day, 2 std)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(20).std() * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']) * 100
    df['BB_Width_ZScore'] = (df['BB_Width'] - df['BB_Width'].rolling(20).mean()) / df['BB_Width'].rolling(20).std()

    # Volume indicators
    df['Avg_Volume_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume_20']
    df['High_Volume'] = (df['Volume_Ratio'] > 1.5).astype(int)
    df['Low_Volume'] = (df['Volume_Ratio'] < 0.7).astype(int)

    # Gap size
    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100
    df['Gap_Up'] = (df['Gap_Pct'] > 0.5).astype(int)
    df['Gap_Down'] = (df['Gap_Pct'] < -0.5).astype(int)

    # Large Gap indicators (momentum)
    df['Large_Gap_Up'] = (df['Gap_Pct'] > 1.5).astype(int)
    df['Large_Gap_Down'] = (df['Gap_Pct'] < -1.5).astype(int)
    df['Large_Gap_Any'] = ((df['Gap_Pct'].abs()) > 1.5).astype(int)

    # Consecutive range/directional days (will be added after merge)

    return df

# Calculate indicators per symbol
indicator_data = []
for symbol in SYMBOLS:
    symbol_price = price_df[price_df['Symbol'] == symbol].copy()
    symbol_indicators = calculate_indicators(symbol_price)
    indicator_data.append(symbol_indicators)

indicators_df = pd.concat(indicator_data, ignore_index=True)

print(f"✓ Calculated indicators for {len(indicators_df):,} daily bars")

# ============================================================================
# STEP 4: Merge with Day Types and Create Predictive Dataset
# ============================================================================
print("\nSTEP 4: Creating predictive dataset...")

# Merge indicators with day types
merged_df = indicators_df.merge(
    daytype_df[['Symbol', 'Date', 'DayType', 'IsDirectional', 'IsDNP', 'IsRange']],
    on=['Symbol', 'Date'],
    how='inner'
)

print(f"✓ Merged {len(merged_df):,} days with both price and classification data")

# Add consecutive patterns
merged_df = merged_df.sort_values(['Symbol', 'Date'])

def add_consecutive_patterns(df):
    df = df.copy()
    df['Prev_DayType'] = df.groupby('Symbol')['DayType'].shift(1)
    df['Prev_IsDirectional'] = df.groupby('Symbol')['IsDirectional'].shift(1)
    df['Prev_IsRange'] = df.groupby('Symbol')['IsRange'].shift(1)

    # Count consecutive range days before current day
    df['Consec_Range'] = 0
    for symbol in df['Symbol'].unique():
        mask = df['Symbol'] == symbol
        symbol_data = df[mask].copy()

        consec = []
        count = 0
        for is_range in symbol_data['IsRange']:
            consec.append(count)
            if is_range:
                count += 1
            else:
                count = 0

        df.loc[mask, 'Consec_Range'] = consec

    # Binary indicator for high consecutive range (3+ days)
    df['High_Consec_Range'] = (df['Consec_Range'] >= 3).astype(int)

    return df

merged_df = add_consecutive_patterns(merged_df)

# Create next-day target (what we want to predict)
merged_df['Next_IsDirectional'] = merged_df.groupby('Symbol')['IsDirectional'].shift(-1)
merged_df['Next_IsDNP'] = merged_df.groupby('Symbol')['IsDNP'].shift(-1)
merged_df['Next_DayType'] = merged_df.groupby('Symbol')['DayType'].shift(-1)

# Remove rows without next-day data
predictive_df = merged_df.dropna(subset=['Next_IsDirectional']).copy()

print(f"✓ Created predictive dataset with {len(predictive_df):,} samples")

# Create filename suffix based on symbol selection
if len(SYMBOLS) <= 3:
    symbol_suffix = "_" + "_".join(SYMBOLS)
elif len(SYMBOLS) == len(all_symbols):
    symbol_suffix = "_ALL"
else:
    symbol_suffix = f"_{len(SYMBOLS)}symbols"

print(f"\nAnalyzing: {', '.join(SYMBOLS)}")
print(f"Filename suffix: {symbol_suffix}")

# ============================================================================
# STEP 5: Analyze Predictive Power of Each Indicator
# ============================================================================
print("\n" + "="*80)
print("STEP 5: ANALYZING PREDICTIVE POWER")
print("="*80)

# Define indicator columns to test
indicator_columns = [
    # Narrow Range indicators
    'NR2', 'NR3', 'NR4', 'NR7',
    # Wide Range indicators (MOMENTUM)
    'WR2', 'WR3', 'WR4', 'WR7',
    # Volatility
    'Vol_Contraction', 'Vol_Expansion',
    # Volume
    'High_Volume', 'Low_Volume',
    # Gaps
    'Gap_Up', 'Gap_Down', 'Large_Gap_Up', 'Large_Gap_Down', 'Large_Gap_Any',
    # Previous day patterns
    'Prev_IsRange', 'Prev_IsDirectional',
    # Consecutive patterns
    'High_Consec_Range'
]

continuous_indicators = [
    'ADR_ZScore', 'ATR_ZScore', 'BB_Width_ZScore', 'Volume_Ratio',
    'Gap_Pct', 'Consec_Range', 'Range_Pct', 'TR_Pct'
]

results = []

print("\nBinary Indicators (comparing when indicator=1 vs indicator=0):")
print("-"*80)

for indicator in indicator_columns:
    # Skip if too many NaN
    if predictive_df[indicator].isna().sum() / len(predictive_df) > 0.5:
        continue

    # When indicator is TRUE (1)
    when_true = predictive_df[predictive_df[indicator] == 1]
    # When indicator is FALSE (0)
    when_false = predictive_df[predictive_df[indicator] == 0]

    if len(when_true) < 10 or len(when_false) < 10:
        continue

    # Directional rate when indicator is true
    dir_rate_true = when_true['Next_IsDirectional'].mean() * 100
    # Directional rate when indicator is false
    dir_rate_false = when_false['Next_IsDirectional'].mean() * 100

    # Baseline (overall directional rate)
    baseline = predictive_df['Next_IsDirectional'].mean() * 100

    # Edge (difference from baseline)
    edge = dir_rate_true - baseline

    # Count
    count_true = len(when_true)
    count_false = len(when_false)

    results.append({
        'Indicator': indicator,
        'Type': 'Binary',
        'When_True_Count': count_true,
        'Directional%_When_True': dir_rate_true,
        'Directional%_When_False': dir_rate_false,
        'Baseline%': baseline,
        'Edge': edge,
        'Abs_Edge': abs(edge)
    })

    print(f"{indicator:25s}: {dir_rate_true:5.1f}% directional (n={count_true:4}) vs {dir_rate_false:5.1f}% when false | Edge: {edge:+5.1f}%")

print("\n" + "-"*80)
print("Continuous Indicators (comparing low vs high values):")
print("-"*80)

for indicator in continuous_indicators:
    # Skip if too many NaN
    valid_data = predictive_df[[indicator, 'Next_IsDirectional']].dropna()

    if len(valid_data) < 50:
        continue

    try:
        # Split into quartiles
        valid_data['Quartile'] = pd.qcut(valid_data[indicator], q=4, labels=False, duplicates='drop')

        # Get min and max quartile numbers (in case duplicates reduced number of bins)
        min_q = valid_data['Quartile'].min()
        max_q = valid_data['Quartile'].max()

        # Directional rate in bottom quartile (low values)
        q1_data = valid_data[valid_data['Quartile'] == min_q]
        # Directional rate in top quartile (high values)
        q4_data = valid_data[valid_data['Quartile'] == max_q]

        if len(q1_data) < 10 or len(q4_data) < 10:
            continue
    except (ValueError, TypeError):
        # Skip if can't create quartiles (too many duplicates)
        continue

    dir_rate_low = q1_data['Next_IsDirectional'].mean() * 100
    dir_rate_high = q4_data['Next_IsDirectional'].mean() * 100

    baseline = valid_data['Next_IsDirectional'].mean() * 100

    # Use the quartile with bigger edge
    edge_low = dir_rate_low - baseline
    edge_high = dir_rate_high - baseline

    if abs(edge_low) > abs(edge_high):
        edge = edge_low
        best_quartile = 'Q1 (Low)'
        best_rate = dir_rate_low
    else:
        edge = edge_high
        best_quartile = 'Q4 (High)'
        best_rate = dir_rate_high

    results.append({
        'Indicator': indicator,
        'Type': 'Continuous',
        'When_True_Count': len(q1_data) if abs(edge_low) > abs(edge_high) else len(q4_data),
        'Directional%_When_True': best_rate,
        'Directional%_When_False': dir_rate_high if abs(edge_low) > abs(edge_high) else dir_rate_low,
        'Baseline%': baseline,
        'Edge': edge,
        'Abs_Edge': abs(edge)
    })

    print(f"{indicator:25s}: Q1={dir_rate_low:5.1f}%, Q4={dir_rate_high:5.1f}% | Best: {best_quartile} | Edge: {edge:+5.1f}%")

# ============================================================================
# STEP 6: Rank Indicators by Predictive Power
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TOP PREDICTIVE INDICATORS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Abs_Edge', ascending=False)

print("\nRanked by Absolute Edge:")
print(results_df[['Indicator', 'Type', 'Directional%_When_True', 'Edge', 'When_True_Count']].to_string(index=False))

# Save results
indicators_file = os.path.join(OUTPUT_DIR, f'predictive_indicators_analysis{symbol_suffix}.csv')
results_df.to_csv(indicators_file, index=False)
print(f"\n✓ Saved to: {indicators_file}")

# ============================================================================
# STEP 7: Combination Analysis - MOMENTUM & CONTRACTION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: COMBINATION ANALYSIS")
print("="*80)

print("\n--- MOMENTUM COMBINATIONS (High Range + Gaps + Volume) ---")

momentum_combinations = [
    # Wide Range + Gaps
    ('WR2', 'Large_Gap_Any'),
    ('WR3', 'Large_Gap_Any'),
    ('WR4', 'Gap_Up'),
    ('WR7', 'Gap_Up'),
    # Wide Range + Volume
    ('WR2', 'High_Volume'),
    ('WR3', 'High_Volume'),
    ('WR4', 'High_Volume'),
    # Volatility Expansion + Gaps
    ('Vol_Expansion', 'Large_Gap_Any'),
    ('Vol_Expansion', 'Gap_Up'),
    ('Vol_Expansion', 'High_Volume'),
    # Multiple momentum signals
    ('Large_Gap_Any', 'High_Volume'),
    ('Gap_Up', 'High_Volume'),
]

print("\n--- CONTRACTION COMBINATIONS (NR + Low Volume + Consecutive Range) ---")

contraction_combinations = [
    # NR variants
    ('NR2', 'Low_Volume'),
    ('NR3', 'Low_Volume'),
    ('NR4', 'Low_Volume'),
    ('NR2', 'Vol_Contraction'),
    ('NR3', 'Vol_Contraction'),
    # Consecutive range breakouts
    ('High_Consec_Range', 'Vol_Contraction'),
    ('High_Consec_Range', 'Low_Volume'),
    ('High_Consec_Range', 'NR4'),
    ('Prev_IsRange', 'NR4'),
    ('Prev_IsRange', 'Vol_Contraction'),
]

# Combine all combinations
combinations = momentum_combinations + contraction_combinations

combo_results = []

# Process momentum combinations
for ind1, ind2 in momentum_combinations:
    # Both indicators true
    both_true = predictive_df[(predictive_df[ind1] == 1) & (predictive_df[ind2] == 1)]

    if len(both_true) < 10:
        continue

    dir_rate = both_true['Next_IsDirectional'].mean() * 100
    baseline = predictive_df['Next_IsDirectional'].mean() * 100
    edge = dir_rate - baseline

    combo_results.append({
        'Type': 'Momentum',
        'Combination': f"{ind1} + {ind2}",
        'Count': len(both_true),
        'Directional%': dir_rate,
        'Edge': edge
    })

    print(f"  {ind1:20s} + {ind2:20s}: {dir_rate:5.1f}% (n={len(both_true):3}) | Edge: {edge:+5.1f}%")

# Process contraction combinations
for ind1, ind2 in contraction_combinations:
    # Both indicators true
    both_true = predictive_df[(predictive_df[ind1] == 1) & (predictive_df[ind2] == 1)]

    if len(both_true) < 10:
        continue

    dir_rate = both_true['Next_IsDirectional'].mean() * 100
    baseline = predictive_df['Next_IsDirectional'].mean() * 100
    edge = dir_rate - baseline

    combo_results.append({
        'Type': 'Contraction',
        'Combination': f"{ind1} + {ind2}",
        'Count': len(both_true),
        'Directional%': dir_rate,
        'Edge': edge
    })

    print(f"  {ind1:20s} + {ind2:20s}: {dir_rate:5.1f}% (n={len(both_true):3}) | Edge: {edge:+5.1f}%")

combo_df = pd.DataFrame(combo_results)
combo_df = combo_df.sort_values('Edge', ascending=False)

print("\n" + "="*80)
print("TOP COMBINATIONS (sorted by Edge)")
print("="*80)
print(combo_df.head(15).to_string(index=False))

combo_file = os.path.join(OUTPUT_DIR, f'combination_analysis{symbol_suffix}.csv')
combo_df.to_csv(combo_file, index=False)

# ============================================================================
# STEP 8: Visualizations
# ============================================================================
print("\n" + "="*80)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Technical Indicator Predictive Power Analysis', fontsize=16, fontweight='bold')

# 1. Top indicators by edge
ax1 = axes[0, 0]
top_10 = results_df.head(10).sort_values('Edge')
ax1.barh(range(len(top_10)), top_10['Edge'], color=['green' if x > 0 else 'red' for x in top_10['Edge']])
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels(top_10['Indicator'])
ax1.set_xlabel('Edge (% above/below baseline)')
ax1.set_title('Top 10 Predictive Indicators')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 2. Sample counts
ax2 = axes[0, 1]
top_10_counts = results_df.head(10).sort_values('When_True_Count')
ax2.barh(range(len(top_10_counts)), top_10_counts['When_True_Count'], color='steelblue')
ax2.set_yticks(range(len(top_10_counts)))
ax2.set_yticklabels(top_10_counts['Indicator'])
ax2.set_xlabel('Number of Occurrences')
ax2.set_title('Sample Sizes for Top Indicators')
ax2.grid(axis='x', alpha=0.3)

# 3. Directional% when indicator is true
ax3 = axes[1, 0]
baseline_rate = predictive_df['Next_IsDirectional'].mean() * 100
top_10_dir = results_df.head(10).sort_values('Directional%_When_True')
ax3.barh(range(len(top_10_dir)), top_10_dir['Directional%_When_True'], color='orange')
ax3.axvline(x=baseline_rate, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_rate:.1f}%)')
ax3.set_yticks(range(len(top_10_dir)))
ax3.set_yticklabels(top_10_dir['Indicator'])
ax3.set_xlabel('Directional % (Next Day)')
ax3.set_title('Directional Rate When Indicator Fires')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# 4. Combination analysis
ax4 = axes[1, 1]
if len(combo_df) > 0:
    top_combos = combo_df.head(8).sort_values('Edge')
    ax4.barh(range(len(top_combos)), top_combos['Edge'], color=['green' if x > 0 else 'red' for x in top_combos['Edge']])
    ax4.set_yticks(range(len(top_combos)))
    ax4.set_yticklabels(top_combos['Combination'], fontsize=8)
    ax4.set_xlabel('Edge (% above/below baseline)')
    ax4.set_title('Top Indicator Combinations')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

viz_file = os.path.join(OUTPUT_DIR, f'predictive_indicators_charts{symbol_suffix}.png')
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"✓ Visualizations saved to: {viz_file}")
plt.show()

# ============================================================================
# STEP 9: Export Full Dataset for Further Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 9: EXPORTING FULL DATASET")
print("="*80)

# Export the predictive dataset
export_df = predictive_df[[
    'Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Range_Pct', 'TR_Pct',
    # Narrow Range
    'NR2', 'NR3', 'NR4', 'NR7',
    # Wide Range (MOMENTUM)
    'WR2', 'WR3', 'WR4', 'WR7',
    # Volatility indicators
    'ADR_ZScore', 'ATR_ZScore', 'BB_Width_ZScore',
    'Vol_Contraction', 'Vol_Expansion',
    # Volume
    'Volume_Ratio', 'High_Volume', 'Low_Volume',
    # Gaps
    'Gap_Pct', 'Gap_Up', 'Gap_Down', 'Large_Gap_Up', 'Large_Gap_Down', 'Large_Gap_Any',
    # Previous day patterns
    'Prev_DayType', 'Prev_IsRange', 'Prev_IsDirectional',
    'Consec_Range', 'High_Consec_Range',
    # Current day type
    'DayType', 'IsDirectional', 'IsDNP', 'IsRange',
    # Next day (target)
    'Next_DayType', 'Next_IsDirectional', 'Next_IsDNP'
]].copy()

dataset_file = os.path.join(OUTPUT_DIR, f'predictive_dataset_with_indicators{symbol_suffix}.csv')
export_df.to_csv(dataset_file, index=False)
print(f"✓ Full dataset exported to: {dataset_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - KEY FINDINGS")
print("="*80)

print(f"\nSymbols tested: {', '.join(SYMBOLS)}")
print(f"Total samples: {len(predictive_df):,} trading days")

baseline = predictive_df['Next_IsDirectional'].mean() * 100
print(f"\nBaseline (overall directional rate): {baseline:.1f}%")

if len(results_df) > 0:
    best = results_df.iloc[0]
    print(f"\nBest Single Indicator: {best['Indicator']}")
    print(f"  - Directional rate when true: {best['Directional%_When_True']:.1f}%")
    print(f"  - Edge: {best['Edge']:+.1f}%")
    print(f"  - Sample size: {best['When_True_Count']:.0f} occurrences")

if len(combo_df) > 0:
    best_combo = combo_df.iloc[0]
    print(f"\nBest Combination: {best_combo['Combination']}")
    print(f"  - Directional rate: {best_combo['Directional%']:.1f}%")
    print(f"  - Edge: {best_combo['Edge']:+.1f}%")
    print(f"  - Sample size: {best_combo['Count']:.0f} occurrences")

print("\n" + "="*80)
print("FILES GENERATED:")
print("="*80)
print(f"  1. {indicators_file}")
print(f"  2. {combo_file}")
print(f"  3. {viz_file}")
print(f"  4. {dataset_file}")

try:
    from google.colab import files
    print("\nDownloading files...")
    files.download(indicators_file)
    files.download(combo_file)
    files.download(viz_file)
    files.download(dataset_file)
    print("✓ Files downloaded!")
except:
    print("\nNot in Colab - files saved to Drive only")

print("\n" + "="*80)
