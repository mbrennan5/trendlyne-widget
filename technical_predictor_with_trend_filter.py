"""
============================================================================
TECHNICAL PREDICTOR - WITH TREND FILTERS (200-DAY MA)
============================================================================
Tests technical indicators with trend filters:
- Stock above/below 200-day MA
- SPY above/below 200-day MA
- Combined regimes (both above, both below, mixed)

Hypothesis: Momentum indicators work better when stocks are in uptrends.

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
print("TECHNICAL PREDICTOR - WITH TREND FILTERS (200-DAY MA)")
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
daytype_df['IsDNP'] = (daytype_df['DayType'] == 'DNP (Directional No Pullbacks)').astype(int)
daytype_df['IsRange'] = (daytype_df['DayType'] == 'RANGE DAY').astype(int)

print(f"✓ Loaded {len(daytype_df):,} classified days for {daytype_df['Symbol'].nunique()} symbols")

all_symbols = sorted(daytype_df['Symbol'].unique())

# ============================================================================
# DEFINE SYMBOL GROUPS
# ============================================================================
SYMBOL_GROUPS = {
    'Volatility ETFs': ['SOXL', 'TQQQ', 'TSLL'],

    'High Beta Tech': ['NVDA', 'TSLA', 'AMD', 'PLTR', 'SNOW', 'NET', 'DDOG',
                       'SMCI', 'MSTR', 'RIVN', 'ABNB', 'UBER'],

    'Major Indices': ['SPY', 'QQQ', 'IWM', 'DIA'],

    'Semiconductor': ['SMH', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'MU', 'MRVL'],

    'Mega Cap Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX'],
}

# Filter to only available symbols
SYMBOL_GROUPS_FILTERED = {}
for group_name, symbols in SYMBOL_GROUPS.items():
    available = [s for s in symbols if s in all_symbols]
    if available:
        SYMBOL_GROUPS_FILTERED[group_name] = available

print("\nSymbol Groups:")
for group_name, symbols in SYMBOL_GROUPS_FILTERED.items():
    print(f"  {group_name:20s}: {', '.join(symbols)}")

# ============================================================================
# USER INPUT: Select Symbol Group
# ============================================================================
print("\n" + "="*80)
print("SYMBOL GROUP SELECTION")
print("="*80)
print("Available groups:")
for i, group_name in enumerate(SYMBOL_GROUPS_FILTERED.keys(), 1):
    print(f"  {i}. {group_name}")

group_input = input("\nEnter group number (or 'ALL' for all symbols): ").strip()

if group_input.upper() == 'ALL':
    SYMBOLS = all_symbols
    group_label = 'All_Symbols'
    print(f"\n✓ Using ALL {len(SYMBOLS)} symbols")
else:
    try:
        group_idx = int(group_input) - 1
        group_name = list(SYMBOL_GROUPS_FILTERED.keys())[group_idx]
        SYMBOLS = SYMBOL_GROUPS_FILTERED[group_name]
        group_label = group_name.replace(' ', '_')
        print(f"\n✓ Selected: {group_name}")
        print(f"  Symbols: {', '.join(SYMBOLS)}")
    except:
        print("\n⚠ Invalid selection. Using ALL symbols.")
        SYMBOLS = all_symbols
        group_label = 'All_Symbols'

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
    combined['Time'] = combined['datetime'].dt.time

    # Create daily OHLCV
    daily = combined.groupby('Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).reset_index()

    # Get first 30-minute bar volume
    first_30min = combined[combined['Time'] <= pd.Timestamp('10:00:00').time()]
    first_30min_vol = first_30min.groupby('Date')['Volume'].first().reset_index()
    first_30min_vol.columns = ['Date', 'First_30Min_Volume']

    daily = daily.merge(first_30min_vol, on='Date', how='left')
    daily['First_30Min_Volume'] = daily['First_30Min_Volume'].fillna(0)

    daily['Date'] = pd.to_datetime(daily['Date'])
    daily['Symbol'] = symbol

    return daily

# Year range input
start_year_input = input("Start YEAR (e.g., 2020, or press Enter for all): ").strip()
START_YEAR = None if start_year_input == '' else int(start_year_input)
end_year_input = input("End YEAR (e.g., 2025, or press Enter for all): ").strip()
END_YEAR = None if end_year_input == '' else int(end_year_input)

# Load selected symbols + SPY (needed for trend filter)
symbols_to_load = list(set(SYMBOLS + ['SPY']))
print(f"\nLoading price data for {len(symbols_to_load)} symbols (including SPY for trend filter)...")

all_daily_data = []
for symbol in symbols_to_load:
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
# STEP 3: Calculate Technical Indicators + 200-Day MA
# ============================================================================
print("\nSTEP 3: Calculating technical indicators and 200-day moving averages...")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators including 200-day MA"""

    df = df.copy().sort_values('Date')

    # 200-day moving average (TREND FILTER)
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Above_200MA'] = (df['Close'] > df['SMA_200']).astype(int)

    # Daily Range
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100

    # True Range
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'PrevClose']].max(axis=1) - df[['Low', 'PrevClose']].min(axis=1)
    df['TR_Pct'] = (df['TR'] / df['Close']) * 100

    # Narrow Range
    df['NR4'] = (df['Range'] == df['Range'].rolling(4).min()).astype(int)
    df['NR7'] = (df['Range'] == df['Range'].rolling(7).min()).astype(int)

    # Wide Range (MOMENTUM)
    df['WR2'] = (df['Range'] == df['Range'].rolling(2).max()).astype(int)
    df['WR4'] = (df['Range'] == df['Range'].rolling(4).max()).astype(int)
    df['WR7'] = (df['Range'] == df['Range'].rolling(7).max()).astype(int)

    # Volatility
    df['ADR_20'] = df['Range_Pct'].rolling(20).mean()
    df['Vol_Expansion'] = (df['Range_Pct'] > (df['ADR_20'] * 1.3)).astype(int)

    # Volume
    df['Avg_Volume_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume_20']
    df['High_Volume'] = (df['Volume_Ratio'] > 1.5).astype(int)

    # Gaps
    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100
    df['Large_Gap_Any'] = ((df['Gap_Pct'].abs()) > 1.5).astype(int)
    df['Gap_Up'] = (df['Gap_Pct'] > 0.5).astype(int)

    # GapStat
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

    df['Extreme_GapStat'] = ((df['GapStat'].abs()) > 2).astype(int)

    # First 30-min volume
    if 'First_30Min_Volume' in df.columns:
        df['Avg_First_30Min_Vol'] = df['First_30Min_Volume'].rolling(20, min_periods=1).mean()
        df['First_30Min_Vol_Ratio'] = df['First_30Min_Volume'] / df['Avg_First_30Min_Vol']
        df['High_First_30Min_Vol'] = (df['First_30Min_Vol_Ratio'] > 1.5).astype(int)
    else:
        df['First_30Min_Vol_Ratio'] = np.nan
        df['High_First_30Min_Vol'] = 0

    return df

# Calculate indicators per symbol
indicator_data = []
for symbol in symbols_to_load:
    symbol_price = price_df[price_df['Symbol'] == symbol].copy()
    if not symbol_price.empty:
        symbol_indicators = calculate_indicators(symbol_price)
        indicator_data.append(symbol_indicators)

indicators_df = pd.concat(indicator_data, ignore_index=True)

print(f"✓ Calculated indicators for {len(indicators_df):,} daily bars")

# ============================================================================
# STEP 4: Extract SPY 200-Day MA as Market Trend Filter
# ============================================================================
print("\nSTEP 4: Creating SPY trend filter...")

spy_data = indicators_df[indicators_df['Symbol'] == 'SPY'][['Date', 'Above_200MA']].copy()
spy_data = spy_data.rename(columns={'Above_200MA': 'SPY_Above_200MA'})

print(f"✓ Extracted SPY 200-day MA status for {len(spy_data):,} days")

# ============================================================================
# STEP 5: Merge Everything Together
# ============================================================================
print("\nSTEP 5: Creating predictive dataset with trend filters...")

# Merge indicators with day types
merged_df = indicators_df.merge(
    daytype_df[['Symbol', 'Date', 'DayType', 'IsDirectional']],
    on=['Symbol', 'Date'],
    how='inner'
)

# Add SPY trend filter
merged_df = merged_df.merge(spy_data, on='Date', how='left')
merged_df['SPY_Above_200MA'] = merged_df['SPY_Above_200MA'].fillna(0).astype(int)

# Create next-day target
merged_df = merged_df.sort_values(['Symbol', 'Date'])
merged_df['Next_IsDirectional'] = merged_df.groupby('Symbol')['IsDirectional'].shift(-1)
predictive_df = merged_df.dropna(subset=['Next_IsDirectional']).copy()

# Create trend regime categories
predictive_df['Trend_Regime'] = 'Unknown'
predictive_df.loc[(predictive_df['Above_200MA'] == 1) & (predictive_df['SPY_Above_200MA'] == 1), 'Trend_Regime'] = 'Both_Above_200MA'
predictive_df.loc[(predictive_df['Above_200MA'] == 0) & (predictive_df['SPY_Above_200MA'] == 0), 'Trend_Regime'] = 'Both_Below_200MA'
predictive_df.loc[(predictive_df['Above_200MA'] == 1) & (predictive_df['SPY_Above_200MA'] == 0), 'Trend_Regime'] = 'Stock_Above_SPY_Below'
predictive_df.loc[(predictive_df['Above_200MA'] == 0) & (predictive_df['SPY_Above_200MA'] == 1), 'Trend_Regime'] = 'Stock_Below_SPY_Above'

print(f"✓ Created predictive dataset with {len(predictive_df):,} samples")

print("\nTrend Regime Distribution:")
for regime in ['Both_Above_200MA', 'Both_Below_200MA', 'Stock_Above_SPY_Below', 'Stock_Below_SPY_Above']:
    count = len(predictive_df[predictive_df['Trend_Regime'] == regime])
    pct = count / len(predictive_df) * 100
    baseline = predictive_df[predictive_df['Trend_Regime'] == regime]['Next_IsDirectional'].mean() * 100 if count > 0 else 0
    print(f"  {regime:25s}: {count:6,} samples ({pct:5.1f}%) | Baseline: {baseline:5.1f}% directional")

# ============================================================================
# STEP 6: Analyze Indicators by Trend Regime
# ============================================================================
print("\n" + "="*80)
print("STEP 6: ANALYZING INDICATORS BY TREND REGIME")
print("="*80)

# Top indicators to test
TOP_INDICATORS = [
    'WR2', 'WR4', 'WR7',
    'Large_Gap_Any', 'Gap_Up',
    'High_Volume', 'Vol_Expansion',
    'Extreme_GapStat', 'High_First_30Min_Vol',
    'Range_Pct', 'TR_Pct', 'Gap_Pct',
]

def analyze_indicator(df: pd.DataFrame, indicator: str, is_continuous: bool = False) -> Dict:
    """Analyze a single indicator's predictive power"""

    if len(df) < 30:
        return None

    baseline = df['Next_IsDirectional'].mean() * 100

    if is_continuous:
        # Quartile analysis
        valid_data = df[[indicator, 'Next_IsDirectional']].dropna()
        if len(valid_data) < 50:
            return None

        try:
            valid_data['Quartile'] = pd.qcut(valid_data[indicator], q=4, labels=False, duplicates='drop')
            max_q = valid_data['Quartile'].max()
            q4_data = valid_data[valid_data['Quartile'] == max_q]

            if len(q4_data) < 10:
                return None

            dir_rate = q4_data['Next_IsDirectional'].mean() * 100
            edge = dir_rate - baseline
            count = len(q4_data)
        except:
            return None
    else:
        # Binary indicator
        when_true = df[df[indicator] == 1]

        if len(when_true) < 10:
            return None

        dir_rate = when_true['Next_IsDirectional'].mean() * 100
        edge = dir_rate - baseline
        count = len(when_true)

    return {
        'Indicator': indicator,
        'Count': count,
        'Directional%': dir_rate,
        'Baseline%': baseline,
        'Edge': edge,
        'Abs_Edge': abs(edge)
    }

# Analyze each trend regime
all_results = []

for regime in ['Both_Above_200MA', 'Both_Below_200MA', 'Stock_Above_SPY_Below', 'Stock_Below_SPY_Above']:
    regime_data = predictive_df[predictive_df['Trend_Regime'] == regime].copy()

    if len(regime_data) < 100:
        print(f"\n{regime:25s}: SKIP (only {len(regime_data)} samples)")
        continue

    baseline = regime_data['Next_IsDirectional'].mean() * 100
    print(f"\n{regime:25s}: {len(regime_data):6,} samples | Baseline: {baseline:.1f}%")
    print("-" * 80)

    regime_results = []
    for indicator in TOP_INDICATORS:
        is_continuous = indicator in ['Range_Pct', 'TR_Pct', 'Gap_Pct']
        result = analyze_indicator(regime_data, indicator, is_continuous)

        if result:
            result['Trend_Regime'] = regime
            regime_results.append(result)

    # Sort and show top 5
    if regime_results:
        regime_df = pd.DataFrame(regime_results).sort_values('Abs_Edge', ascending=False)
        for _, row in regime_df.head(5).iterrows():
            print(f"  {row['Indicator']:20s}: {row['Edge']:+6.2f}% edge | {row['Directional%']:5.1f}% (n={row['Count']:5,})")
        all_results.extend(regime_results)

# Also analyze without any filter (all data)
print(f"\n{'NO FILTER (All Data)':25s}: {len(predictive_df):6,} samples")
print("-" * 80)
no_filter_results = []
for indicator in TOP_INDICATORS:
    is_continuous = indicator in ['Range_Pct', 'TR_Pct', 'Gap_Pct']
    result = analyze_indicator(predictive_df, indicator, is_continuous)

    if result:
        result['Trend_Regime'] = 'No_Filter'
        no_filter_results.append(result)

if no_filter_results:
    no_filter_df = pd.DataFrame(no_filter_results).sort_values('Abs_Edge', ascending=False)
    for _, row in no_filter_df.head(5).iterrows():
        print(f"  {row['Indicator']:20s}: {row['Edge']:+6.2f}% edge | {row['Directional%']:5.1f}% (n={row['Count']:5,})")
    all_results.extend(no_filter_results)

# ============================================================================
# STEP 7: Comparative Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 7: COMPARATIVE RANKINGS")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Abs_Edge', ascending=False)

print("\nTOP 20 STRONGEST EDGES (Any Trend Regime):")
print("-" * 80)
print(results_df[['Trend_Regime', 'Indicator', 'Edge', 'Directional%', 'Count']].head(20).to_string(index=False))

# Compare same indicator across different regimes
print("\n" + "="*80)
print("INDICATOR PERFORMANCE ACROSS TREND REGIMES")
print("="*80)

for indicator in ['WR4', 'Large_Gap_Any', 'High_Volume', 'Range_Pct']:
    print(f"\n{indicator}:")
    indicator_results = results_df[results_df['Indicator'] == indicator].sort_values('Edge', ascending=False)
    for _, row in indicator_results.iterrows():
        print(f"  {row['Trend_Regime']:25s}: {row['Edge']:+6.2f}% edge | {row['Directional%']:5.1f}% (n={row['Count']:5,})")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("="*80)

output_file = os.path.join(OUTPUT_DIR, f'technical_predictor_trend_filter_{group_label}.csv')
results_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Technical Indicators with Trend Filters - {group_label}', fontsize=16, fontweight='bold')

# 1. Edge comparison by trend regime
ax1 = axes[0, 0]
regime_comparison = results_df.groupby('Trend_Regime')['Abs_Edge'].mean().sort_values()
ax1.barh(range(len(regime_comparison)), regime_comparison.values, color='steelblue')
ax1.set_yticks(range(len(regime_comparison)))
ax1.set_yticklabels(regime_comparison.index)
ax1.set_xlabel('Average Absolute Edge (%)')
ax1.set_title('Average Edge by Trend Regime')
ax1.grid(axis='x', alpha=0.3)

# 2. Top indicators - Both Above 200MA
ax2 = axes[0, 1]
both_above = results_df[results_df['Trend_Regime'] == 'Both_Above_200MA'].sort_values('Edge', ascending=False).head(8)
if len(both_above) > 0:
    colors = ['green' if x > 0 else 'red' for x in both_above['Edge']]
    ax2.barh(range(len(both_above)), both_above['Edge'], color=colors)
    ax2.set_yticks(range(len(both_above)))
    ax2.set_yticklabels(both_above['Indicator'])
    ax2.set_xlabel('Edge (%)')
    ax2.set_title('Top Indicators: Both Above 200MA')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

# 3. Top indicators - Both Below 200MA
ax3 = axes[1, 0]
both_below = results_df[results_df['Trend_Regime'] == 'Both_Below_200MA'].sort_values('Edge', ascending=False).head(8)
if len(both_below) > 0:
    colors = ['green' if x > 0 else 'red' for x in both_below['Edge']]
    ax3.barh(range(len(both_below)), both_below['Edge'], color=colors)
    ax3.set_yticks(range(len(both_below)))
    ax3.set_yticklabels(both_below['Indicator'])
    ax3.set_xlabel('Edge (%)')
    ax3.set_title('Top Indicators: Both Below 200MA')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)

# 4. WR4 performance across regimes
ax4 = axes[1, 1]
wr4_by_regime = results_df[results_df['Indicator'] == 'WR4'].sort_values('Edge')
if len(wr4_by_regime) > 0:
    colors = ['green' if x > 0 else 'red' for x in wr4_by_regime['Edge']]
    ax4.barh(range(len(wr4_by_regime)), wr4_by_regime['Edge'], color=colors)
    ax4.set_yticks(range(len(wr4_by_regime)))
    ax4.set_yticklabels(wr4_by_regime['Trend_Regime'], fontsize=9)
    ax4.set_xlabel('Edge (%)')
    ax4.set_title('WR4 Edge by Trend Regime')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

viz_file = os.path.join(OUTPUT_DIR, f'technical_predictor_trend_filter_{group_label}.png')
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
print("\nKey Questions Answered:")
print("- Do momentum indicators work better when both stock & SPY are above 200MA?")
print("- Are there different indicators that work in downtrends vs uptrends?")
print("- How much does the trend filter improve the edge?")
print("="*80)
