"""
============================================================================
TECHNICAL PREDICTOR - BY SYMBOL GROUP & MARKET REGIME
============================================================================
Enhanced version that tests technical indicators separately for:
- Symbol groups (high-beta tech, indices, sectors, etc.)
- Market regimes (time periods, volatility environments)

This helps identify where technical edges actually exist vs getting washed
out in aggregate data.

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
print("TECHNICAL PREDICTOR - BY SYMBOL GROUP & MARKET REGIME")
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

# Extract year for regime analysis
daytype_df['Year'] = daytype_df['Date'].dt.year

# ============================================================================
# DEFINE SYMBOL GROUPS
# ============================================================================
print("\n" + "="*80)
print("DEFINING SYMBOL GROUPS")
print("="*80)

all_symbols = sorted(daytype_df['Symbol'].unique())

SYMBOL_GROUPS = {
    'Major Indices': ['SPY', 'QQQ', 'IWM', 'DIA'],

    'High Beta Tech': ['NVDA', 'TSLA', 'AMD', 'PLTR', 'SNOW', 'NET', 'DDOG',
                       'SMCI', 'MSTR', 'RIVN', 'ABNB', 'UBER'],

    'Mega Cap Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX', 'TSLA', 'NVDA'],

    'Meme/High Vol': ['GME', 'MSTR', 'SMCI', 'HOOD', 'SOFI', 'PLTR', 'CVNA', 'RIVN'],

    'Energy Sector': ['XLE', 'CVX', 'XOM', 'COP', 'SLB', 'HAL', 'OXY', 'MPC', 'VLO'],

    'Financial Sector': ['XLF', 'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'SCHW'],

    'Tech Sector': ['XLK', 'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CSCO', 'ADBE'],

    'Sector ETFs': ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE'],

    'Semiconductor': ['SMH', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'MU', 'MRVL', 'KLAC', 'LRCX'],

    'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'PM', 'MO'],

    'Volatility ETFs': ['SOXL', 'TQQQ', 'TSLL'],
}

# Filter to only symbols that exist in our data
SYMBOL_GROUPS_FILTERED = {}
for group_name, symbols in SYMBOL_GROUPS.items():
    available = [s for s in symbols if s in all_symbols]
    if available:
        SYMBOL_GROUPS_FILTERED[group_name] = available

print("\nSymbol Groups Available:")
for group_name, symbols in SYMBOL_GROUPS_FILTERED.items():
    print(f"  {group_name:20s}: {len(symbols):2d} symbols - {', '.join(symbols[:5])}{' ...' if len(symbols) > 5 else ''}")

# ============================================================================
# DEFINE MARKET REGIMES (Time Periods)
# ============================================================================
print("\n" + "="*80)
print("DEFINING MARKET REGIMES (Time Periods)")
print("="*80)

MARKET_REGIMES = {
    '2020-2021 (Bull)': (2020, 2021),
    '2022 (Bear)': (2022, 2022),
    '2023-2024 (Recovery)': (2023, 2024),
    '2024 Only': (2024, 2024),
    '2023 Only': (2023, 2023),
}

print("\nMarket Regimes:")
for regime_name, (start_year, end_year) in MARKET_REGIMES.items():
    regime_data = daytype_df[(daytype_df['Year'] >= start_year) & (daytype_df['Year'] <= end_year)]
    print(f"  {regime_name:25s}: {len(regime_data):6,} days")

# ============================================================================
# STEP 2: Load Raw Price Data and Calculate Daily OHLCV
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading raw price data and calculating daily OHLCV...")
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

# User input for year range
print("\n" + "-"*80)
print("YEAR RANGE SELECTION")
print("-"*80)
start_year_input = input("Start YEAR for price data (e.g., 2020, or press Enter for all): ").strip()
START_YEAR = None if start_year_input == '' else int(start_year_input)
end_year_input = input("End YEAR for price data (e.g., 2025, or press Enter for all): ").strip()
END_YEAR = None if end_year_input == '' else int(end_year_input)

# Collect all unique symbols from all groups
all_group_symbols = set()
for symbols in SYMBOL_GROUPS_FILTERED.values():
    all_group_symbols.update(symbols)
all_group_symbols = sorted(list(all_group_symbols))

print(f"\nLoading price data for {len(all_group_symbols)} unique symbols across all groups...")

all_daily_data = []
for i, symbol in enumerate(all_group_symbols, 1):
    print(f"  [{i:3d}/{len(all_group_symbols):3d}] {symbol:6s}...", end=' ')
    daily = load_symbol_daily_data(symbol, START_YEAR, END_YEAR)
    if not daily.empty:
        print(f"✓ {len(daily):4d} days")
        all_daily_data.append(daily)
    else:
        print("✗ No data")

if not all_daily_data:
    raise ValueError("No price data loaded! Check your data directory and year range.")

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

    # True Range
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'PrevClose']].max(axis=1) - df[['Low', 'PrevClose']].min(axis=1)
    df['TR_Pct'] = (df['TR'] / df['Close']) * 100

    # Narrow Range indicators
    df['NR2'] = (df['Range'] == df['Range'].rolling(2).min()).astype(int)
    df['NR4'] = (df['Range'] == df['Range'].rolling(4).min()).astype(int)
    df['NR7'] = (df['Range'] == df['Range'].rolling(7).min()).astype(int)

    # Wide Range indicators (MOMENTUM)
    df['WR2'] = (df['Range'] == df['Range'].rolling(2).max()).astype(int)
    df['WR4'] = (df['Range'] == df['Range'].rolling(4).max()).astype(int)
    df['WR7'] = (df['Range'] == df['Range'].rolling(7).max()).astype(int)

    # ADR
    df['ADR_20'] = df['Range_Pct'].rolling(20).mean()
    df['ADR_StdDev'] = df['Range_Pct'].rolling(20).std()
    df['ADR_ZScore'] = (df['Range_Pct'] - df['ADR_20']) / df['ADR_StdDev']

    # Volatility
    df['Vol_Contraction'] = (df['Range_Pct'] < (df['ADR_20'] * 0.7)).astype(int)
    df['Vol_Expansion'] = (df['Range_Pct'] > (df['ADR_20'] * 1.3)).astype(int)

    # Volume indicators
    df['Avg_Volume_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume_20']
    df['High_Volume'] = (df['Volume_Ratio'] > 1.5).astype(int)

    # Gap indicators
    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100
    df['Large_Gap_Any'] = ((df['Gap_Pct'].abs()) > 1.5).astype(int)
    df['Gap_Up'] = (df['Gap_Pct'] > 0.5).astype(int)

    # GapStat (TOS-style normalized gap)
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
for symbol in all_group_symbols:
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
    daytype_df[['Symbol', 'Date', 'DayType', 'IsDirectional', 'Year']],
    on=['Symbol', 'Date'],
    how='inner'
)

# Create next-day target
merged_df = merged_df.sort_values(['Symbol', 'Date'])
merged_df['Next_IsDirectional'] = merged_df.groupby('Symbol')['IsDirectional'].shift(-1)
predictive_df = merged_df.dropna(subset=['Next_IsDirectional']).copy()

print(f"✓ Created predictive dataset with {len(predictive_df):,} samples")

# ============================================================================
# STEP 5: Analyze by Symbol Group & Market Regime
# ============================================================================
print("\n" + "="*80)
print("STEP 5: ANALYZING BY SYMBOL GROUP & MARKET REGIME")
print("="*80)

# Top indicators to test (based on previous analysis)
TOP_INDICATORS = [
    'Large_Gap_Any',
    'Range_Pct',
    'TR_Pct',
    'Gap_Pct',
    'WR2',
    'WR4',
    'High_Volume',
    'Vol_Expansion',
    'Extreme_GapStat',
    'High_First_30Min_Vol',
]

def analyze_indicator_by_group(df: pd.DataFrame, indicator: str, is_continuous: bool = False) -> Dict:
    """Analyze a single indicator's predictive power"""

    if len(df) < 30:
        return None

    baseline = df['Next_IsDirectional'].mean() * 100

    if is_continuous:
        # Use quartile analysis for continuous
        valid_data = df[[indicator, 'Next_IsDirectional']].dropna()
        if len(valid_data) < 50:
            return None

        try:
            valid_data['Quartile'] = pd.qcut(valid_data[indicator], q=4, labels=False, duplicates='drop')
            min_q = valid_data['Quartile'].min()
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
        'Abs_Edge': abs(edge),
        'Total_Samples': len(df)
    }

# Analyze each group
all_results = []

print("\nAnalyzing Symbol Groups...")
print("-" * 80)

for group_name, group_symbols in SYMBOL_GROUPS_FILTERED.items():
    group_data = predictive_df[predictive_df['Symbol'].isin(group_symbols)].copy()

    if len(group_data) < 100:
        print(f"  {group_name:25s}: SKIP (only {len(group_data)} samples)")
        continue

    baseline = group_data['Next_IsDirectional'].mean() * 100
    print(f"\n  {group_name:25s}: {len(group_data):6,} samples | Baseline: {baseline:.1f}% directional")

    # Test each indicator
    group_results = []
    for indicator in TOP_INDICATORS:
        is_continuous = indicator in ['Range_Pct', 'TR_Pct', 'Gap_Pct']
        result = analyze_indicator_by_group(group_data, indicator, is_continuous)

        if result:
            result['Group'] = group_name
            result['Regime'] = 'All Years'
            group_results.append(result)

    # Show top 3 for this group
    if group_results:
        group_df = pd.DataFrame(group_results).sort_values('Abs_Edge', ascending=False)
        for _, row in group_df.head(3).iterrows():
            print(f"    {row['Indicator']:20s}: {row['Edge']:+5.2f}% edge ({row['Directional%']:.1f}% vs {row['Baseline%']:.1f}% baseline)")
        all_results.extend(group_results)

# Analyze each market regime (using all symbols)
print("\n" + "="*80)
print("Analyzing Market Regimes...")
print("-" * 80)

for regime_name, (start_year, end_year) in MARKET_REGIMES.items():
    regime_data = predictive_df[(predictive_df['Year'] >= start_year) & (predictive_df['Year'] <= end_year)].copy()

    if len(regime_data) < 100:
        print(f"  {regime_name:25s}: SKIP (only {len(regime_data)} samples)")
        continue

    baseline = regime_data['Next_IsDirectional'].mean() * 100
    print(f"\n  {regime_name:25s}: {len(regime_data):6,} samples | Baseline: {baseline:.1f}% directional")

    regime_results = []
    for indicator in TOP_INDICATORS:
        is_continuous = indicator in ['Range_Pct', 'TR_Pct', 'Gap_Pct']
        result = analyze_indicator_by_group(regime_data, indicator, is_continuous)

        if result:
            result['Group'] = 'All Symbols'
            result['Regime'] = regime_name
            regime_results.append(result)

    # Show top 3 for this regime
    if regime_results:
        regime_df = pd.DataFrame(regime_results).sort_values('Abs_Edge', ascending=False)
        for _, row in regime_df.head(3).iterrows():
            print(f"    {row['Indicator']:20s}: {row['Edge']:+5.2f}% edge ({row['Directional%']:.1f}% vs {row['Baseline%']:.1f}% baseline)")
        all_results.extend(regime_results)

# ============================================================================
# STEP 6: Comparative Analysis & Rankings
# ============================================================================
print("\n" + "="*80)
print("STEP 6: COMPARATIVE RANKINGS")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Abs_Edge', ascending=False)

print("\nTOP 20 STRONGEST EDGES (Any Group or Regime):")
print("-" * 80)
print(results_df[['Group', 'Regime', 'Indicator', 'Edge', 'Directional%', 'Count']].head(20).to_string(index=False))

print("\n" + "="*80)
print("BEST INDICATORS BY SYMBOL GROUP:")
print("="*80)

for group_name in SYMBOL_GROUPS_FILTERED.keys():
    group_results = results_df[(results_df['Group'] == group_name) & (results_df['Regime'] == 'All Years')]
    if len(group_results) > 0:
        best = group_results.iloc[0]
        print(f"{group_name:25s}: {best['Indicator']:20s} | Edge: {best['Edge']:+5.2f}% | Dir: {best['Directional%']:.1f}%")

print("\n" + "="*80)
print("BEST INDICATORS BY MARKET REGIME:")
print("="*80)

for regime_name in MARKET_REGIMES.keys():
    regime_results = results_df[(results_df['Regime'] == regime_name) & (results_df['Group'] == 'All Symbols')]
    if len(regime_results) > 0:
        best = regime_results.iloc[0]
        print(f"{regime_name:25s}: {best['Indicator']:20s} | Edge: {best['Edge']:+5.2f}% | Dir: {best['Directional%']:.1f}%")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

output_file = os.path.join(OUTPUT_DIR, 'technical_predictor_by_group_and_regime.csv')
results_df.to_csv(output_file, index=False)
print(f"✓ Saved detailed results to: {output_file}")

# Create summary visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Technical Indicator Edges by Symbol Group & Market Regime', fontsize=16, fontweight='bold')

# 1. Best edge by symbol group
ax1 = axes[0, 0]
group_best = results_df[results_df['Regime'] == 'All Years'].groupby('Group')['Abs_Edge'].max().sort_values()
if len(group_best) > 0:
    ax1.barh(range(len(group_best)), group_best.values, color='steelblue')
    ax1.set_yticks(range(len(group_best)))
    ax1.set_yticklabels(group_best.index, fontsize=9)
    ax1.set_xlabel('Best Absolute Edge (%)')
    ax1.set_title('Strongest Edge by Symbol Group')
    ax1.grid(axis='x', alpha=0.3)

# 2. Best edge by market regime
ax2 = axes[0, 1]
regime_best = results_df[results_df['Group'] == 'All Symbols'].groupby('Regime')['Abs_Edge'].max().sort_values()
if len(regime_best) > 0:
    ax2.barh(range(len(regime_best)), regime_best.values, color='coral')
    ax2.set_yticks(range(len(regime_best)))
    ax2.set_yticklabels(regime_best.index, fontsize=9)
    ax2.set_xlabel('Best Absolute Edge (%)')
    ax2.set_title('Strongest Edge by Market Regime')
    ax2.grid(axis='x', alpha=0.3)

# 3. Top 15 edges overall
ax3 = axes[1, 0]
top_15 = results_df.head(15).sort_values('Edge')
colors = ['green' if x > 0 else 'red' for x in top_15['Edge']]
ax3.barh(range(len(top_15)), top_15['Edge'], color=colors)
ax3.set_yticks(range(len(top_15)))
ax3.set_yticklabels([f"{row['Group'][:12]}: {row['Indicator']}" for _, row in top_15.iterrows()], fontsize=8)
ax3.set_xlabel('Edge (%)')
ax3.set_title('Top 15 Edges (Group + Indicator)')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.grid(axis='x', alpha=0.3)

# 4. Baseline directional rate by group
ax4 = axes[1, 1]
baseline_by_group = results_df[results_df['Regime'] == 'All Years'].groupby('Group')['Baseline%'].first().sort_values()
if len(baseline_by_group) > 0:
    ax4.barh(range(len(baseline_by_group)), baseline_by_group.values, color='orange')
    ax4.set_yticks(range(len(baseline_by_group)))
    ax4.set_yticklabels(baseline_by_group.index, fontsize=9)
    ax4.set_xlabel('Baseline Directional Rate (%)')
    ax4.set_title('Base Directional % by Symbol Group')
    ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

viz_file = os.path.join(OUTPUT_DIR, 'technical_predictor_by_group_charts.png')
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
print("- Check which symbol groups show the strongest edges")
print("- Compare market regimes to see if edges are regime-dependent")
print("- Look for indicators that work consistently across groups/regimes")
print("="*80)
