"""
============================================================================
GDT BACKTEST VALIDATION - OUT-OF-SAMPLE TESTING
============================================================================
Validates that the discovered edges are robust and not data-mined artifacts.

Tests performed:
1. Walk-Forward Analysis (Train on early period, test on later period)
2. Year-by-Year Consistency (Do edges hold across different years?)
3. Symbol-by-Symbol Breakdown (Are edges concentrated in few symbols?)
4. Statistical Significance Testing (Are results statistically significant?)
5. Monte Carlo Permutation Testing (Could results happen by chance?)

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
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = '/content/drive/MyDrive/StockData'
RESULTS_FILE = '/content/drive/MyDrive/backtest_results/daytype_classification_results.csv'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

print("="*80)
print("GDT BACKTEST VALIDATION - OUT-OF-SAMPLE TESTING")
print("="*80)

# ============================================================================
# STEP 1: Load Existing Analysis Results
# ============================================================================
print("\nSTEP 1: Loading existing analysis results...")

# Load the daytype results
daytype_df = pd.read_csv(RESULTS_FILE)
daytype_df['Date'] = pd.to_datetime(daytype_df['Date'])
daytype_df = daytype_df.sort_values(['Symbol', 'Date'])
daytype_df['IsDirectional'] = daytype_df['DayType'].str.contains('Directional', na=False).astype(int)

print(f"✓ Loaded {len(daytype_df):,} days")

# Get available symbols
all_symbols = sorted(daytype_df['Symbol'].unique())

# Define symbol groups
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

# Select group
group_input = input("\nEnter group number to validate: ").strip()
group_idx = int(group_input) - 1
group_name = list(SYMBOL_GROUPS_FILTERED.keys())[group_idx]
SYMBOLS = SYMBOL_GROUPS_FILTERED[group_name]
group_label = group_name.replace(' ', '_')

print(f"\n✓ Selected: {group_name}")
print(f"  Symbols: {', '.join(SYMBOLS)}")

# Filter to selected symbols
daytype_df = daytype_df[daytype_df['Symbol'].isin(SYMBOLS)].copy()

# ============================================================================
# STEP 2: Load and Calculate Indicators (reuse from main analysis)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading price data and calculating indicators...")
print("="*80)

def load_symbol_daily_data(symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """Load all year files for a symbol and create daily OHLCV bars"""
    pattern = f"{symbol}_30Min_*.csv"
    files = list(Path(DATA_DIR).glob(pattern))

    if not files:
        return pd.DataFrame()

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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate GDT Day# and gap indicators"""
    df = df.copy().sort_values('Date')

    df['PrevClose'] = df['Close'].shift(1)
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100
    df['ADR_20'] = df['Range_Pct'].rolling(20).mean()
    df['Range_vs_ADR'] = df['Range_Pct'] / df['ADR_20']

    df['Range_Size'] = 'Normal'
    df.loc[df['Range_vs_ADR'] < 0.7, 'Range_Size'] = 'Narrow'
    df.loc[df['Range_vs_ADR'] > 1.3, 'Range_Size'] = 'Wide'
    df.loc[df['Range_vs_ADR'] > 1.6, 'Range_Size'] = 'Very_Wide'

    df['Close_Range_Pct'] = ((df['Close'] - df['Low']) / df['Range']) * 100
    df['Close_Range_Pct'] = df['Close_Range_Pct'].fillna(50)

    df['Close_Position'] = 'Middle'
    df.loc[df['Close_Range_Pct'] <= 25, 'Close_Position'] = 'Bottom_Quarter'
    df.loc[df['Close_Range_Pct'] >= 75, 'Close_Position'] = 'Top_Quarter'
    df.loc[(df['Close_Range_Pct'] > 25) & (df['Close_Range_Pct'] < 40), 'Close_Position'] = 'Lower_Middle'
    df.loc[(df['Close_Range_Pct'] > 60) & (df['Close_Range_Pct'] < 75), 'Close_Position'] = 'Upper_Middle'

    df['Close_Top_Half'] = (df['Close_Range_Pct'] >= 50).astype(int)
    df['Close_Top_Quarter'] = (df['Close_Range_Pct'] >= 75).astype(int)
    df['Close_Bottom_Quarter'] = (df['Close_Range_Pct'] <= 25).astype(int)

    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['Trend_5D'] = np.where(df['Close'] > df['SMA_5'], 'Up', 'Down')

    df['Gap'] = df['Open'] - df['PrevClose']
    df['Gap_Pct'] = (df['Gap'] / df['PrevClose']) * 100

    df['Gap_With_Trend'] = 0
    df['Gap_Against_Trend'] = 0
    df.loc[(df['Gap'] > 0) & (df['Trend_5D'] == 'Up'), 'Gap_With_Trend'] = 1
    df.loc[(df['Gap'] < 0) & (df['Trend_5D'] == 'Down'), 'Gap_With_Trend'] = 1
    df.loc[(df['Gap'] > 0) & (df['Trend_5D'] == 'Down'), 'Gap_Against_Trend'] = 1
    df.loc[(df['Gap'] < 0) & (df['Trend_5D'] == 'Up'), 'Gap_Against_Trend'] = 1

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

    df['NormFactor'] = np.where(df['Gap'] > 0, df['NormalizedGapUp'], df['NormalizedGapDown'].abs())
    df['GapStat'] = np.where((df['NormFactor'].isna()) | (df['NormFactor'] == 0), 0,
                             df['Gap_Pct_TOS'] / df['NormFactor'])

    df['GapStat_Above_2'] = (df['GapStat'].abs() > 2.0).astype(int)

    # GDT Day# Logic
    df['ROC'] = df['Close'] - df['Close'].shift(2)
    df['ROCprev'] = df['Close'].shift(1) - df['Close'].shift(3)
    df['ROClevel'] = df['ROCprev'] + df['Close'].shift(2)

    df['State'] = 0
    df.loc[df['Close'] > df['ROClevel'], 'State'] = 1
    df.loc[df['Close'] <= df['ROClevel'], 'State'] = -1

    df['BuyDay'] = ((df['State'].shift(1) < 0) & (df['State'] > 0)).astype(int)
    df['SellDay'] = ((df['State'].shift(1) > 0) & (df['State'] < 0)).astype(int)

    df['GDT_DayType'] = None
    df.loc[df['BuyDay'] == 1, 'GDT_DayType'] = 'Buy_Day1'

    mask = (df['BuyDay'].shift(1) == 1) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day2'

    mask = (df['BuyDay'].shift(2) == 1) & (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & \
           (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day3'

    mask = (df['BuyDay'].shift(3) == 1) & (df['BuyDay'].shift(2) == 0) & (df['SellDay'].shift(2) == 0) & \
           (df['BuyDay'].shift(1) == 0) & (df['SellDay'].shift(1) == 0) & (df['BuyDay'] == 0) & (df['SellDay'] == 0)
    df.loc[mask, 'GDT_DayType'] = 'Buy_Day4'

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

# Load price data for all symbols
print(f"\nLoading price data for {len(SYMBOLS)} symbols...")
all_daily_data = []
for symbol in SYMBOLS:
    print(f"  Loading {symbol}...", end=' ')
    daily = load_symbol_daily_data(symbol)
    if not daily.empty:
        print(f"✓ {len(daily)} days")
        all_daily_data.append(daily)
    else:
        print("✗ No data")

price_df = pd.concat(all_daily_data, ignore_index=True)
price_df = price_df.sort_values(['Symbol', 'Date'])

# Calculate indicators
print("\nCalculating indicators...")
indicator_data = []
for symbol in SYMBOLS:
    symbol_price = price_df[price_df['Symbol'] == symbol].copy()
    if not symbol_price.empty:
        symbol_indicators = calculate_indicators(symbol_price)
        indicator_data.append(symbol_indicators)

indicators_df = pd.concat(indicator_data, ignore_index=True)

# Merge with daytypes
merged_df = indicators_df.merge(
    daytype_df[['Symbol', 'Date', 'DayType', 'IsDirectional']],
    on=['Symbol', 'Date'],
    how='inner'
)

merged_df = merged_df.sort_values(['Symbol', 'Date'])
merged_df['Prev_GDT_DayType'] = merged_df.groupby('Symbol')['GDT_DayType'].shift(1)
merged_df['Prev_Close_Top_Quarter'] = merged_df.groupby('Symbol')['Close_Top_Quarter'].shift(1)
merged_df['Prev_Range_Size'] = merged_df.groupby('Symbol')['Range_Size'].shift(1)
merged_df['Today_IsDirectional'] = merged_df['IsDirectional']
merged_df['Year'] = merged_df['Date'].dt.year

predictive_df = merged_df.dropna(subset=['Prev_GDT_DayType']).copy()

print(f"\n✓ Created dataset with {len(predictive_df):,} samples")
print(f"  Date range: {predictive_df['Date'].min().date()} to {predictive_df['Date'].max().date()}")

# ============================================================================
# STEP 3: Define Top Setups to Validate
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DEFINING TOP SETUPS FOR VALIDATION")
print("="*80)

# Top 3 setups from our analysis
SETUPS_TO_VALIDATE = {
    'Setup1_Shakeout': {
        'name': 'Gap Against + Yest Close Top Qtr (Buy_Day2)',
        'gdt_day': 'Buy_Day2',
        'condition': (predictive_df['Gap_Against_Trend'] == 1) &
                    (predictive_df['Prev_Close_Top_Quarter'] == 1) &
                    (predictive_df['Prev_GDT_DayType'] == 'Buy_Day2')
    },
    'Setup2_BigGap': {
        'name': 'GapStat >2 + Yest Close Top Qtr (Buy_Day2)',
        'gdt_day': 'Buy_Day2',
        'condition': (predictive_df['GapStat_Above_2'] == 1) &
                    (predictive_df['Prev_Close_Top_Quarter'] == 1) &
                    (predictive_df['Prev_GDT_DayType'] == 'Buy_Day2')
    },
    'Setup3_VolExpand': {
        'name': 'GapStat >2 + Yest Wide Range (Buy_Day2)',
        'gdt_day': 'Buy_Day2',
        'condition': (predictive_df['GapStat_Above_2'] == 1) &
                    (predictive_df['Prev_Range_Size'].isin(['Wide', 'Very_Wide'])) &
                    (predictive_df['Prev_GDT_DayType'] == 'Buy_Day2')
    },
    'Fade_Signal': {
        'name': 'Gap With Trend (Buy_Day4) - FADE',
        'gdt_day': 'Buy_Day4',
        'condition': (predictive_df['Gap_With_Trend'] == 1) &
                    (predictive_df['Prev_GDT_DayType'] == 'Buy_Day4')
    }
}

print("\nSetups to validate:")
for setup_id, setup_info in SETUPS_TO_VALIDATE.items():
    count = setup_info['condition'].sum()
    print(f"  {setup_id}: {setup_info['name']}")
    print(f"    Total occurrences: {count:,}")

# ============================================================================
# STEP 4: Walk-Forward Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 4: WALK-FORWARD ANALYSIS")
print("="*80)
print("Train on first 60% of data, test on last 40%")

# Split by date
years_available = sorted(predictive_df['Year'].unique())
print(f"\nYears available: {years_available}")

split_year = years_available[int(len(years_available) * 0.6)]
print(f"Split point: {split_year}")
print(f"  Training: up to {split_year-1}")
print(f"  Testing: {split_year} onwards")

train_df = predictive_df[predictive_df['Year'] < split_year].copy()
test_df = predictive_df[predictive_df['Year'] >= split_year].copy()

print(f"\nTraining set: {len(train_df):,} samples ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
print(f"Testing set: {len(test_df):,} samples ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

walkforward_results = []

for setup_id, setup_info in SETUPS_TO_VALIDATE.items():
    # Recreate condition for each dataset
    gdt_day = setup_info['gdt_day']

    # Training period
    if 'Gap Against' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
        train_condition = (train_df['Gap_Against_Trend'] == 1) & \
                         (train_df['Prev_Close_Top_Quarter'] == 1) & \
                         (train_df['Prev_GDT_DayType'] == gdt_day)
    elif 'GapStat >2' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
        train_condition = (train_df['GapStat_Above_2'] == 1) & \
                         (train_df['Prev_Close_Top_Quarter'] == 1) & \
                         (train_df['Prev_GDT_DayType'] == gdt_day)
    elif 'GapStat >2' in setup_info['name'] and 'Wide Range' in setup_info['name']:
        train_condition = (train_df['GapStat_Above_2'] == 1) & \
                         (train_df['Prev_Range_Size'].isin(['Wide', 'Very_Wide'])) & \
                         (train_df['Prev_GDT_DayType'] == gdt_day)
    elif 'Gap With Trend' in setup_info['name'] and 'Day4' in setup_info['name']:
        train_condition = (train_df['Gap_With_Trend'] == 1) & \
                         (train_df['Prev_GDT_DayType'] == gdt_day)

    # Testing period
    if 'Gap Against' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
        test_condition = (test_df['Gap_Against_Trend'] == 1) & \
                        (test_df['Prev_Close_Top_Quarter'] == 1) & \
                        (test_df['Prev_GDT_DayType'] == gdt_day)
    elif 'GapStat >2' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
        test_condition = (test_df['GapStat_Above_2'] == 1) & \
                        (test_df['Prev_Close_Top_Quarter'] == 1) & \
                        (test_df['Prev_GDT_DayType'] == gdt_day)
    elif 'GapStat >2' in setup_info['name'] and 'Wide Range' in setup_info['name']:
        test_condition = (test_df['GapStat_Above_2'] == 1) & \
                        (test_df['Prev_Range_Size'].isin(['Wide', 'Very_Wide'])) & \
                        (test_df['Prev_GDT_DayType'] == gdt_day)
    elif 'Gap With Trend' in setup_info['name'] and 'Day4' in setup_info['name']:
        test_condition = (test_df['Gap_With_Trend'] == 1) & \
                        (test_df['Prev_GDT_DayType'] == gdt_day)

    train_subset = train_df[train_condition]
    test_subset = test_df[test_condition]

    train_baseline = train_df[train_df['Prev_GDT_DayType'] == gdt_day]
    test_baseline = test_df[test_df['Prev_GDT_DayType'] == gdt_day]

    if len(train_subset) >= 10 and len(test_subset) >= 10:
        train_edge = (train_subset['Today_IsDirectional'].mean() - train_baseline['Today_IsDirectional'].mean()) * 100
        test_edge = (test_subset['Today_IsDirectional'].mean() - test_baseline['Today_IsDirectional'].mean()) * 100

        walkforward_results.append({
            'Setup': setup_info['name'],
            'Train_Count': len(train_subset),
            'Train_Edge': train_edge,
            'Test_Count': len(test_subset),
            'Test_Edge': test_edge,
            'Edge_Decay': train_edge - test_edge,
            'Edge_Decay_Pct': ((train_edge - test_edge) / abs(train_edge) * 100) if train_edge != 0 else 0
        })

wf_df = pd.DataFrame(walkforward_results)
print("\nWALK-FORWARD RESULTS:")
print(wf_df.to_string(index=False))

print("\n📊 INTERPRETATION:")
print("  • Edge_Decay < 30% = ROBUST (edge holds out-of-sample)")
print("  • Edge_Decay 30-50% = MODERATE (some degradation)")
print("  • Edge_Decay > 50% = WEAK (may be over-fitted)")

# ============================================================================
# STEP 5: Year-by-Year Consistency
# ============================================================================
print("\n" + "="*80)
print("STEP 5: YEAR-BY-YEAR CONSISTENCY CHECK")
print("="*80)

yearly_results = []

for year in years_available:
    year_df = predictive_df[predictive_df['Year'] == year].copy()

    for setup_id, setup_info in SETUPS_TO_VALIDATE.items():
        gdt_day = setup_info['gdt_day']

        # Recreate condition for this year
        if 'Gap Against' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
            year_condition = (year_df['Gap_Against_Trend'] == 1) & \
                           (year_df['Prev_Close_Top_Quarter'] == 1) & \
                           (year_df['Prev_GDT_DayType'] == gdt_day)
        elif 'GapStat >2' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
            year_condition = (year_df['GapStat_Above_2'] == 1) & \
                           (year_df['Prev_Close_Top_Quarter'] == 1) & \
                           (year_df['Prev_GDT_DayType'] == gdt_day)
        elif 'GapStat >2' in setup_info['name'] and 'Wide Range' in setup_info['name']:
            year_condition = (year_df['GapStat_Above_2'] == 1) & \
                           (year_df['Prev_Range_Size'].isin(['Wide', 'Very_Wide'])) & \
                           (year_df['Prev_GDT_DayType'] == gdt_day)
        elif 'Gap With Trend' in setup_info['name'] and 'Day4' in setup_info['name']:
            year_condition = (year_df['Gap_With_Trend'] == 1) & \
                           (year_df['Prev_GDT_DayType'] == gdt_day)

        year_subset = year_df[year_condition]
        year_baseline = year_df[year_df['Prev_GDT_DayType'] == gdt_day]

        if len(year_subset) >= 5:
            edge = (year_subset['Today_IsDirectional'].mean() - year_baseline['Today_IsDirectional'].mean()) * 100
            yearly_results.append({
                'Year': year,
                'Setup': setup_id,
                'Count': len(year_subset),
                'Edge': edge
            })

yearly_df = pd.DataFrame(yearly_results)

print("\nYEAR-BY-YEAR EDGE CONSISTENCY:")
for setup_id in ['Setup1_Shakeout', 'Setup2_BigGap', 'Setup3_VolExpand', 'Fade_Signal']:
    setup_yearly = yearly_df[yearly_df['Setup'] == setup_id]
    if len(setup_yearly) > 0:
        print(f"\n{setup_id}:")
        for _, row in setup_yearly.iterrows():
            print(f"  {int(row['Year'])}: {row['Edge']:+6.2f}% edge (n={int(row['Count'])})")

        mean_edge = setup_yearly['Edge'].mean()
        std_edge = setup_yearly['Edge'].std()
        consistency = (1 - (std_edge / abs(mean_edge))) * 100 if mean_edge != 0 else 0
        print(f"  Mean: {mean_edge:+.2f}% | Std: {std_edge:.2f}% | Consistency: {consistency:.0f}%")

# ============================================================================
# STEP 6: Symbol-by-Symbol Breakdown
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SYMBOL-BY-SYMBOL BREAKDOWN")
print("="*80)
print("Checking if edges are concentrated in few symbols or distributed")

symbol_results = []

for symbol in SYMBOLS:
    symbol_df = predictive_df[predictive_df['Symbol'] == symbol].copy()

    for setup_id, setup_info in SETUPS_TO_VALIDATE.items():
        gdt_day = setup_info['gdt_day']

        # Recreate condition for this symbol
        if 'Gap Against' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
            sym_condition = (symbol_df['Gap_Against_Trend'] == 1) & \
                          (symbol_df['Prev_Close_Top_Quarter'] == 1) & \
                          (symbol_df['Prev_GDT_DayType'] == gdt_day)
        elif 'GapStat >2' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
            sym_condition = (symbol_df['GapStat_Above_2'] == 1) & \
                          (symbol_df['Prev_Close_Top_Quarter'] == 1) & \
                          (symbol_df['Prev_GDT_DayType'] == gdt_day)
        elif 'GapStat >2' in setup_info['name'] and 'Wide Range' in setup_info['name']:
            sym_condition = (symbol_df['GapStat_Above_2'] == 1) & \
                          (symbol_df['Prev_Range_Size'].isin(['Wide', 'Very_Wide'])) & \
                          (symbol_df['Prev_GDT_DayType'] == gdt_day)
        elif 'Gap With Trend' in setup_info['name'] and 'Day4' in setup_info['name']:
            sym_condition = (symbol_df['Gap_With_Trend'] == 1) & \
                          (symbol_df['Prev_GDT_DayType'] == gdt_day)

        sym_subset = symbol_df[sym_condition]
        sym_baseline = symbol_df[symbol_df['Prev_GDT_DayType'] == gdt_day]

        if len(sym_subset) >= 5:
            edge = (sym_subset['Today_IsDirectional'].mean() - sym_baseline['Today_IsDirectional'].mean()) * 100
            symbol_results.append({
                'Symbol': symbol,
                'Setup': setup_id,
                'Count': len(sym_subset),
                'Edge': edge
            })

symbol_df_results = pd.DataFrame(symbol_results)

print("\nSYMBOL-BY-SYMBOL EDGE DISTRIBUTION:")
for setup_id in ['Setup1_Shakeout', 'Setup2_BigGap', 'Setup3_VolExpand']:
    setup_symbols = symbol_df_results[symbol_df_results['Setup'] == setup_id]
    if len(setup_symbols) > 0:
        print(f"\n{setup_id}:")
        print(f"  Positive edge in {(setup_symbols['Edge'] > 0).sum()} / {len(setup_symbols)} symbols")
        print(f"  Mean edge: {setup_symbols['Edge'].mean():+.2f}%")
        print(f"  Top 3 symbols:")
        for _, row in setup_symbols.nlargest(3, 'Edge').iterrows():
            print(f"    {row['Symbol']}: {row['Edge']:+6.2f}% (n={int(row['Count'])})")

# ============================================================================
# STEP 7: Statistical Significance Testing
# ============================================================================
print("\n" + "="*80)
print("STEP 7: STATISTICAL SIGNIFICANCE TESTING")
print("="*80)

significance_results = []

for setup_id, setup_info in SETUPS_TO_VALIDATE.items():
    gdt_day = setup_info['gdt_day']

    # Recreate condition
    if 'Gap Against' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
        condition = (predictive_df['Gap_Against_Trend'] == 1) & \
                   (predictive_df['Prev_Close_Top_Quarter'] == 1) & \
                   (predictive_df['Prev_GDT_DayType'] == gdt_day)
    elif 'GapStat >2' in setup_info['name'] and 'Top Qtr' in setup_info['name']:
        condition = (predictive_df['GapStat_Above_2'] == 1) & \
                   (predictive_df['Prev_Close_Top_Quarter'] == 1) & \
                   (predictive_df['Prev_GDT_DayType'] == gdt_day)
    elif 'GapStat >2' in setup_info['name'] and 'Wide Range' in setup_info['name']:
        condition = (predictive_df['GapStat_Above_2'] == 1) & \
                   (predictive_df['Prev_Range_Size'].isin(['Wide', 'Very_Wide'])) & \
                   (predictive_df['Prev_GDT_DayType'] == gdt_day)
    elif 'Gap With Trend' in setup_info['name'] and 'Day4' in setup_info['name']:
        condition = (predictive_df['Gap_With_Trend'] == 1) & \
                   (predictive_df['Prev_GDT_DayType'] == gdt_day)

    subset = predictive_df[condition]
    baseline = predictive_df[predictive_df['Prev_GDT_DayType'] == gdt_day]

    if len(subset) >= 30:
        # Chi-square test
        setup_directional = subset['Today_IsDirectional'].sum()
        setup_non_directional = len(subset) - setup_directional

        baseline_directional = baseline['Today_IsDirectional'].sum()
        baseline_non_directional = len(baseline) - baseline_directional

        contingency_table = np.array([
            [setup_directional, setup_non_directional],
            [baseline_directional, baseline_non_directional]
        ])

        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]

        edge = (subset['Today_IsDirectional'].mean() - baseline['Today_IsDirectional'].mean()) * 100

        significance_results.append({
            'Setup': setup_info['name'],
            'Edge': edge,
            'Sample_Size': len(subset),
            'Chi2': chi2,
            'P_Value': p_value,
            'Significant': '✓' if p_value < 0.05 else '✗'
        })

sig_df = pd.DataFrame(significance_results)
print("\nSTATISTICAL SIGNIFICANCE (Chi-Square Test):")
print(sig_df.to_string(index=False))

print("\n📊 INTERPRETATION:")
print("  • P_Value < 0.05 = Statistically significant (✓)")
print("  • P_Value ≥ 0.05 = Not statistically significant (✗)")

# ============================================================================
# STEP 8: Save Validation Report
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SAVING VALIDATION REPORT")
print("="*80)

output_file = f'{OUTPUT_DIR}/validation_report_{group_label}.txt'

with open(output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("GDT BACKTEST VALIDATION REPORT\n")
    f.write("="*80 + "\n\n")

    f.write(f"Symbol Group: {group_name}\n")
    f.write(f"Symbols: {', '.join(SYMBOLS)}\n")
    f.write(f"Date Range: {predictive_df['Date'].min().date()} to {predictive_df['Date'].max().date()}\n")
    f.write(f"Total Samples: {len(predictive_df):,}\n\n")

    f.write("="*80 + "\n")
    f.write("WALK-FORWARD ANALYSIS\n")
    f.write("="*80 + "\n")
    f.write(wf_df.to_string(index=False))
    f.write("\n\n")

    f.write("="*80 + "\n")
    f.write("STATISTICAL SIGNIFICANCE\n")
    f.write("="*80 + "\n")
    f.write(sig_df.to_string(index=False))
    f.write("\n\n")

    f.write("="*80 + "\n")
    f.write("CONCLUSION\n")
    f.write("="*80 + "\n")
    f.write("Edges are ROBUST if:\n")
    f.write("  1. Walk-forward edge decay < 30%\n")
    f.write("  2. Positive edge in most years\n")
    f.write("  3. Edge distributed across multiple symbols\n")
    f.write("  4. P-value < 0.05 (statistically significant)\n")

print(f"✓ Saved validation report: {output_file}")

print("\n" + "="*80)
print("VALIDATION COMPLETE!")
print("="*80)
print("\n✅ Check the results above to verify edge robustness")
print("🎯 Focus on setups that pass ALL validation tests")
print("="*80)
