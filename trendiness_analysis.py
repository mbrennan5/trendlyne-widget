"""
============================================================================
TRENDINESS ANALYSIS - Test Symbol Directional Behavior Over Time
============================================================================

This script analyzes whether certain stocks are consistently more directional
over time, or if "trendiness" changes across different time periods.

Metrics tracked:
- DNP Ratio: % of clean trending days (no pullbacks)
- Directional Ratio: % of days that moved away from opening (DNP + DWP)
- Range Ratio: % of days that stayed in range
- Trend Stability: Consistency of trending behavior across time periods

Copy this entire code block into Google Colab and run it.
============================================================================
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# Import libraries
import pandas as pd
import numpy as np
import pytz
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier Class (same as before)
class DayTypeClassifier30Min:
    RTH_START = time(9, 30)
    RTH_END = time(16, 15)
    CLASSIFICATION = {0: "N/A", 1: "RANGE DAY", 2: "DWP (Directional w/ Pullbacks)", 3: "DNP (Directional No Pullbacks)"}

    def __init__(self, data_directory: str, timezone: str = 'America/New_York'):
        self.data_directory = Path(data_directory)
        self.timezone = pytz.timezone(timezone)
        self.results = {}

    def find_30min_files(self) -> Dict[str, Path]:
        files = {}
        for file_path in self.data_directory.glob('*30Min*.csv'):
            filename = file_path.stem
            symbol = filename.split('_')[0].split('-')[0].upper()
            files[symbol] = file_path
        return files

    def load_30min_data(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        # Diagnostic: Show how many rows we loaded
        print(f"Loaded {len(df)} rows from CSV...", end=' ')

        if 't' in df.columns:
            df['datetime'] = pd.to_datetime(df['t'], utc=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df.set_index('datetime', inplace=True)
        try:
            df.index = df.index.tz_convert(self.timezone)
        except Exception as e:
            if not hasattr(df.index, 'tz') or df.index.tz is None:
                df.index = pd.DatetimeIndex(df.index).tz_localize('UTC').tz_convert(self.timezone)

        # Diagnostic: Show date range in file
        if len(df) > 0:
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}", end=' ')

        return df

    def aggregate_to_hourly(self, df_30min: pd.DataFrame) -> pd.DataFrame:
        df_rth = df_30min.between_time(self.RTH_START, self.RTH_END).copy()
        if df_rth.empty:
            return pd.DataFrame()
        df_rth['date'] = df_rth.index.date
        df_rth['time'] = df_rth.index.time

        def assign_hourly_session(t):
            if time(9, 30) <= t < time(10, 30): return 0
            elif time(10, 30) <= t < time(11, 30): return 1
            elif time(11, 30) <= t < time(12, 30): return 2
            elif time(12, 30) <= t < time(13, 30): return 3
            elif time(13, 30) <= t < time(14, 30): return 4
            elif time(14, 30) <= t < time(15, 30): return 5
            elif time(15, 30) <= t <= time(16, 15): return 6
            else: return -1

        df_rth['hourly_session'] = df_rth['time'].apply(assign_hourly_session)
        df_rth = df_rth[df_rth['hourly_session'] >= 0].copy()

        hourly_bars = []
        for date in df_rth['date'].unique():
            day_data = df_rth[df_rth['date'] == date]
            for session_num in range(7):
                session_data = day_data[day_data['hourly_session'] == session_num]
                if session_data.empty:
                    continue
                hourly_bar = {
                    'datetime': session_data.index[0],
                    'date': date,
                    'session': session_num,
                    'Open': session_data.iloc[0]['Open'],
                    'High': session_data['High'].max(),
                    'Low': session_data['Low'].min(),
                    'Close': session_data.iloc[-1]['Close'],
                    'Volume': session_data['Volume'].sum()
                }
                hourly_bars.append(hourly_bar)

        df_hourly = pd.DataFrame(hourly_bars)
        if not df_hourly.empty:
            df_hourly.set_index('datetime', inplace=True)
        return df_hourly

    def classify_day(self, day_data: pd.DataFrame) -> Tuple[int, Dict]:
        if len(day_data) < 2:
            return 0, {"reason": "Insufficient data"}
        day_data = day_data.sort_values('session')
        opening_price = day_data.iloc[0]['Open']
        H930 = L930 = opening_price
        dir_up = False
        dir_down = False
        has_pullback = False
        test_count = 0
        reversed_to_opening = False
        prev_high = None
        prev_low = None

        for i, (idx, row) in enumerate(day_data.iterrows()):
            hour_high = row['High']
            hour_low = row['Low']
            session_num = row['session']

            if i > 0:
                if hour_high > H930:
                    dir_up = True
                if hour_low < L930:
                    dir_down = True

            if hour_low <= H930 and hour_high >= L930:
                test_count += 1

            is_last_bar = (i == len(day_data) - 1)
            if i > 0 and prev_high is not None and prev_low is not None and not is_last_bar:
                if dir_up and hour_low < prev_low:
                    has_pullback = True
                elif dir_down and hour_high > prev_high:
                    has_pullback = True

            if session_num >= 2:
                if hour_high >= L930 and hour_low <= H930:
                    reversed_to_opening = True

            prev_high = hour_high
            prev_low = hour_low

        is_directional = dir_up or dir_down
        range1 = test_count >= 4
        range2 = is_directional and reversed_to_opening
        is_range_day = range1 or range2

        if is_range_day:
            classification = 1
        elif is_directional and has_pullback:
            classification = 2
        elif is_directional and not has_pullback:
            classification = 3
        else:
            classification = 0

        debug_info = {
            "opening_price": opening_price,
            "dir_up": dir_up,
            "dir_down": dir_down,
            "has_pullback": has_pullback,
            "test_count": test_count,
            "reversed_to_opening": reversed_to_opening,
            "is_directional": is_directional,
            "range1": range1,
            "range2": range2,
            "num_sessions": len(day_data)
        }

        return classification, debug_info

    def analyze_symbol(self, symbol: str, file_path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        print(f"\nAnalyzing {symbol}...", end=' ')
        df_30min = self.load_30min_data(file_path)
        if df_30min.empty:
            print(f"No data")
            return pd.DataFrame()

        # Apply date filtering if specified
        if start_date:
            before_filter = len(df_30min)
            df_30min = df_30min[df_30min.index >= start_date]
            print(f"\nAfter start_date filter: {len(df_30min)} rows (filtered out {before_filter - len(df_30min)})", end=' ')
        if end_date:
            before_filter = len(df_30min)
            df_30min = df_30min[df_30min.index <= end_date]
            print(f"\nAfter end_date filter: {len(df_30min)} rows (filtered out {before_filter - len(df_30min)})", end=' ')

        df_hourly = self.aggregate_to_hourly(df_30min)
        if df_hourly.empty:
            print(f"\nNo hourly data")
            return pd.DataFrame()

        # Show unique dates
        unique_dates = df_hourly['date'].unique()
        print(f"\n→ {len(unique_dates)} unique trading days", end=' ')

        results = []
        for date in unique_dates:
            day_data = df_hourly[df_hourly['date'] == date]
            classification, debug_info = self.classify_day(day_data)
            results.append({
                'Date': pd.to_datetime(date),
                'DayType': self.CLASSIFICATION[classification],
                'Classification': classification,
                'Opening': debug_info.get('opening_price'),
                'Directional': 'Up' if debug_info.get('dir_up') else ('Down' if debug_info.get('dir_down') else 'None'),
                'HasPullback': debug_info.get('has_pullback'),
                'TestCount': debug_info.get('test_count'),
                'NumSessions': debug_info.get('num_sessions')
            })

        results_df = pd.DataFrame(results)
        print(f"✓")
        return results_df

    def analyze_all(self, symbols: List[str] = None, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        files = self.find_30min_files()
        if not files:
            print("No 30-minute files found!")
            return {}
        if symbols:
            files = {s: p for s, p in files.items() if s in symbols}
        for symbol, file_path in files.items():
            try:
                results_df = self.analyze_symbol(symbol, file_path, start_date, end_date)
                if not results_df.empty:
                    self.results[symbol] = results_df
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        return self.results


# ============================================================================
# TRENDINESS ANALYZER
# ============================================================================
class TrendinessAnalyzer:
    """Analyzes trendiness metrics over time for multiple symbols."""

    def __init__(self, classifier_results: Dict[str, pd.DataFrame]):
        self.results = classifier_results
        self.metrics = {}

    def calculate_metrics(self, df: pd.DataFrame, period_name: str = "Overall") -> Dict:
        """Calculate trendiness metrics for a dataframe."""
        total_days = len(df)
        if total_days == 0:
            return {}

        dnp_count = (df['DayType'] == 'DNP (Directional No Pullbacks)').sum()
        dwp_count = (df['DayType'] == 'DWP (Directional w/ Pullbacks)').sum()
        range_count = (df['DayType'] == 'RANGE DAY').sum()
        na_count = (df['DayType'] == 'N/A').sum()

        directional_count = dnp_count + dwp_count

        return {
            'Period': period_name,
            'Total_Days': total_days,
            'DNP_Count': dnp_count,
            'DWP_Count': dwp_count,
            'Range_Count': range_count,
            'NA_Count': na_count,
            'DNP_Pct': (dnp_count / total_days * 100) if total_days > 0 else 0,
            'DWP_Pct': (dwp_count / total_days * 100) if total_days > 0 else 0,
            'Directional_Pct': (directional_count / total_days * 100) if total_days > 0 else 0,
            'Range_Pct': (range_count / total_days * 100) if total_days > 0 else 0,
            'NA_Pct': (na_count / total_days * 100) if total_days > 0 else 0
        }

    def analyze_by_period(self, df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
        """
        Analyze trendiness by time period.

        Args:
            df: DataFrame with Date column
            freq: Period frequency ('M' = monthly, 'Q' = quarterly, 'Y' = yearly)
        """
        df['Period'] = pd.to_datetime(df['Date']).dt.to_period(freq)

        period_metrics = []
        for period in df['Period'].unique():
            period_df = df[df['Period'] == period]
            metrics = self.calculate_metrics(period_df, str(period))
            period_metrics.append(metrics)

        return pd.DataFrame(period_metrics)

    def analyze_all_symbols(self, freq: str = 'M') -> Dict[str, pd.DataFrame]:
        """Analyze all symbols by time period."""
        symbol_metrics = {}

        for symbol, df in self.results.items():
            # Overall metrics
            overall = self.calculate_metrics(df, "Overall")

            # Period metrics
            period_df = self.analyze_by_period(df, freq)

            symbol_metrics[symbol] = {
                'overall': overall,
                'by_period': period_df
            }

        self.metrics = symbol_metrics
        return symbol_metrics

    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of overall metrics across symbols."""
        comparison_data = []

        for symbol, metrics in self.metrics.items():
            row = metrics['overall'].copy()
            row['Symbol'] = symbol
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Directional_Pct', ascending=False)

        return df

    def plot_trendiness_over_time(self, symbols: List[str] = None, metric: str = 'Directional_Pct'):
        """Plot trendiness metric over time for selected symbols."""
        if symbols is None:
            symbols = list(self.metrics.keys())[:10]  # Top 10 if not specified

        plt.figure(figsize=(14, 8))

        for symbol in symbols:
            if symbol not in self.metrics:
                continue

            period_df = self.metrics[symbol]['by_period']
            if not period_df.empty:
                plt.plot(range(len(period_df)), period_df[metric], marker='o', label=symbol, linewidth=2)

        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel(f'{metric} (%)', fontsize=12)
        plt.title(f'{metric} Over Time by Symbol', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_stability_heatmap(self):
        """Plot heatmap showing stability of trendiness across time periods."""
        # Collect all symbols' directional percentages by period
        data = {}

        for symbol, metrics in self.metrics.items():
            period_df = metrics['by_period']
            if not period_df.empty:
                data[symbol] = period_df['Directional_Pct'].values

        if not data:
            print("No data to plot")
            return

        # Find max length and pad shorter arrays
        max_len = max(len(v) for v in data.values())
        padded_data = {}
        for symbol, values in data.items():
            if len(values) < max_len:
                padded_values = np.pad(values, (0, max_len - len(values)), constant_values=np.nan)
                padded_data[symbol] = padded_values
            else:
                padded_data[symbol] = values

        df_heatmap = pd.DataFrame(padded_data).T

        plt.figure(figsize=(16, 10))
        sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                    vmin=0, vmax=100, cbar_kws={'label': 'Directional %'})
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Symbol', fontsize=12)
        plt.title('Directional % Heatmap Across Time Periods', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = '/content/drive/MyDrive/StockData'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

# Define test basket of symbols
# Mix of different types: indices, tech, value, volatility
TEST_BASKET = [
    # Indices
    'SPY', 'QQQ', 'IWM', 'DIA',
    # High-vol tech
    'NVDA', 'TSLA', 'AMD', 'SMCI',
    # Stable mega-caps
    'AAPL', 'MSFT', 'GOOGL', 'AMZN',
    # Value/defensive
    'JNJ', 'PG', 'KO', 'WMT',
    # Other
    'XLE', 'XLF', 'GLD', 'TLT'
]

print("="*80)
print("TRENDINESS ANALYSIS - Symbol Behavior Over Time")
print("="*80)
print(f"\nTest Basket ({len(TEST_BASKET)} symbols):")
print(", ".join(TEST_BASKET))

# User inputs
symbols_input = input("\nUse test basket? (y/n, or enter custom symbols): ")
if symbols_input.lower().strip() in ['y', 'yes']:
    SYMBOLS = TEST_BASKET
    print(f"✓ Using test basket")
elif symbols_input.upper().strip() == 'ALL':
    SYMBOLS = None
    print("✓ Analyzing ALL symbols")
else:
    SYMBOLS = [s.strip().upper() for s in symbols_input.split(',')]
    print(f"✓ Custom symbols: {', '.join(SYMBOLS)}")

start_date_input = input("Start date (YYYY-MM-DD or ALL): ")
START_DATE = None if start_date_input.upper().strip() == 'ALL' else start_date_input.strip()

end_date_input = input("End date (YYYY-MM-DD or ALL): ")
END_DATE = None if end_date_input.upper().strip() == 'ALL' else end_date_input.strip()

period_freq = input("Analysis period (M=monthly, Q=quarterly, Y=yearly) [default: M]: ").strip().upper()
if period_freq not in ['M', 'Q', 'Y']:
    period_freq = 'M'

print("\n" + "="*80)
print("RUNNING CLASSIFICATION...")
print("="*80)

# Run classifier
classifier = DayTypeClassifier30Min(data_directory=DATA_DIR)
results = classifier.analyze_all(symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE)

print("\n" + "="*80)
print("CALCULATING TRENDINESS METRICS...")
print("="*80)

# Analyze trendiness
analyzer = TrendinessAnalyzer(results)
metrics = analyzer.analyze_all_symbols(freq=period_freq)

# Display comparison table
print("\n" + "="*80)
print("OVERALL TRENDINESS COMPARISON")
print("="*80)
comparison_df = analyzer.create_comparison_table()
print(comparison_df[['Symbol', 'Total_Days', 'DNP_Pct', 'DWP_Pct', 'Directional_Pct', 'Range_Pct']].to_string(index=False))

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'trendiness_comparison.csv'), index=False)

# Plot trendiness over time
print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

# Top trending symbols
top_symbols = comparison_df.nlargest(10, 'Directional_Pct')['Symbol'].tolist()
print(f"\nPlotting top 10 most directional symbols...")
analyzer.plot_trendiness_over_time(symbols=top_symbols)

# Stability heatmap
print("\nGenerating stability heatmap...")
analyzer.plot_stability_heatmap()

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nKey Findings:")
print(f"- Most directional: {comparison_df.iloc[0]['Symbol']} ({comparison_df.iloc[0]['Directional_Pct']:.1f}%)")
print(f"- Least directional: {comparison_df.iloc[-1]['Symbol']} ({comparison_df.iloc[-1]['Directional_Pct']:.1f}%)")
print(f"\nResults saved to: {OUTPUT_DIR}")
print("="*80)
