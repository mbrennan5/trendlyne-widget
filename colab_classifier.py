"""
============================================================================
DAY TYPE CLASSIFIER - 30-MINUTE DATA (GOOGLE COLAB) - MULTI-YEAR VERSION
============================================================================
Copy this entire code block into a Google Colab cell and run it.
Handles multiple year files per symbol: SYMBOL_30Min_YEAR_startdate_enddate.csv
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
import re

# Classifier Class
class DayTypeClassifier30Min:
    RTH_START = time(9, 30)
    RTH_END = time(16, 15)
    CLASSIFICATION = {0: "N/A", 1: "RANGE DAY", 2: "DWP (Directional w/ Pullbacks)", 3: "DNP (Directional No Pullbacks)"}

    def __init__(self, data_directory: str, timezone: str = 'America/New_York'):
        self.data_directory = Path(data_directory)
        self.timezone = pytz.timezone(timezone)
        self.results = {}

    def find_30min_files(self) -> Dict[str, List[Tuple[int, Path]]]:
        """
        Find all 30-minute files and group by symbol with year information.
        Returns: {symbol: [(year, file_path), ...]}
        """
        symbol_files = {}

        for file_path in self.data_directory.glob('*30Min*.csv'):
            filename = file_path.stem

            # Extract symbol - everything before first underscore
            parts = filename.split('_')
            if len(parts) < 2:
                continue
            symbol = parts[0].upper()

            # Try to extract year from filename
            # Pattern: SYMBOL_30Min_YEAR_startdate_enddate
            year = None
            for part in parts:
                if part.isdigit() and len(part) == 4 and 2000 <= int(part) <= 2100:
                    year = int(part)
                    break

            if year is None:
                print(f"  Warning: Could not extract year from {filename}, skipping")
                continue

            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append((year, file_path))

        # Sort each symbol's files by year
        for symbol in symbol_files:
            symbol_files[symbol].sort(key=lambda x: x[0])

        return symbol_files

    def load_multi_year_data(self, symbol: str, year_files: List[Tuple[int, Path]],
                             start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """
        Load and concatenate multiple year files for a symbol.
        """
        # Filter files by year range
        if start_year is None:
            start_year = min(year for year, _ in year_files)
        if end_year is None:
            end_year = max(year for year, _ in year_files)

        filtered_files = [(year, path) for year, path in year_files
                         if start_year <= year <= end_year]

        if not filtered_files:
            print(f"  No files found for {symbol} in year range {start_year}-{end_year}")
            return pd.DataFrame()

        print(f"  Loading {len(filtered_files)} file(s) for years {start_year}-{end_year}:")

        all_dataframes = []

        for year, file_path in filtered_files:
            print(f"    - {year}: {file_path.name}", end=' ')

            try:
                df = pd.read_csv(file_path)
                print(f"({len(df):,} rows)", end='')

                # Handle datetime column
                if 't' in df.columns:
                    df['datetime'] = pd.to_datetime(df['t'], utc=True)
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                else:
                    df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)

                # Rename columns
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

                # Set datetime as index
                df.set_index('datetime', inplace=True)

                # Convert to target timezone
                try:
                    df.index = df.index.tz_convert(self.timezone)
                except Exception as e:
                    if not hasattr(df.index, 'tz') or df.index.tz is None:
                        df.index = pd.DatetimeIndex(df.index).tz_localize('UTC').tz_convert(self.timezone)

                # Show date range
                if len(df) > 0:
                    print(f" → {df.index.min().date()} to {df.index.max().date()}")

                all_dataframes.append(df)

            except Exception as e:
                print(f" ✗ Error: {e}")
                continue

        if not all_dataframes:
            return pd.DataFrame()

        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes)

        # Sort by datetime
        combined_df.sort_index(inplace=True)

        # Remove duplicates if any
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        print(f"  Combined: {len(combined_df):,} total rows | {combined_df.index.min().date()} to {combined_df.index.max().date()}")

        return combined_df

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

    def analyze_symbol(self, symbol: str, year_files: List[Tuple[int, Path]],
                       start_year: int = None, end_year: int = None,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print(f"Analyzing {symbol}")
        print(f"{'='*60}")

        df_30min = self.load_multi_year_data(symbol, year_files, start_year, end_year)
        if df_30min.empty:
            print(f"  ✗ No data loaded")
            return pd.DataFrame()

        # Apply date filtering with diagnostics
        original_len = len(df_30min)
        if start_date:
            df_30min = df_30min[df_30min.index >= start_date]
            print(f"  After start_date filter: {len(df_30min):,} rows ({original_len - len(df_30min):,} filtered)")
            original_len = len(df_30min)

        if end_date:
            df_30min = df_30min[df_30min.index <= end_date]
            print(f"  After end_date filter: {len(df_30min):,} rows ({original_len - len(df_30min):,} filtered)")

        df_hourly = self.aggregate_to_hourly(df_30min)
        if df_hourly.empty:
            print(f"  ✗ No hourly data after aggregation")
            return pd.DataFrame()

        # Show unique trading days
        unique_dates = df_hourly['date'].unique()
        print(f"  Trading days found: {len(unique_dates)}")
        print(f"  Date range: {unique_dates.min()} to {unique_dates.max()}")

        results = []
        for date in unique_dates:
            day_data = df_hourly[df_hourly['date'] == date]
            classification, debug_info = self.classify_day(day_data)
            results.append({
                'Date': date,
                'DayType': self.CLASSIFICATION[classification],
                'Classification': classification,
                'Opening': debug_info.get('opening_price'),
                'Directional': 'Up' if debug_info.get('dir_up') else ('Down' if debug_info.get('dir_down') else 'None'),
                'HasPullback': debug_info.get('has_pullback'),
                'TestCount': debug_info.get('test_count'),
                'NumSessions': debug_info.get('num_sessions')
            })

        print(f"  ✓ Classification complete")
        return pd.DataFrame(results)

    def analyze_all(self, symbols: List[str] = None,
                    start_year: int = None, end_year: int = None,
                    start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        symbol_files = self.find_30min_files()
        if not symbol_files:
            print("No 30-minute files found!")
            return {}

        print(f"\nFound {len(symbol_files)} symbol(s) with year files:")
        for symbol, year_list in sorted(symbol_files.items()):
            years = [year for year, _ in year_list]
            print(f"  {symbol}: {len(year_list)} file(s) → years {min(years)}-{max(years)}")

        if symbols:
            symbol_files = {s: f for s, f in symbol_files.items() if s in symbols}
            if not symbol_files:
                print(f"Warning: None of the requested symbols found!")
                return {}

        for symbol, year_files in symbol_files.items():
            try:
                results_df = self.analyze_symbol(symbol, year_files, start_year, end_year, start_date, end_date)
                if not results_df.empty:
                    self.results[symbol] = results_df
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                import traceback
                traceback.print_exc()

        return self.results

    def print_results(self):
        for symbol, df in self.results.items():
            print(f"\n{'='*80}")
            print(f"Symbol: {symbol}")
            print(f"{'='*80}")
            print(df.to_string(index=False))
            if len(df) > 0:
                print(f"\n{'-'*80}")
                print("Summary:")
                print(df['DayType'].value_counts().to_string())

    def export_to_csv(self, filename: str = "daytype_results.csv"):
        all_results = []
        for symbol, df in self.results.items():
            df_copy = df.copy()
            df_copy.insert(0, 'Symbol', symbol)
            all_results.append(df_copy)
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df.to_csv(filename, index=False)
            print(f"\nResults exported to {filename}")
            try:
                from google.colab import files
                files.download(filename)
                print("CSV file downloaded!")
            except:
                print("Not in Colab - CSV saved locally")
            return combined_df
        return None

# Configuration
DATA_DIR = '/content/drive/MyDrive/StockData'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

# User Input
print("\n" + "="*80)
print("DAY TYPE CLASSIFIER - MULTI-YEAR VERSION")
print("="*80)
symbols_input = input("Enter symbols (comma-separated, or 'ALL'): ")
SYMBOLS = None if symbols_input.upper().strip() == 'ALL' else [s.strip().upper() for s in symbols_input.split(',')]

print("\n" + "-"*80)
print("YEAR RANGE (files are per-year: SYMBOL_30Min_YEAR_startdate_enddate.csv)")
print("-"*80)
start_year_input = input("Start YEAR (e.g., 2020, or press Enter for all): ")
START_YEAR = None if start_year_input.strip() == '' else int(start_year_input.strip())
end_year_input = input("End YEAR (e.g., 2025, or press Enter for all): ")
END_YEAR = None if end_year_input.strip() == '' else int(end_year_input.strip())

print("\n" + "-"*80)
print("DATE RANGE (optional - further filter by specific dates)")
print("-"*80)
start_date_input = input("Start date (YYYY-MM-DD or press Enter for all): ")
START_DATE = None if start_date_input.strip() == '' else start_date_input.strip()
end_date_input = input("End date (YYYY-MM-DD or press Enter for all): ")
END_DATE = None if end_date_input.strip() == '' else end_date_input.strip()

# Run Analysis
print("\n" + "="*80)
print("RUNNING ANALYSIS...")
print("="*80)
classifier = DayTypeClassifier30Min(data_directory=DATA_DIR)
classifier.analyze_all(symbols=SYMBOLS, start_year=START_YEAR, end_year=END_YEAR,
                       start_date=START_DATE, end_date=END_DATE)
classifier.print_results()

# Export
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, 'daytype_classification_results.csv')
classifier.export_to_csv(output_file)

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print(f"Results saved to: {output_file}")
print("CSV downloaded to your computer")
print("="*80)
