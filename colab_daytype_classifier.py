"""
Google Colab - Day Type Classifier using 30-Minute Data
Complete code for analyzing trading day types from 30-minute CSV files

To use in Google Colab:
1. Mount Google Drive
2. Ensure 30-minute data files are in /content/drive/MyDrive/StockData/
3. Run this cell
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import pytz
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Tuple
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted successfully!")
print("=" * 80)


# ============================================================================
# DAY TYPE CLASSIFIER CLASS
# ============================================================================
class DayTypeClassifier30Min:
    """Classifies trading days using 30-minute bars aggregated into hourly sessions."""

    # RTH session times
    RTH_START = time(9, 30)
    RTH_END = time(16, 15)

    # Hourly session start times (EST)
    HOURLY_SESSIONS = [
        time(9, 30),   # 0: 9:30-10:30
        time(10, 30),  # 1: 10:30-11:30
        time(11, 30),  # 2: 11:30-12:30
        time(12, 30),  # 3: 12:30-13:30
        time(13, 30),  # 4: 13:30-14:30
        time(14, 30),  # 5: 14:30-15:30
        time(15, 30),  # 6: 15:30-16:15 (partial)
    ]

    CLASSIFICATION = {
        0: "N/A",
        1: "RANGE DAY",
        2: "DWP (Directional w/ Pullbacks)",
        3: "DNP (Directional No Pullbacks)"
    }

    def __init__(self, data_directory: str, timezone: str = 'America/New_York'):
        self.data_directory = Path(data_directory)
        self.timezone = pytz.timezone(timezone)
        self.results = {}

    def find_30min_files(self) -> Dict[str, Path]:
        """Find all 30-minute data files."""
        files = {}
        for file_path in self.data_directory.glob('*30Min*.csv'):
            filename = file_path.stem
            symbol = filename.split('_')[0].split('-')[0].upper()
            files[symbol] = file_path
        return files

    def load_30min_data(self, file_path: Path) -> pd.DataFrame:
        """Load 30-minute data from CSV file."""
        df = pd.read_csv(file_path)

        # Handle datetime column (based on your config: 't' column)
        if 't' in df.columns:
            df['datetime'] = pd.to_datetime(df['t'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])

        # Ensure OHLC columns
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        df.set_index('datetime', inplace=True)

        # Timezone handling
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.timezone)
        else:
            df.index = df.index.tz_convert(self.timezone)

        return df

    def aggregate_to_hourly(self, df_30min: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 30-minute bars into hourly sessions."""
        df_rth = df_30min.between_time(self.RTH_START, self.RTH_END).copy()

        if df_rth.empty:
            return pd.DataFrame()

        df_rth['date'] = df_rth.index.date
        df_rth['time'] = df_rth.index.time

        def assign_hourly_session(t):
            """Assign 30-min bar to hourly session."""
            if time(9, 30) <= t < time(10, 30):
                return 0
            elif time(10, 30) <= t < time(11, 30):
                return 1
            elif time(11, 30) <= t < time(12, 30):
                return 2
            elif time(12, 30) <= t < time(13, 30):
                return 3
            elif time(13, 30) <= t < time(14, 30):
                return 4
            elif time(14, 30) <= t < time(15, 30):
                return 5
            elif time(15, 30) <= t <= time(16, 15):
                return 6
            else:
                return -1

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
        """
        Classify a single trading day.

        Implements exact ThinkScript logic:
        1. Skip first bar (9:30) for directional detection
        2. Skip last bar (3:30) for pullback detection
        3. Use opening price as both H930 and L930
        """
        if len(day_data) < 2:
            return 0, {"reason": "Insufficient data"}

        day_data = day_data.sort_values('session')

        # Opening price is both H930 and L930
        opening_price = day_data.iloc[0]['Open']
        H930 = L930 = opening_price

        # Initialize
        dir_up = False
        dir_down = False
        has_pullback = False
        test_count = 0
        reversed_to_opening = False
        prev_high = None
        prev_low = None

        # Process each hourly session
        for i, (idx, row) in enumerate(day_data.iterrows()):
            hour_high = row['High']
            hour_low = row['Low']
            session_num = row['session']

            # Directional bias (SKIP FIRST BAR)
            if i > 0:
                if hour_high > H930:
                    dir_up = True
                if hour_low < L930:
                    dir_down = True

            # Test opening price overlap
            if hour_low <= H930 and hour_high >= L930:
                test_count += 1

            # Pullback check (SKIP FIRST AND LAST BARS)
            is_last_bar = (i == len(day_data) - 1)

            if i > 0 and prev_high is not None and prev_low is not None and not is_last_bar:
                if dir_up and hour_low < prev_low:
                    has_pullback = True
                elif dir_down and hour_high > prev_high:
                    has_pullback = True

            # Late reversal (after 11:00 AM)
            if session_num >= 2:
                if hour_high >= L930 and hour_low <= H930:
                    reversed_to_opening = True

            prev_high = hour_high
            prev_low = hour_low

        # Classification logic
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

    def analyze_symbol(self, symbol: str, file_path: Path,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Analyze all trading days for a symbol."""
        print(f"\nAnalyzing {symbol}...")

        df_30min = self.load_30min_data(file_path)
        if df_30min.empty:
            print(f"No data loaded for {symbol}")
            return pd.DataFrame()

        if start_date:
            df_30min = df_30min[df_30min.index >= start_date]
        if end_date:
            df_30min = df_30min[df_30min.index <= end_date]

        df_hourly = self.aggregate_to_hourly(df_30min)
        if df_hourly.empty:
            print(f"No hourly data for {symbol}")
            return pd.DataFrame()

        results = []
        for date in df_hourly['date'].unique():
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

        return pd.DataFrame(results)

    def analyze_all(self, symbols: List[str] = None,
                    start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Analyze all symbols."""
        files = self.find_30min_files()

        if not files:
            print("No 30-minute files found!")
            return {}

        print(f"Found {len(files)} symbol(s): {', '.join(sorted(files.keys()))}")

        if symbols:
            files = {s: p for s, p in files.items() if s in symbols}

        for symbol, file_path in files.items():
            try:
                results_df = self.analyze_symbol(symbol, file_path, start_date, end_date)
                if not results_df.empty:
                    self.results[symbol] = results_df
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                import traceback
                traceback.print_exc()

        return self.results

    def print_results(self):
        """Print formatted results."""
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
        """Export results to CSV."""
        all_results = []
        for symbol, df in self.results.items():
            df_copy = df.copy()
            df_copy.insert(0, 'Symbol', symbol)
            all_results.append(df_copy)

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df.to_csv(filename, index=False)
            print(f"\nResults exported to {filename}")

            # Auto-download in Colab
            try:
                from google.colab import files
                files.download(filename)
                print("CSV file downloaded!")
            except:
                print("Not in Colab - CSV saved locally")

            return combined_df
        return None


# ============================================================================
# CONFIGURATION AND EXECUTION
# ============================================================================

# Configuration
DATA_DIR = '/content/drive/MyDrive/StockData'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

# User inputs
print("\n" + "="*80)
print("DAY TYPE CLASSIFICATION TOOL")
print("="*80)

# Ask for symbols
symbols_input = input("Enter symbols to analyze (comma-separated, or 'ALL'): ")
if symbols_input.upper().strip() == 'ALL':
    SYMBOLS = None
else:
    SYMBOLS = [s.strip().upper() for s in symbols_input.split(',')]

# Ask for date range
start_date_input = input("Enter start date (YYYY-MM-DD, or 'ALL'): ")
START_DATE = None if start_date_input.upper().strip() == 'ALL' else start_date_input.strip()

end_date_input = input("Enter end date (YYYY-MM-DD, or 'ALL'): ")
END_DATE = None if end_date_input.upper().strip() == 'ALL' else end_date_input.strip()

print("\n" + "="*80)
print("Running Analysis...")
print("="*80)

# Create classifier
classifier = DayTypeClassifier30Min(data_directory=DATA_DIR)

# Analyze
classifier.analyze_all(symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE)

# Print results
classifier.print_results()

# Export to CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, 'daytype_classification_results.csv')
classifier.export_to_csv(output_file)

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
