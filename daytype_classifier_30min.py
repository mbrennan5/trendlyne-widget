#!/usr/bin/env python3
"""
Trading Day Classification - 30-Minute Data Version
Matches ThinkScript logic exactly using 30-minute bars aggregated into hourly sessions

Based on the debugging insights:
1. Opening price (9:30 open) is used as both H930 and L930
2. Directional bias detection starts from SECOND hourly session (10:30)
3. First bar (9:30) is skipped for directional detection (matches isNewDay reset)
4. Last bar (3:30 PM) is excluded from pullback detection
"""

import pandas as pd
import numpy as np
import pytz
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Tuple
import os


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
        """
        Initialize classifier.

        Args:
            data_directory: Path to directory containing 30-minute CSV files
            timezone: Timezone for the data (default: America/New_York)
        """
        self.data_directory = Path(data_directory)
        self.timezone = pytz.timezone(timezone)
        self.results = {}

    def find_30min_files(self) -> Dict[str, Path]:
        """
        Find all 30-minute data files in the directory.

        Returns:
            Dictionary mapping symbol to file path
        """
        files = {}

        # Look for files matching pattern: SYMBOL_30Min.csv or similar
        for file_path in self.data_directory.glob('*30Min*.csv'):
            # Extract symbol from filename
            filename = file_path.stem
            # Assuming format like "SPY_30Min" or "SPY-30Min"
            symbol = filename.split('_')[0].split('-')[0].upper()
            files[symbol] = file_path

        return files

    def load_30min_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load 30-minute data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with 30-minute bars
        """
        # Read CSV - adjust column names based on your file format
        df = pd.read_csv(file_path)

        # Handle different possible column name formats
        # Based on your config: 't', 'Open', 'High', 'Low', 'Close', 'Volume'
        if 't' in df.columns:
            df['datetime'] = pd.to_datetime(df['t'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        else:
            # Try first column
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])

        # Ensure we have OHLC columns
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Set datetime as index
        df.set_index('datetime', inplace=True)

        # Ensure timezone aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.timezone)
        else:
            df.index = df.index.tz_convert(self.timezone)

        return df

    def aggregate_to_hourly(self, df_30min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 30-minute bars into hourly sessions.

        Two 30-minute bars combine into one hourly bar:
        - 9:30-10:00 + 10:00-10:30 → 9:30 hourly session
        - 10:30-11:00 + 11:00-11:30 → 10:30 hourly session
        - etc.

        Args:
            df_30min: DataFrame with 30-minute bars

        Returns:
            DataFrame with hourly aggregated bars
        """
        # Filter to RTH only
        df_rth = df_30min.between_time(self.RTH_START, self.RTH_END).copy()

        if df_rth.empty:
            return pd.DataFrame()

        # Group by date and hourly session
        df_rth['date'] = df_rth.index.date
        df_rth['time'] = df_rth.index.time

        # Assign each 30-min bar to an hourly session
        def assign_hourly_session(t):
            """Assign 30-min bar to hourly session."""
            # 9:30-10:29 → session 0 (9:30)
            # 10:30-11:29 → session 1 (10:30)
            # etc.
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

        # Remove any bars outside RTH
        df_rth = df_rth[df_rth['hourly_session'] >= 0].copy()

        # Aggregate into hourly bars
        hourly_bars = []

        for date in df_rth['date'].unique():
            day_data = df_rth[df_rth['date'] == date]

            for session_num in range(7):  # 0-6 sessions
                session_data = day_data[day_data['hourly_session'] == session_num]

                if session_data.empty:
                    continue

                # Aggregate: Open of first bar, High/Low of all bars, Close of last bar
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
        1. Skip first bar (9:30) for directional detection (isNewDay reset)
        2. Skip last bar (3:30) for pullback detection
        3. Use opening price as both H930 and L930

        Args:
            day_data: DataFrame with hourly bars for one day

        Returns:
            Tuple of (classification_code, debug_info)
        """
        if len(day_data) < 2:
            return 0, {"reason": "Insufficient data"}

        # Sort by session to ensure order
        day_data = day_data.sort_values('session')

        # Opening price (9:30 open) is both H930 and L930
        opening_price = day_data.iloc[0]['Open']
        H930 = L930 = opening_price

        # Initialize tracking variables
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

            # Check directional bias (SKIP FIRST BAR - matches isNewDay reset)
            if i > 0:  # Start checking from 10:30 (second bar)
                if hour_high > H930:
                    dir_up = True
                if hour_low < L930:
                    dir_down = True

            # Check if this hour overlaps with opening price (for Range1 condition)
            if hour_low <= H930 and hour_high >= L930:
                test_count += 1

            # Check for pullbacks (SKIP FIRST AND LAST BARS)
            is_last_bar = (i == len(day_data) - 1)

            if i > 0 and prev_high is not None and prev_low is not None and not is_last_bar:
                # Pullback detection
                if dir_up and hour_low < prev_low:
                    has_pullback = True
                elif dir_down and hour_high > prev_high:
                    has_pullback = True

            # Check for late reversal (after 11:00 AM, which is session 1 or later)
            # In ThinkScript: isAfter1100 = SecondsFromTime(1100) >= 0
            if session_num >= 2:  # Sessions 2+ are 11:30 and later
                if hour_high >= L930 and hour_low <= H930:
                    reversed_to_opening = True

            # Store for next iteration
            prev_high = hour_high
            prev_low = hour_low

        # Determine if directional
        is_directional = dir_up or dir_down

        # Range Day conditions
        range1 = test_count >= 4  # 4+ hours tested opening price
        range2 = is_directional and reversed_to_opening  # Directional but reversed
        is_range_day = range1 or range2

        # Final classification (matches ThinkScript exactly)
        if is_range_day:
            classification = 1  # RANGE DAY
        elif is_directional and has_pullback:
            classification = 2  # DWP
        elif is_directional and not has_pullback:
            classification = 3  # DNP
        else:
            classification = 0  # N/A

        # Debug info
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
        """
        Analyze all trading days for a symbol.

        Args:
            symbol: Ticker symbol
            file_path: Path to 30-minute CSV file
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with day-by-day classifications
        """
        print(f"\nAnalyzing {symbol}...")

        # Load 30-minute data
        df_30min = self.load_30min_data(file_path)

        if df_30min.empty:
            print(f"No data loaded for {symbol}")
            return pd.DataFrame()

        # Filter by date range if specified
        if start_date:
            df_30min = df_30min[df_30min.index >= start_date]
        if end_date:
            df_30min = df_30min[df_30min.index <= end_date]

        # Aggregate to hourly
        df_hourly = self.aggregate_to_hourly(df_30min)

        if df_hourly.empty:
            print(f"No hourly data for {symbol}")
            return pd.DataFrame()

        # Group by date and classify
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
                'NumSessions': debug_info.get('num_sessions'),
                'Range1': debug_info.get('range1'),
                'Range2': debug_info.get('range2')
            })

        results_df = pd.DataFrame(results)
        return results_df

    def analyze_all(self, symbols: List[str] = None,
                    start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Analyze all symbols (or specified symbols).

        Args:
            symbols: Optional list of symbols to analyze (None = all)
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbol to results DataFrame
        """
        # Find all 30-minute files
        files = self.find_30min_files()

        if not files:
            print("No 30-minute files found!")
            return {}

        print(f"Found {len(files)} symbol(s): {', '.join(sorted(files.keys()))}")

        # Filter to specified symbols if provided
        if symbols:
            files = {s: p for s, p in files.items() if s in symbols}
            if not files:
                print(f"None of the specified symbols found!")
                return {}

        # Analyze each symbol
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
        """Print formatted results for all symbols."""
        for symbol, df in self.results.items():
            print(f"\n{'='*80}")
            print(f"Symbol: {symbol}")
            print(f"{'='*80}")
            print(df.to_string(index=False))

            # Summary statistics
            if len(df) > 0:
                print(f"\n{'-'*80}")
                print("Summary:")
                print(df['DayType'].value_counts().to_string())
                print(f"\nTotal trading days analyzed: {len(df)}")

    def export_to_csv(self, filename: str = "daytype_results.csv"):
        """
        Export results to CSV.

        Args:
            filename: Output CSV filename
        """
        all_results = []
        for symbol, df in self.results.items():
            df_copy = df.copy()
            df_copy.insert(0, 'Symbol', symbol)
            all_results.append(df_copy)

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df.to_csv(filename, index=False)
            print(f"\nResults exported to {filename}")
            return combined_df
        return None


# ============================================================================
# STANDALONE USAGE (for local Python, not Colab)
# ============================================================================
def main():
    """Main function for standalone usage."""

    # Configuration
    DATA_DIR = "/path/to/your/30min/data"  # Update this path
    SYMBOLS = ['SPY', 'QQQ']  # Or None for all symbols
    START_DATE = "2024-01-01"  # Or None for all available
    END_DATE = None  # Or None for all available

    print("="*80)
    print("Day Type Classification - 30-Minute Data")
    print("="*80)

    # Create classifier
    classifier = DayTypeClassifier30Min(data_directory=DATA_DIR)

    # Analyze
    classifier.analyze_all(symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE)

    # Print results
    classifier.print_results()

    # Export
    classifier.export_to_csv("daytype_classification_30min.csv")


if __name__ == "__main__":
    main()
