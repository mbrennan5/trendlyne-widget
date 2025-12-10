#!/usr/bin/env python3
"""
Trading Day Classification Backtest
Using 30-minute bar data with 9:30 OPEN as reference level
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import time


class TradingDayClassifier:
    """
    Classifies trading days based on 9:30 open reference level

    Classification Types:
    - Range Day: Price tested 9:30 level 4+ times OR reversed back after going directional
    - DWP (Directional with Pullbacks): Directional move with hourly pullbacks
    - DNP (Directional No Pullbacks): Clean directional move
    """

    def __init__(self):
        self.results = []

    def classify_day(self, day_data: pd.DataFrame) -> Dict:
        """
        Classify a single trading day using 30-minute bars

        Args:
            day_data: DataFrame with columns [timestamp, open, high, low, close]
                     Must include 9:30 bar and subsequent 30-min bars

        Returns:
            Dictionary with classification results
        """
        # Ensure data is sorted by time
        day_data = day_data.sort_values('timestamp').reset_index(drop=True)

        # Extract date for reference
        trade_date = day_data['timestamp'].iloc[0].date()

        # Get 9:30 bar OPEN as reference level (modified from H930/L930)
        bar_930 = day_data[day_data['timestamp'].dt.time == time(9, 30)]

        if len(bar_930) == 0:
            return {
                'date': trade_date,
                'classification': 'N/A',
                'reason': 'Missing 9:30 bar',
                'open_930': None
            }

        open_930 = bar_930['open'].iloc[0]

        # Initialize tracking variables
        dir_up = False
        dir_down = False
        pullback_occurred = False
        test_count = 0
        reversed_to_930 = False

        # Track hourly highs/lows for pullback detection
        # With 30-min bars: 9:30, 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 15:30, 16:00
        # Hourly marks: 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00
        hourly_times = [time(10, 0), time(11, 0), time(12, 0), time(13, 0),
                       time(14, 0), time(15, 0), time(16, 0)]

        prev_hour_high = None
        prev_hour_low = None
        current_hour_high = None
        current_hour_low = None

        # Process each bar
        for idx, row in day_data.iterrows():
            bar_time = row['timestamp'].time()

            # Skip bars before 9:30
            if bar_time < time(9, 30):
                continue

            # Stop at 16:15 (last RTH bar)
            if bar_time > time(16, 15):
                break

            # Check directional breaks
            if row['high'] > open_930:
                dir_up = True
            if row['low'] < open_930:
                dir_down = True

            is_directional = dir_up or dir_down

            # Test930: Does this bar overlap/test the 9:30 open level?
            # Since we only have one reference (open), we check if price crossed it
            bar_tested_930 = row['low'] <= open_930 <= row['high']

            # Track hourly sessions for pullback detection
            # Update current hour high/low
            if current_hour_high is None:
                current_hour_high = row['high']
                current_hour_low = row['low']
            else:
                current_hour_high = max(current_hour_high, row['high'])
                current_hour_low = min(current_hour_low, row['low'])

            # Check if we hit an hourly mark
            if bar_time in hourly_times:
                # Count tests of 930 level
                if bar_tested_930:
                    test_count += 1

                # Check for pullbacks (skip first hour - 10:00)
                if prev_hour_high is not None and bar_time != time(10, 0):
                    if dir_up and current_hour_low < prev_hour_low:
                        pullback_occurred = True
                    if dir_down and current_hour_high > prev_hour_high:
                        pullback_occurred = True

                # Store this hour's data for next comparison
                prev_hour_high = current_hour_high
                prev_hour_low = current_hour_low

                # Reset for next hour
                current_hour_high = None
                current_hour_low = None

            # Range2 check: Reversed back to 930 level after 11:00
            if bar_time >= time(11, 0) and is_directional and bar_tested_930:
                reversed_to_930 = True

        # Final classification logic
        range1 = test_count >= 4
        range2 = is_directional and reversed_to_930
        is_range_day = range1 or range2

        if is_range_day:
            classification = 'RANGE'
            reason = f"Range1={range1}, Range2={range2}, Tests={test_count}"
        elif is_directional and pullback_occurred:
            classification = 'DWP'
            direction = 'UP' if dir_up and not dir_down else 'DOWN' if dir_down and not dir_up else 'BOTH'
            reason = f"Directional ({direction}) with pullbacks"
        elif is_directional and not pullback_occurred:
            classification = 'DNP'
            direction = 'UP' if dir_up and not dir_down else 'DOWN' if dir_down and not dir_up else 'BOTH'
            reason = f"Directional ({direction}) no pullbacks"
        else:
            classification = 'N/A'
            reason = 'No breakout from 930 level'

        return {
            'date': trade_date,
            'classification': classification,
            'reason': reason,
            'open_930': open_930,
            'dir_up': dir_up,
            'dir_down': dir_down,
            'pullback_occurred': pullback_occurred,
            'test_count': test_count,
            'reversed_to_930': reversed_to_930,
            'range1': range1,
            'range2': range2
        }

    def run_backtest(self, data_path: str, symbol: str = None) -> pd.DataFrame:
        """
        Run backtest on 30-minute bar data

        Args:
            data_path: Path to CSV file or directory of CSV files
            symbol: Optional symbol filter

        Returns:
            DataFrame with classification results
        """
        path = Path(data_path)

        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.glob('*.csv'))
            if symbol:
                files = [f for f in files if symbol.upper() in f.name.upper()]
        else:
            raise ValueError(f"Path not found: {data_path}")

        print(f"Processing {len(files)} file(s)...")

        all_results = []

        for file in files:
            print(f"Processing {file.name}...")

            # Load data
            df = pd.read_csv(file)

            # Standardize column names (handle various formats)
            df.columns = df.columns.str.lower()

            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            else:
                print(f"Warning: Cannot find timestamp column in {file.name}")
                continue

            # Group by trading day
            df['date'] = df['timestamp'].dt.date

            for date, day_data in df.groupby('date'):
                result = self.classify_day(day_data)
                result['symbol'] = file.stem
                all_results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Generate summary statistics
        self.print_summary(results_df)

        return results_df

    def print_summary(self, results_df: pd.DataFrame):
        """Print backtest summary statistics"""
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)

        total_days = len(results_df)
        print(f"\nTotal Trading Days: {total_days}")

        # Classification breakdown
        print("\nClassification Breakdown:")
        for classification in ['RANGE', 'DWP', 'DNP', 'N/A']:
            count = len(results_df[results_df['classification'] == classification])
            pct = (count / total_days * 100) if total_days > 0 else 0
            print(f"  {classification:6s}: {count:4d} ({pct:5.1f}%)")

        # Directional statistics
        directional = results_df[results_df['classification'].isin(['DWP', 'DNP'])]
        if len(directional) > 0:
            print(f"\nDirectional Days: {len(directional)} ({len(directional)/total_days*100:.1f}%)")
            print(f"  With Pullbacks (DWP): {len(directional[directional['classification']=='DWP'])}")
            print(f"  No Pullbacks (DNP):   {len(directional[directional['classification']=='DNP'])}")

        # Range day breakdown
        range_days = results_df[results_df['classification'] == 'RANGE']
        if len(range_days) > 0:
            range1_count = range_days['range1'].sum()
            range2_count = range_days['range2'].sum()
            print(f"\nRange Day Breakdown:")
            print(f"  Range1 (4+ tests): {range1_count}")
            print(f"  Range2 (reversal): {range2_count}")

        print("\n" + "="*80 + "\n")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Trading Day Classification Backtest')
    parser.add_argument('data_path', help='Path to data file or directory')
    parser.add_argument('--symbol', help='Filter by symbol', default=None)
    parser.add_argument('--output', help='Output CSV file', default='backtest_results.csv')

    args = parser.parse_args()

    # Run backtest
    classifier = TradingDayClassifier()
    results = classifier.run_backtest(args.data_path, args.symbol)

    # Save results
    results.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")

    # Display first few results
    print("\nSample Results:")
    print(results.head(10).to_string())


if __name__ == '__main__':
    main()
