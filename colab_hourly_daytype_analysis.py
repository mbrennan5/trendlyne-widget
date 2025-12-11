"""
Google Colab - Hourly Day Type Analysis
Complete code block for analyzing trading days using 1-hour bars from Yahoo Finance

To use in Google Colab:
1. Copy this entire cell
2. Paste into a Colab notebook cell
3. Run the cell
4. Modify SYMBOLS and LOOKBACK_DAYS as needed
"""

# Install dependencies (run this first in Colab)
# !pip install yfinance pandas numpy pytz

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Tuple


class HourlyDayTypeAnalyzer:
    """Analyzes trading days using 1-hour bars and classifies them by day type."""

    # RTH session times (Regular Trading Hours)
    RTH_START = "09:30"
    RTH_END = "16:15"

    # Day type classifications
    CLASSIFICATION = {
        0: "N/A",
        1: "RANGE DAY",
        2: "DWP (Directional w/ Pullbacks)",
        3: "DNP (Directional No Pullbacks)"
    }

    def __init__(self, symbols: List[str], lookback_days: int = 30):
        """
        Initialize the analyzer.

        Args:
            symbols: List of ticker symbols to analyze
            lookback_days: Number of trading days to look back
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.results = {}

    def download_data(self, symbol: str) -> pd.DataFrame:
        """
        Download 1-hour data from Yahoo Finance.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with 1-hour OHLCV data
        """
        # Download extra days to ensure we have enough trading days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days * 2)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1h")

        if df.empty:
            print(f"No data returned for {symbol}")
            return df

        # Ensure timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')

        return df

    def filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to Regular Trading Hours (9:30 AM - 4:15 PM EST).

        Args:
            df: DataFrame with hourly data

        Returns:
            Filtered DataFrame with only RTH hours
        """
        # Keep only hours during RTH
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        # RTH: 9:30 AM to 4:15 PM (09:30 to 16:15)
        # Keep bars at 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
        rth_hours = [9, 10, 11, 12, 13, 14, 15]

        # Filter for RTH hours
        df_rth = df[
            ((df['hour'].isin(rth_hours[1:])) & (df['minute'] == 30)) |  # 10:30-15:30
            ((df['hour'] == 9) & (df['minute'] == 30))  # 9:30
        ].copy()

        return df_rth

    def classify_day(self, day_data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Classify a single trading day.

        Args:
            day_data: DataFrame with hourly bars for one trading day

        Returns:
            Tuple of (classification_code, debug_info)
        """
        if len(day_data) < 2:
            return 0, {"reason": "Insufficient data"}

        # Sort by time to ensure chronological order
        day_data = day_data.sort_index()

        # Opening price is both H930 and L930
        opening_price = day_data.iloc[0]['Open']
        H930 = L930 = opening_price

        # Initialize tracking variables
        dir_up = False
        dir_down = False
        has_pullback = False
        test_count = 0
        reversed_to_opening = False

        # Track previous hour's high and low for pullback detection
        prev_high = None
        prev_low = None

        # Process each hour
        for i, (timestamp, row) in enumerate(day_data.iterrows()):
            hour_high = row['High']
            hour_low = row['Low']

            # Check directional bias
            if hour_high > H930:
                dir_up = True
            if hour_low < L930:
                dir_down = True

            # Check if this hour overlaps with opening price (for Range1 condition)
            if hour_low <= H930 and hour_high >= L930:
                test_count += 1

            # Check for pullbacks (starting from second hour, which is 10:30)
            if i > 0 and prev_high is not None and prev_low is not None:
                if dir_up and hour_low < prev_low:
                    has_pullback = True
                elif dir_down and hour_high > prev_high:
                    has_pullback = True

            # Check for late reversal (after 11:00 AM, which is the 2nd or 3rd hour)
            # Time check: if timestamp is 11:30 or later
            if timestamp.hour >= 11 and timestamp.hour * 60 + timestamp.minute >= 11 * 60:
                if hour_high >= L930 and hour_low <= H930:
                    reversed_to_opening = True

            # Store current hour's high/low for next iteration
            prev_high = hour_high
            prev_low = hour_low

        # Determine if directional
        is_directional = dir_up or dir_down

        # Range Day conditions
        range1 = test_count >= 4  # 4+ hours tested opening price
        range2 = is_directional and reversed_to_opening  # Directional but reversed
        is_range_day = range1 or range2

        # Final classification
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
            "num_hours": len(day_data)
        }

        return classification, debug_info

    def analyze_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Analyze all trading days for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with day-by-day classifications
        """
        print(f"\nAnalyzing {symbol}...")

        # Download data
        df = self.download_data(symbol)
        if df.empty:
            return pd.DataFrame()

        # Filter to RTH
        df_rth = self.filter_rth(df)

        if df_rth.empty:
            print(f"No RTH data for {symbol}")
            return pd.DataFrame()

        # Group by date
        df_rth['date'] = df_rth.index.date

        # Get unique dates and take the last N trading days
        unique_dates = sorted(df_rth['date'].unique())
        recent_dates = unique_dates[-self.lookback_days:]

        # Analyze each day
        results = []
        for date in recent_dates:
            day_data = df_rth[df_rth['date'] == date]
            classification, debug_info = self.classify_day(day_data)

            results.append({
                'Date': date,
                'DayType': self.CLASSIFICATION[classification],
                'Classification': classification,
                'Opening': debug_info.get('opening_price'),
                'Directional': 'Up' if debug_info.get('dir_up') else ('Down' if debug_info.get('dir_down') else 'None'),
                'HasPullback': debug_info.get('has_pullback'),
                'TestCount': debug_info.get('test_count'),
                'NumHours': debug_info.get('num_hours')
            })

        results_df = pd.DataFrame(results)
        return results_df

    def analyze_all(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze all symbols.

        Returns:
            Dictionary mapping symbol to results DataFrame
        """
        for symbol in self.symbols:
            try:
                results_df = self.analyze_symbol(symbol)
                if not results_df.empty:
                    self.results[symbol] = results_df
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

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

    def export_to_csv(self, filename: str = "daytype_analysis.csv"):
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


# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# List of symbols to analyze
SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']

# Number of trading days to analyze
LOOKBACK_DAYS = 20

# =============================================================================
# RUN ANALYSIS
# =============================================================================

print("="*80)
print("Hourly Day Type Analysis")
print("="*80)
print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"Lookback Days: {LOOKBACK_DAYS}")
print("="*80)

# Create analyzer
analyzer = HourlyDayTypeAnalyzer(symbols=SYMBOLS, lookback_days=LOOKBACK_DAYS)

# Run analysis
analyzer.analyze_all()

# Print results
analyzer.print_results()

# Export to CSV and display in Colab
results_df = analyzer.export_to_csv("daytype_analysis.csv")

# Display combined results as a table in Colab
if results_df is not None:
    print("\n" + "="*80)
    print("COMBINED RESULTS")
    print("="*80)
    display(results_df)  # This will show a nice table in Colab

    # Download the CSV file (for Colab)
    try:
        from google.colab import files
        files.download('daytype_analysis.csv')
        print("\nCSV file downloaded!")
    except:
        print("\nNot running in Colab - CSV saved locally")
