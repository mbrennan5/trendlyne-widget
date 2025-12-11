import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import sys
import glob
import os
from io import StringIO

warnings.filterwarnings('ignore')

# --- Helper Function for Price Type Selection ---
def calculate_price(row, price_type):
    """Calculates the requested price type from OHLC data."""
    if price_type == 'CLOSE':
        return row['Close']
    elif price_type == 'HL2':
        return (row['High'] + row['Low']) / 2
    elif price_type == 'HLC3':
        return (row['High'] + row['Low'] + row['Close']) / 3
    else:
        return row['Close']


class PCALLZScoreBacktest:
    # --- MODIFIED GRID DEFINITION ---
    def __init__(self, price_type='HL2'):
        self.pcall_data = None
        self.spy_data = None
        self.combined_data = None
        self.price_type = price_type
        self.csv_filename = None  # Store the filename here

        # New Lookback and Threshold Grids
        self.lookback_range = np.array([3,5,10,15,20,30], dtype=int)
        self.threshold_range = np.array([0.25,0.5,0.8,0.9,1.0,1.1,1.3, 1.5])

    def upload_pcall_csv(self):
        """
        Upload PCALL CSV file (works in Colab or local mode).
        Returns the filename if successful, otherwise None.
        """
        if self.csv_filename is not None:
            print(f"File already uploaded: **{self.csv_filename}**. Reusing this file.")
            return self.csv_filename

        try:
            # Check for Colab environment
            from google.colab import files
            print("Please upload your $PCALL CSV file:")
            uploaded = files.upload()
            filename = list(uploaded.keys())[0]
            print(f"Uploaded: {filename}")
            self.csv_filename = filename # Store the filename
            return filename
        except ImportError:
            # Not in Colab
            self.csv_filename = input("Enter CSV filename: ")
            return self.csv_filename
        except Exception as e:
            # Handle upload failure in Colab
            print(f"Colab upload failed: {e}. Falling back to manual input/default.")
            return None


    def load_pcall_data(self, csv_file, price_column='CLOSE'):
        """
        Load PCALL data from CSV, with robust date column handling and data cleaning.
        LOADS OHLC columns: Date (0), High (2), Low (4), Open (6), Close (8)
        """
        print(f"Loading $PCALL data from {csv_file}...")

        try:
            # Use tab delimiter as the data structure shows tab-separated values
            df = pd.read_csv(csv_file, header=None, sep='\t')
        except Exception as e:
            print(f"File reading failed: {e}")
            raise

        if df.empty:
            raise ValueError("CSV file is empty after initial read.")

        # Check if we have enough columns
        required_cols = 9
        if df.shape[1] < required_cols:
            print(f"Warning: Expected at least {required_cols} columns, but only found {df.shape[1]}.")
            print("Attempting to read with comma delimiter...")
            try:
                df = pd.read_csv(csv_file, header=None, sep=',')
            except Exception:
                raise ValueError("Could not parse CSV with tab or comma delimiters")

        # --- COLUMN MAPPING FOR OHLC DATA ---
        # Based on structure: Date HIGH value LOW value OPEN value CLOSE value
        # 0-indexed: Date=0, High=2, Low=4, Open=6, Close=8
        DATE_COL = 0
        HIGH_COL = 2
        LOW_COL = 4
        OPEN_COL = 6
        CLOSE_COL = 8

        try:
            # Select and rename the OHLC columns
            df = df[[DATE_COL, HIGH_COL, LOW_COL, OPEN_COL, CLOSE_COL]].copy()
            df.columns = ['DATE', 'High', 'Low', 'Open', 'Close']

        except KeyError as e:
            print(f"Error: Column selection failed: {e}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame head:\n{df.head()}")
            raise

        # --- DATA CLEANING AND CONVERSION ---
        # Convert OHLC columns to numeric
        for col in ['High', 'Low', 'Open', 'Close']:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- DATE PROCESSING ---
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df.dropna(subset=['DATE'], inplace=True)
        except Exception:
            print(f"Date conversion failed. Sample values: {df['DATE'].head().tolist()}")
            raise ValueError("Failed to convert date column to datetime objects.")

        # Set index and drop rows with any NaN in OHLC
        df = df.set_index('DATE').dropna().copy()

        # --- CHECK FOR EMPTY DATAFRAME ---
        if df.empty:
            print("\n" + "="*80)
            print("❌ FATAL DATA ERROR: The dataset is EMPTY after cleaning!")
            print("="*80)
            raise IndexError("Processed dataset is empty (size 0). Cannot proceed.")

        # Store the full OHLC dataframe
        self.pcall_data = df
        print(f"✓ Loaded {len(df)} days of $PCALL OHLC data. Range: {df.index[0].date()} to {df.index[-1].date()}")
        return df

    def download_spy_data(self):
        """Download SPY data matching PCALL date range"""
        if self.pcall_data is None:
            raise ValueError("PCALL data not loaded")

        print("Downloading SPY data...")
        start_date = self.pcall_data.index[0]
        end_date = self.pcall_data.index[-1]

        try:
            spy = yf.download('SPY', start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        except Exception as e:
            print(f"yfinance download failed: {e}. Cannot proceed with SPY data.")
            raise

        if hasattr(spy.columns, 'levels'):
            spy.columns = spy.columns.get_level_values(0)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)

        self.spy_data = spy
        print(f"✓ Downloaded {len(spy)} days of SPY data.")
        return spy

    def combine_data(self):
        """
        Combine PCALL and SPY data, and calculate PCALL price based on price_type.
        SPY uses CLOSE only.
        """
        if self.pcall_data is None or self.spy_data is None:
            raise ValueError("Both PCALL and SPY data must be loaded first")

        # Start with PCALL OHLC data
        combined = self.pcall_data.copy()

        # Calculate the PCALL price based on price_type
        combined['pcall_price'] = combined.apply(lambda row: calculate_price(row, self.price_type), axis=1)

        # Merge with SPY data
        combined = combined.merge(self.spy_data, left_index=True, right_index=True, how='inner', suffixes=('_pcall', '_spy'))

        # Use SPY Close only
        combined['spy_close'] = combined['Close_spy']

        self.combined_data = combined.dropna()

        print(f"✓ Combined data: {len(self.combined_data)} trading days.")
        print(f"Using PCALL {self.price_type} price and SPY Close.")

        return self.combined_data

    def calculate_zscore(self, data_slice, lookback_days):
        """Calculate rolling z-score for PCALL on a specific data slice"""

        lookback_int = int(lookback_days)
        pcall = data_slice['pcall_price']

        # Check to ensure the series is numeric before calculation
        if not pd.api.types.is_numeric_dtype(pcall):
            raise TypeError("PCALL data is not numeric. Check data cleaning steps.")

        rolling_mean = pcall.rolling(window=lookback_int, min_periods=lookback_int).mean()
        rolling_std = pcall.rolling(window=lookback_int, min_periods=lookback_int).std()

        rolling_std[rolling_std.fillna(0) == 0] = np.nan

        zscore = (pcall - rolling_mean) / rolling_std
        return zscore

    def generate_signals(self, data_slice, lookback_days, threshold=1.5):
        """Generate signals when z-score drops below threshold on a specific data slice"""
        zscore = self.calculate_zscore(data_slice, lookback_days)

        signals = pd.DataFrame(index=data_slice.index)
        signals['zscore'] = zscore
        signals['zscore_prev'] = zscore.shift(1)
        signals['signal'] = False

        # Signal fires when Z-score drops below the threshold
        cross_signal = (signals['zscore_prev'] > threshold) & (signals['zscore'] <= threshold)
        signals['signal'] = cross_signal

        return signals

    def calculate_forward_performance(self, signals, data_slice, forward_days=10):
        """
        Calculate performance metrics for forward-looking period on a specific data slice.
        Uses SPY Close only.
        """
        data = data_slice
        signal_dates = signals[signals['signal']].index
        price_column = 'spy_close'

        returns = []
        winning_signals = 0

        for signal_date in signal_dates:
            try:
                entry_price = data.loc[signal_date, price_column]
                future_dates = data.loc[signal_date:].index[1:]

                if len(future_dates) >= forward_days:
                    exit_date = future_dates[forward_days - 1]
                    exit_price = data.loc[exit_date, price_column]
                    forward_return = (exit_price / entry_price) - 1
                    returns.append(forward_return)
                    if exit_price > entry_price:
                        winning_signals += 1
            except Exception:
                continue

        total_valid_signals = len(returns)

        metrics = {
            'win_rate': winning_signals / total_valid_signals if total_valid_signals > 0 else 0,
            'total_signals': total_valid_signals,
            'avg_return': np.mean(returns) if returns else 0,
            'median_return': np.median(returns) if returns else 0,
            'total_return': np.prod([1 + r for r in returns]) - 1 if returns else 0,
            'std_return': np.std(returns) if returns else 0,
        }

        return metrics

    def run_optimization_grid(self, data_slice, forward_days_list):
        """
        Runs the grid on a specific data slice (In-Sample data).
        The scoring logic is set to target Avg Return.
        Returns a DataFrame of results for that slice, sorted by score.
        """
        lookback_range = self.lookback_range
        threshold_range = self.threshold_range

        results = []

        for forward_days in forward_days_list:
            for lookback in lookback_range:
                lookback_int = int(lookback)
                for threshold in threshold_range:

                    signals = self.generate_signals(data_slice, lookback_int, threshold)
                    metrics = self.calculate_forward_performance(signals, data_slice, forward_days)

                    score = -999 # Default score for invalid tests

                    # Only consider if there are enough signals for a valid test
                    if metrics['total_signals'] >= 5:

                        # --- OPTIMIZATION TARGET: PURE AVG RETURN ---
                        score = metrics['avg_return']
                        # --------------------------------------------

                        results.append({
                            'lookback': lookback_int,
                            'threshold': threshold,
                            'forward_days': forward_days,
                            'win_rate': metrics['win_rate'],
                            'total_signals': metrics['total_signals'],
                            'avg_return': metrics['avg_return'],
                            'median_return': metrics['median_return'],
                            'total_return': metrics['total_return'],
                            'std_return': metrics['std_return'],
                            'score': score,
                        })

        results_df = pd.DataFrame(results)
        return results_df.sort_values('score', ascending=False) if not results_df.empty else pd.DataFrame()

    def print_full_grid_results(self, results_df, fold_number):
        """Prints the results of the entire grid for a given fold."""
        if results_df.empty:
            print(f"| No valid combinations found for In-Sample Optimization in Fold {fold_number}.")
            return

        print("\n" + "="*90)
        print(f"| 📊 FULL IN-SAMPLE OPTIMIZATION GRID RESULTS - FOLD {fold_number} (Optimizing for Avg Return) |")
        print("="*90)

        # Select and format columns for display
        display_df = results_df.copy()
        display_df['Avg Ret'] = (display_df['avg_return'] * 100).map('{:.2f}%'.format)
        display_df['Std Dev'] = (display_df['std_return'] * 100).map('{:.2f}%'.format)
        display_df['Win Rate'] = (display_df['win_rate'] * 100).map('{:.1f}%'.format)
        display_df['Total Ret'] = (display_df['total_return'] * 100).map('{:.1f}%'.format)

        # Keep only essential columns and rename
        display_df = display_df[['lookback', 'threshold', 'forward_days', 'score',
                                 'Win Rate', 'Avg Ret', 'Std Dev', 'Total Ret', 'total_signals']]
        display_df.columns = ['L', 'T', 'H', 'Score (Avg Ret)',
                              'WR', 'Avg Ret', 'Std Dev', 'Tot Ret', 'Signals']

        # Print the top 10 rows for easy viewing (entire DataFrame if small)
        print(display_df.head(min(len(display_df), 15)).to_markdown(index=False, floatfmt=(".0f", ".2f", ".0f", ".4f")))

        if len(display_df) > 15:
             print(f"\n... (Showing top 15 of {len(display_df)} total combinations) ...")

        print("="*90)


    def cross_validate_parameters(self, forward_days_list, K=5):
        """
        Perform time-series K-Fold Cross-Validation.
        """
        total_tests_per_fold = len(self.lookback_range) * len(self.threshold_range) * len(forward_days_list)
        print("\n" + "~"*100)
        print(f"🏁 RUNNING K={K}-FOLD CV for **{self.price_type}** (H Range: {forward_days_list}) (Total Grid: {total_tests_per_fold} tests per fold)")
        print(f"🎯 Optimization Target: **Average Return**")
        print("~"*100)

        data = self.combined_data
        data_length = len(data)
        fold_size = data_length // K
        oos_results = []

        for i in range(K):
            sys.stdout.write(f"\n--- Processing Fold {i+1}/{K} ---")

            # Define OOS Test Set (Fold i)
            oos_start_idx = i * fold_size
            oos_end_idx = (i + 1) * fold_size if i < K - 1 else data_length
            oos_data = data.iloc[oos_start_idx:oos_end_idx]

            # Define IS Optimization Set (All data BEFORE the OOS segment)
            is_data = data.iloc[:oos_start_idx]

            min_data_required = self.lookback_range.max()

            if len(is_data) < min_data_required or len(oos_data) < min(forward_days_list):
                sys.stdout.write(f" (Skipping: Insufficient data: IS={len(is_data)}, Min Req={min_data_required})\n")
                continue

            sys.stdout.write(f" (IS: {len(is_data)} days, OOS: {len(oos_data)} days)\n")

            # 3. Find the Optimal Parameters (P*) on the IS Data
            is_results_df = self.run_optimization_grid(is_data, forward_days_list)

            # --- NEW: Print the results of the entire optimization grid ---
            self.print_full_grid_results(is_results_df, i + 1)
            # -------------------------------------------------------------

            if is_results_df.empty:
                print(f"No valid signals found in IS data for Fold {i+1}. Skipping OOS test.")
                continue

            best_params = is_results_df.iloc[0]
            L_star = int(best_params['lookback'])
            T_star = best_params['threshold']
            H_star = int(best_params['forward_days'])

            # Append Trailing Context for Z-Score Calculation
            oos_start_date_index = data.index.get_loc(oos_data.iloc[0].name)
            context_start_index = max(0, oos_start_date_index - L_star)
            oos_test_block = data.iloc[context_start_index:oos_end_idx]

            # 4. Validate P* on the OOS Test Data
            oos_signals = self.generate_signals(oos_test_block, L_star, T_star)

            oos_test_signals = oos_signals.loc[oos_data.index]

            oos_metrics = self.calculate_forward_performance(oos_test_signals, oos_data, H_star)

            oos_results.append({
                'Fold': i + 1,
                'L_star': L_star,
                'T_star': T_star,
                'H_star': H_star,
                'OOS_WinRate': oos_metrics['win_rate'],
                'OOS_AvgReturn': oos_metrics['avg_return'],
                'OOS_TotalSignals': oos_metrics['total_signals'],
                'OOS_TotalReturn': oos_metrics['total_return'],
                'OOS_StdReturn': oos_metrics['std_return'],
                'OOS_Start': oos_data.iloc[0].name.date(),
                'OOS_End': oos_data.iloc[-1].name.date(),
                'IS_Score': best_params['score'],
            })

            print(f"Optimal IS Params: L={L_star}, T={T_star:.2f}, H={H_star} (OOS WR: {oos_metrics['win_rate']:.1%}, AR: {oos_metrics['avg_return']:.2%}, Signals: {oos_metrics['total_signals']:.0f})")

        return pd.DataFrame(oos_results)


    def print_cross_validation_results(self, oos_results_df, forward_days_range_name):
        """Prints the detailed cross-validation results with all required metrics."""

        print("\n" + "="*120)
        print(f"📈 CV ROBUSTNESS SUMMARY (Price Type: **{self.price_type}**) | H-Optimization Range: **{forward_days_range_name}**")
        print("="*120)

        if oos_results_df.empty:
            print(f"No valid out-of-sample test results were generated for this H-range.")
            return

        # --- Table 1: Detailed Fold Results ---
        print("\n## 1. Out-of-Sample Performance by Fold")
        print("-----------------------------------------------------------------------------------------------------------------------------------------")

        sorted_oos = oos_results_df.sort_values('OOS_TotalReturn', ascending=False)
        top_folds = sorted_oos.head(len(oos_results_df))

        print(f"{'Fold':>4} {'Test Period':>25} {'L*':>4} {'T*':>4} {'H*':>4} {'IS Score':>8} {'WR':>8} {'Avg Ret':>8} {'Std Dev':>8} {'Tot Ret':>8} {'Signals':>8}")
        print("-" * 120)
        for i, row in top_folds.iterrows():
            print(f"{row['Fold']:4.0f} {str(row['OOS_Start'])} to {str(row['OOS_End']):10} "
                  f"{row['L_star']:4.0f} {row['T_star']:4.2f} {row['H_star']:4.0f} "
                  f"{row['IS_Score']:8.3f} "
                  f"{row['OOS_WinRate']:7.1%} {row['OOS_AvgReturn']:7.2%} {row['OOS_StdReturn']:7.2%} {row['OOS_TotalReturn']:7.1%} "
                  f"{row['OOS_TotalSignals']:8.0f}")

        # --- Table 2: Robustness Metrics (Mean/Std) ---
        print("\n## 2. Robustness and Consistency Metrics")
        print("--------------------------------------------------------------------------------")

        win_rate_mean = oos_results_df['OOS_WinRate'].mean()
        win_rate_std = oos_results_df['OOS_WinRate'].std()
        return_mean = oos_results_df['OOS_AvgReturn'].mean()
        return_std = oos_results_df['OOS_AvgReturn'].std()
        total_signals_mean = oos_results_df['OOS_TotalSignals'].mean()

        print(f"| Metric | Average (Mean) | Std Deviation (Risk) | Max | Min |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        print(f"| **Win Rate** | **{win_rate_mean:.1%}** | {win_rate_std:.2%} | {oos_results_df['OOS_WinRate'].max():.1%} | {oos_results_df['OOS_WinRate'].min():.1%} |")
        print(f"| **Avg Return** | **{return_mean:.2%}** | {return_std:.2%} | {oos_results_df['OOS_AvgReturn'].max():.2%} | {oos_results_df['OOS_AvgReturn'].min():.2%} |")
        print(f"| **Avg Signals** | {total_signals_mean:.1f} | - | {oos_results_df['OOS_TotalSignals'].max():.0f} | {oos_results_df['OOS_TotalSignals'].min():.0f} |")

        # --- Table 3: Parameter Stability Matrix ---
        print("\n## 3. Optimal Parameter Stability Matrix (L*, T*, H*)")
        print("--------------------------------------------------------------------------------")

        print(f"{'Fold':>4} | {'Test Period':>25} | {'Optimal L*':>10} | {'Optimal T*':>10} | {'Optimal H*':>10}")
        print("-" * 70)
        for i, row in oos_results_df.iterrows():
            print(f"{row['Fold']:4.0f} | {str(row['OOS_Start'])} to {str(row['OOS_End']):10} | {row['L_star']:10.0f} | {row['T_star']:10.2f} | {row['H_star']:10.0f}")

        # Parameter Choice Frequency Summary
        L_star_counts = oos_results_df['L_star'].value_counts(normalize=True).round(3)
        T_star_counts = oos_results_df['T_star'].value_counts(normalize=True).round(3)
        H_star_counts = oos_results_df['H_star'].value_counts(normalize=True).round(3)

        print("\nFrequency of Chosen Parameters:")
        print(f"    - Lookbacks Chosen (L): {L_star_counts.to_dict()}")
        print(f"    - Thresholds Chosen (T): {T_star_counts.to_dict()}")
        print(f"    - Holding Periods Chosen (H): {H_star_counts.to_dict()}")

        print("\n" + "="*120)
        print(f"END CROSS-VALIDATION ANALYSIS for H-Range: {forward_days_range_name}")
        print("="*120)

        return oos_results_df

    def get_forward_days_list(self, h_input):
        """
        Calculates the H, H-1, H+1 list from a single H input.
        """
        h = int(h_input)
        full_forward_days_list = set()

        if h > 0:
            full_forward_days_list.add(h)

        if h - 1 > 0:
            full_forward_days_list.add(h - 1)

        full_forward_days_list.add(h + 1)

        return sorted(list(full_forward_days_list))

# =============================================================================
# Execution Block - Test Multiple PCALL Price Types
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🎯 $PCALL Z-SCORE K-FOLD ROBUSTNESS TESTER")
    print("="*80)
    print("Testing PCALL Price Types: CLOSE, HL2, HLC3")
    print("SPY Price: CLOSE only")
    print("Method: Time-Series K-Fold Cross-Validation")
    print("")

    # Define H inputs to test
    h_inputs_to_test = [9, 10, 11]

    # Define PCALL price types to test
    price_types_to_test = ['CLOSE', 'HL2', 'HLC3']

    all_results = {}

    # Load data once (will be reused for all price types)
    base_backtest = PCALLZScoreBacktest(price_type='CLOSE')
    csv_file_to_load = base_backtest.upload_pcall_csv()

    if csv_file_to_load is None:
        csv_file_to_load = "StrategyReports_$PCALL_12925.csv"

    try:
        base_backtest.load_pcall_data(csv_file_to_load)
        base_backtest.download_spy_data()
    except Exception as e:
        print(f"\nFATAL ERROR during data loading: {e}")
        sys.exit(1)

    # Test each PCALL price type
    for price_type in price_types_to_test:
        print("\n" + "#"*120)
        print(f"🔬 TESTING PCALL PRICE TYPE: **{price_type}**")
        print("#"*120)

        # Create backtest instance for this price type
        backtest = PCALLZScoreBacktest(price_type=price_type)
        backtest.pcall_data = base_backtest.pcall_data
        backtest.spy_data = base_backtest.spy_data
        backtest.csv_filename = base_backtest.csv_filename

        # Combine data with the specific price type
        backtest.combine_data()

        all_results[price_type] = []

        # Test each H input
        for h_target in h_inputs_to_test:
            forward_days_list = backtest.get_forward_days_list(h_target)
            h_range_name = f"H={h_target} (Testing: {forward_days_list})"

            print("\n" + "~"*100)
            print(f"📊 H-RANGE: {h_range_name}")
            print("~"*100)

            # Run CV
            oos_results = backtest.cross_validate_parameters(
                forward_days_list=forward_days_list,
                K=5
            )

            # Print detailed results
            oos_results = backtest.print_cross_validation_results(oos_results, h_range_name)

            if not oos_results.empty:
                all_results[price_type].append({
                    'H_Target': h_target,
                    'H_Range': h_range_name,
                    'Mean_WinRate': oos_results['OOS_WinRate'].mean(),
                    'Mean_AvgReturn': oos_results['OOS_AvgReturn'].mean(),
                    'Mean_TotalSignals': oos_results['OOS_TotalSignals'].mean(),
                    'Std_AvgReturn': oos_results['OOS_AvgReturn'].std(),
                })

    # --- Final Comparison Across All Price Types ---
    print("\n" + "!"*120)
    print("🌟 FINAL COMPARISON: PCALL Price Types Performance")
    print("!"*120)

    for price_type in price_types_to_test:
        if all_results[price_type]:
            print(f"\n### {price_type} Results:")
            print(f"| H Target | Mean Win Rate | Mean Avg Return | Std Avg Return | Avg Signals |")
            print(f"| :---: | :---: | :---: | :---: | :---: |")

            for result in all_results[price_type]:
                print(f"| {result['H_Target']} | {result['Mean_WinRate']:.1%} | "
                      f"**{result['Mean_AvgReturn']:.2%}** | {result['Std_AvgReturn']:.2%} | "
                      f"{result['Mean_TotalSignals']:.1f} |")

    print("!"*120)
    print("✓ Analysis complete!")
