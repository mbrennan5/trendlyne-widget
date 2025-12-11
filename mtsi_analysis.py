import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import sys

warnings.filterwarnings('ignore')

class MTSIZScoreBacktest:
    """
    Modified True Strength Index (MTSI) with Z-Score Analysis
    Tests long and short signals based on MTSI z-score reversals
    """

    def __init__(self, long_length=3, short_length=2):
        self.spy_data = None
        self.mtsi_data = None
        self.combined_data = None

        # MTSI parameters (can be fixed or made variable)
        self.long_length = long_length
        self.short_length = short_length

        # Grid search parameters
        self.zscore_lookback_range = np.array([20, 50, 100, 150, 200], dtype=int)
        self.zscore_threshold_range = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

    def download_spy_data(self, start_date='2015-01-01', end_date=None):
        """Download SPY data with OHLC and Volume for VWAP calculation"""
        print(f"Downloading SPY data from {start_date}...")

        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        try:
            # Download with 1-minute data is not available via yfinance for long periods
            # We'll use daily data and approximate intraday VWAP using typical price
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        except Exception as e:
            print(f"yfinance download failed: {e}")
            raise

        if hasattr(spy.columns, 'levels'):
            spy.columns = spy.columns.get_level_values(0)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)

        # Calculate VWAP approximation for daily data
        # VWAP = (Volume * Typical Price) / Total Volume
        spy['TypicalPrice'] = (spy['High'] + spy['Low'] + spy['Close']) / 3
        spy['VWAP'] = spy['TypicalPrice']  # For daily data, using typical price as VWAP proxy

        self.spy_data = spy
        print(f"✓ Downloaded {len(spy)} days of SPY data from {spy.index[0].date()} to {spy.index[-1].date()}")
        return spy

    def calculate_ema(self, series, length):
        """Calculate exponential moving average"""
        return series.ewm(span=length, adjust=False).mean()

    def calculate_mtsi(self, data):
        """
        Calculate Modified True Strength Index (MTSI)
        Based on the ThinkScript implementation
        """
        df = data.copy()

        # Step 1: Calculate diff = Log(close / VWAP)
        df['diff'] = np.log(df['Close'] / df['VWAP'])

        # Step 2: Double smooth the absolute difference
        abs_diff = df['diff'].abs()
        abs_diff_smooth1 = self.calculate_ema(abs_diff, self.long_length)
        abs_diff_smooth2 = self.calculate_ema(abs_diff_smooth1, self.short_length)

        # Step 3: Double smooth the difference
        diff_smooth1 = self.calculate_ema(df['diff'], self.long_length)
        diff_smooth2 = self.calculate_ema(diff_smooth1, self.short_length)

        # Step 4: Calculate MTSI
        df['MTSI'] = np.where(
            abs_diff_smooth2 == 0,
            0,
            100 * diff_smooth2 / abs_diff_smooth2
        )

        return df

    def calculate_mtsi_zscore(self, data, lookback):
        """
        Calculate z-score of MTSI over a given lookback period
        """
        df = data.copy()
        lookback_int = int(lookback)

        # Calculate rolling mean and std of MTSI
        rolling_mean = df['MTSI'].rolling(window=lookback_int, min_periods=lookback_int).mean()
        rolling_std = df['MTSI'].rolling(window=lookback_int, min_periods=lookback_int).std()

        # Avoid division by zero
        rolling_std[rolling_std.fillna(0) == 0] = np.nan

        # Calculate z-score
        df['MTSI_ZScore'] = (df['MTSI'] - rolling_mean) / rolling_std

        return df

    def generate_signals(self, data, zscore_lookback, zscore_threshold):
        """
        Generate long and short signals based on MTSI z-score reversals

        Long signal (bull):
        - zscore[0] > zscore[1] AND zscore[1] < zscore[2] AND zscore[1] < -threshold
        - This identifies a V-shaped bottom where zscore[1] is the low point

        Short signal (bear):
        - zscore[0] < zscore[1] AND zscore[1] > zscore[2] AND zscore[1] > +threshold
        - This identifies an inverted-V top where zscore[1] is the high point
        """
        # First calculate MTSI z-score
        data_with_zscore = self.calculate_mtsi_zscore(data, zscore_lookback)

        signals = pd.DataFrame(index=data_with_zscore.index)
        signals['MTSI_ZScore'] = data_with_zscore['MTSI_ZScore']
        signals['zscore_prev1'] = signals['MTSI_ZScore'].shift(1)
        signals['zscore_prev2'] = signals['MTSI_ZScore'].shift(2)

        # Long signal: V-shaped bottom below negative threshold
        signals['long_signal'] = (
            (signals['MTSI_ZScore'] > signals['zscore_prev1']) &
            (signals['zscore_prev1'] < signals['zscore_prev2']) &
            (signals['zscore_prev1'] < -zscore_threshold)
        )

        # Short signal: Inverted-V top above positive threshold
        signals['short_signal'] = (
            (signals['MTSI_ZScore'] < signals['zscore_prev1']) &
            (signals['zscore_prev1'] > signals['zscore_prev2']) &
            (signals['zscore_prev1'] > zscore_threshold)
        )

        return signals

    def calculate_forward_performance(self, signals, data, forward_days, signal_type='long'):
        """
        Calculate forward performance for long or short signals

        signal_type: 'long' or 'short'
        """
        signal_column = f'{signal_type}_signal'
        signal_dates = signals[signals[signal_column]].index

        returns = []
        winning_signals = 0

        for signal_date in signal_dates:
            try:
                entry_price = data.loc[signal_date, 'Close']
                future_dates = data.loc[signal_date:].index[1:]

                if len(future_dates) >= forward_days:
                    exit_date = future_dates[forward_days - 1]
                    exit_price = data.loc[exit_date, 'Close']

                    # Calculate return based on signal type
                    if signal_type == 'long':
                        forward_return = (exit_price / entry_price) - 1
                        if exit_price > entry_price:
                            winning_signals += 1
                    else:  # short
                        forward_return = (entry_price / exit_price) - 1
                        if entry_price > exit_price:
                            winning_signals += 1

                    returns.append(forward_return)
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

    def run_optimization_grid(self, data_slice, forward_days_list, signal_type='long'):
        """
        Run grid search over z-score lookback and threshold parameters
        Tests for a specific signal type (long or short)
        """
        results = []

        for forward_days in forward_days_list:
            for zscore_lookback in self.zscore_lookback_range:
                for zscore_threshold in self.zscore_threshold_range:

                    signals = self.generate_signals(data_slice, zscore_lookback, zscore_threshold)
                    metrics = self.calculate_forward_performance(
                        signals, data_slice, forward_days, signal_type
                    )

                    score = -999

                    # Only consider if there are enough signals
                    if metrics['total_signals'] >= 5:
                        score = metrics['avg_return']

                        results.append({
                            'signal_type': signal_type,
                            'zscore_lookback': zscore_lookback,
                            'zscore_threshold': zscore_threshold,
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

    def print_grid_results(self, results_df, fold_number, signal_type):
        """Print optimization grid results"""
        if results_df.empty:
            print(f"| No valid combinations found for {signal_type.upper()} signals in Fold {fold_number}.")
            return

        print("\n" + "="*100)
        print(f"| 📊 {signal_type.upper()} SIGNAL OPTIMIZATION - FOLD {fold_number} |")
        print("="*100)

        display_df = results_df.copy()
        display_df['Avg Ret'] = (display_df['avg_return'] * 100).map('{:.2f}%'.format)
        display_df['Std Dev'] = (display_df['std_return'] * 100).map('{:.2f}%'.format)
        display_df['Win Rate'] = (display_df['win_rate'] * 100).map('{:.1f}%'.format)
        display_df['Total Ret'] = (display_df['total_return'] * 100).map('{:.1f}%'.format)

        display_df = display_df[['zscore_lookback', 'zscore_threshold', 'forward_days', 'score',
                                 'Win Rate', 'Avg Ret', 'Std Dev', 'Total Ret', 'total_signals']]
        display_df.columns = ['Z-LB', 'Z-Thresh', 'H', 'Score',
                              'WR', 'Avg Ret', 'Std Dev', 'Tot Ret', 'Signals']

        print(display_df.head(min(len(display_df), 15)).to_markdown(index=False, floatfmt=(".0f", ".2f", ".0f", ".4f")))

        if len(display_df) > 15:
            print(f"\n... (Showing top 15 of {len(display_df)} total combinations) ...")

        print("="*100)

    def cross_validate_parameters(self, forward_days_list, signal_type='long', K=5):
        """
        Perform time-series K-Fold Cross-Validation for MTSI signals
        """
        print("\n" + "~"*100)
        print(f"🏁 RUNNING K={K}-FOLD CV for MTSI {signal_type.upper()} SIGNALS")
        print(f"🎯 H Range: {forward_days_list}")
        print(f"🎯 Z-Score Lookbacks: {self.zscore_lookback_range.tolist()}")
        print(f"🎯 Z-Score Thresholds: {self.zscore_threshold_range.tolist()}")
        print("~"*100)

        data = self.mtsi_data
        data_length = len(data)
        fold_size = data_length // K
        oos_results = []

        for i in range(K):
            sys.stdout.write(f"\n--- Processing Fold {i+1}/{K} ---")

            # Define OOS Test Set
            oos_start_idx = i * fold_size
            oos_end_idx = (i + 1) * fold_size if i < K - 1 else data_length
            oos_data = data.iloc[oos_start_idx:oos_end_idx]

            # Define IS Optimization Set (all data before OOS)
            is_data = data.iloc[:oos_start_idx]

            min_data_required = self.zscore_lookback_range.max()

            if len(is_data) < min_data_required or len(oos_data) < min(forward_days_list):
                sys.stdout.write(f" (Skipping: Insufficient data)\n")
                continue

            sys.stdout.write(f" (IS: {len(is_data)} days, OOS: {len(oos_data)} days)\n")

            # Optimize on IS data
            is_results_df = self.run_optimization_grid(is_data, forward_days_list, signal_type)
            self.print_grid_results(is_results_df, i + 1, signal_type)

            if is_results_df.empty:
                print(f"No valid signals found in IS data for Fold {i+1}. Skipping OOS test.")
                continue

            # Get best parameters
            best_params = is_results_df.iloc[0]
            ZLB_star = int(best_params['zscore_lookback'])
            ZT_star = best_params['zscore_threshold']
            H_star = int(best_params['forward_days'])

            # Test on OOS data with context for z-score calculation
            oos_start_date_index = data.index.get_loc(oos_data.iloc[0].name)
            context_start_index = max(0, oos_start_date_index - ZLB_star)
            oos_test_block = data.iloc[context_start_index:oos_end_idx]

            oos_signals = self.generate_signals(oos_test_block, ZLB_star, ZT_star)
            oos_test_signals = oos_signals.loc[oos_data.index]

            oos_metrics = self.calculate_forward_performance(
                oos_test_signals, oos_data, H_star, signal_type
            )

            oos_results.append({
                'Fold': i + 1,
                'Signal_Type': signal_type,
                'ZLB_star': ZLB_star,
                'ZT_star': ZT_star,
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

            print(f"Optimal Params: Z-LB={ZLB_star}, Z-Thresh={ZT_star:.2f}, H={H_star} "
                  f"(OOS WR: {oos_metrics['win_rate']:.1%}, AR: {oos_metrics['avg_return']:.2%}, "
                  f"Signals: {oos_metrics['total_signals']:.0f})")

        return pd.DataFrame(oos_results)

    def print_cv_results(self, oos_results_df, signal_type, h_range_name):
        """Print cross-validation summary results"""
        print("\n" + "="*120)
        print(f"📈 CV SUMMARY - MTSI {signal_type.upper()} SIGNALS | H-Range: {h_range_name}")
        print("="*120)

        if oos_results_df.empty:
            print(f"No valid out-of-sample results.")
            return oos_results_df

        # Table 1: Fold Results
        print("\n## 1. Out-of-Sample Performance by Fold")
        print("-" * 120)

        sorted_oos = oos_results_df.sort_values('OOS_AvgReturn', ascending=False)

        print(f"{'Fold':>4} {'Test Period':>25} {'Z-LB*':>6} {'Z-Th*':>6} {'H*':>4} "
              f"{'IS Score':>8} {'WR':>8} {'Avg Ret':>8} {'Std Dev':>8} {'Signals':>8}")
        print("-" * 120)

        for i, row in sorted_oos.iterrows():
            print(f"{row['Fold']:4.0f} {str(row['OOS_Start'])} to {str(row['OOS_End']):10} "
                  f"{row['ZLB_star']:6.0f} {row['ZT_star']:6.2f} {row['H_star']:4.0f} "
                  f"{row['IS_Score']:8.3f} "
                  f"{row['OOS_WinRate']:7.1%} {row['OOS_AvgReturn']:7.2%} "
                  f"{row['OOS_StdReturn']:7.2%} {row['OOS_TotalSignals']:8.0f}")

        # Table 2: Robustness Metrics
        print("\n## 2. Robustness Metrics")
        print("-" * 80)

        print(f"| Metric | Mean | Std Dev | Max | Min |")
        print(f"| :--- | :--- | :--- | :--- | :--- |")
        print(f"| **Win Rate** | {oos_results_df['OOS_WinRate'].mean():.1%} | "
              f"{oos_results_df['OOS_WinRate'].std():.2%} | "
              f"{oos_results_df['OOS_WinRate'].max():.1%} | "
              f"{oos_results_df['OOS_WinRate'].min():.1%} |")
        print(f"| **Avg Return** | {oos_results_df['OOS_AvgReturn'].mean():.2%} | "
              f"{oos_results_df['OOS_AvgReturn'].std():.2%} | "
              f"{oos_results_df['OOS_AvgReturn'].max():.2%} | "
              f"{oos_results_df['OOS_AvgReturn'].min():.2%} |")
        print(f"| **Avg Signals** | {oos_results_df['OOS_TotalSignals'].mean():.1f} | - | "
              f"{oos_results_df['OOS_TotalSignals'].max():.0f} | "
              f"{oos_results_df['OOS_TotalSignals'].min():.0f} |")

        # Parameter stability
        print("\n## 3. Parameter Frequency")
        print("-" * 80)
        zlb_counts = oos_results_df['ZLB_star'].value_counts(normalize=True).round(3)
        zt_counts = oos_results_df['ZT_star'].value_counts(normalize=True).round(3)
        h_counts = oos_results_df['H_star'].value_counts(normalize=True).round(3)

        print(f"    - Z-Score Lookbacks: {zlb_counts.to_dict()}")
        print(f"    - Z-Score Thresholds: {zt_counts.to_dict()}")
        print(f"    - Holding Periods: {h_counts.to_dict()}")

        print("="*120)
        return oos_results_df

    def get_forward_days_list(self, h_input):
        """Calculate H, H-1, H+1 list"""
        h = int(h_input)
        full_forward_days_list = set()

        if h > 0:
            full_forward_days_list.add(h)
        if h - 1 > 0:
            full_forward_days_list.add(h - 1)
        full_forward_days_list.add(h + 1)

        return sorted(list(full_forward_days_list))


# =============================================================================
# Execution Block
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🎯 MTSI Z-SCORE BACKTEST - LONG SIGNALS")
    print("="*80)
    print("Testing LONG signals based on MTSI z-score reversals")
    print("Method: Time-Series K-Fold Cross-Validation")
    print("")

    # Initialize backtest
    backtest = MTSIZScoreBacktest(long_length=3, short_length=2)

    # Download and prepare data
    backtest.download_spy_data(start_date='2015-01-01')

    print("\nCalculating MTSI indicator...")
    backtest.mtsi_data = backtest.calculate_mtsi(backtest.spy_data)
    print(f"✓ MTSI calculated for {len(backtest.mtsi_data)} days")

    # Test parameters
    h_inputs_to_test = [3, 5, 7, 9, 11]
    signal_types_to_test = ['long']  # Testing LONG signals only

    all_results = {}

    # Test each signal type
    for signal_type in signal_types_to_test:
        print("\n" + "#"*120)
        print(f"🔬 TESTING {signal_type.upper()} SIGNALS")
        print("#"*120)

        all_results[signal_type] = []

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
                signal_type=signal_type,
                K=5
            )

            # Print results
            oos_results = backtest.print_cv_results(oos_results, signal_type, h_range_name)

            if not oos_results.empty:
                all_results[signal_type].append({
                    'H_Target': h_target,
                    'H_Range': h_range_name,
                    'Mean_WinRate': oos_results['OOS_WinRate'].mean(),
                    'Mean_AvgReturn': oos_results['OOS_AvgReturn'].mean(),
                    'Mean_TotalSignals': oos_results['OOS_TotalSignals'].mean(),
                    'Std_AvgReturn': oos_results['OOS_AvgReturn'].std(),
                })

    # Final comparison
    print("\n" + "!"*120)
    print("🌟 FINAL RESULTS: LONG SIGNALS ACROSS HOLDING PERIODS")
    print("!"*120)

    for signal_type in signal_types_to_test:
        if all_results[signal_type]:
            print(f"\n### {signal_type.upper()} Signal Results:")
            print(f"| H Target | Mean Win Rate | Mean Avg Return | Std Avg Return | Avg Signals |")
            print(f"| :---: | :---: | :---: | :---: | :---: |")

            for result in all_results[signal_type]:
                print(f"| {result['H_Target']} | {result['Mean_WinRate']:.1%} | "
                      f"**{result['Mean_AvgReturn']:.2%}** | {result['Std_AvgReturn']:.2%} | "
                      f"{result['Mean_TotalSignals']:.1f} |")

    print("\n" + "!"*120)
    print("✓ MTSI Analysis Complete!")
    print("!"*120)
