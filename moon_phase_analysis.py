import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import sys
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class MoonPhaseBacktest:
    """
    Moon Phase Analysis - Test if Full Moon and New Moon dates
    have predictive power for SPY forward returns
    """

    def __init__(self):
        self.spy_data = None
        self.moon_data = None

        # Moon phase constants (from ThinkScript code)
        self.SYNOD_MONTH = 29.530588  # Synodic month length in days
        self.SYNOD_FULL_START = 105.2330205  # Base full moon offset
        self.SYNOD_NEW_START = 120  # Base new moon offset
        self.BASE_DAY_COUNT = 25569  # Days between 1/1/1900 and 12/31/1969

        # Test parameters
        self.holding_period_range = np.array([1, 2, 3, 5, 7, 10, 14, 21], dtype=int)

    def download_spy_data(self, start_date='2000-01-01', end_date=None):
        """Download SPY data"""
        print(f"Downloading SPY data from {start_date}...")

        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        try:
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        except Exception as e:
            print(f"yfinance download failed: {e}")
            raise

        if hasattr(spy.columns, 'levels'):
            spy.columns = spy.columns.get_level_values(0)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)

        self.spy_data = spy
        print(f"✓ Downloaded {len(spy)} days of SPY data from {spy.index[0].date()} to {spy.index[-1].date()}")
        return spy

    def days_from_epoch(self, date):
        """Calculate days from Unix epoch (1/1/1970)"""
        epoch = datetime(1970, 1, 1)
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        delta = date - epoch
        return delta.days

    def calculate_moon_phases(self):
        """
        Calculate Full Moon and New Moon dates for the SPY data range
        Based on the astronomical synodic month calculation
        """
        print("Calculating moon phases...")

        moon_events = []

        # Get date range from SPY data
        start_date = self.spy_data.index[0]
        end_date = self.spy_data.index[-1]

        # Calculate for each day in the range
        current_date = start_date
        while current_date <= end_date:
            # Calculate serial date (Excel-style)
            days_from_1970 = self.days_from_epoch(current_date)
            serial_date = days_from_1970 + self.BASE_DAY_COUNT

            # Check if this date is a Full Moon
            # Using the synodic month cycle
            for index in range(1237, 2470):
                full_moon_serial = round(self.SYNOD_FULL_START + index * self.SYNOD_MONTH, 0)
                if abs(serial_date - full_moon_serial) < 0.5:
                    moon_events.append({
                        'date': current_date,
                        'phase': 'full',
                        'weekday': current_date.weekday()
                    })
                    break

            # Check if this date is a New Moon
            for index in range(1237, 2470):
                new_moon_serial = round(self.SYNOD_NEW_START + index * self.SYNOD_MONTH, 0)
                if abs(serial_date - new_moon_serial) < 0.5:
                    moon_events.append({
                        'date': current_date,
                        'phase': 'new',
                        'weekday': current_date.weekday()
                    })
                    break

            current_date += timedelta(days=1)

        moon_df = pd.DataFrame(moon_events)

        if not moon_df.empty:
            # Handle weekend adjustments (similar to ThinkScript)
            # If Saturday (5), move to Friday
            # If Sunday (6), move to Monday
            adjusted_dates = []
            for idx, row in moon_df.iterrows():
                adj_date = row['date']
                if row['weekday'] == 5:  # Saturday
                    adj_date = row['date'] - timedelta(days=1)
                elif row['weekday'] == 6:  # Sunday
                    adj_date = row['date'] + timedelta(days=1)
                adjusted_dates.append(adj_date)

            moon_df['adjusted_date'] = adjusted_dates

            # Remove duplicates that might occur from weekend adjustments
            moon_df = moon_df.drop_duplicates(subset=['adjusted_date', 'phase'])

        self.moon_data = moon_df
        print(f"✓ Calculated {len(moon_df)} moon phase events")

        if not moon_df.empty:
            full_count = len(moon_df[moon_df['phase'] == 'full'])
            new_count = len(moon_df[moon_df['phase'] == 'new'])
            print(f"  - Full Moons: {full_count}")
            print(f"  - New Moons: {new_count}")

        return moon_df

    def generate_moon_signals(self, moon_phase='full'):
        """
        Generate trading signals based on moon phase
        moon_phase: 'full' or 'new'

        Also calculates days until next opposite moon phase for dynamic holding
        """
        signals = pd.DataFrame(index=self.spy_data.index)
        signals['signal'] = False
        signals['days_to_opposite_phase'] = np.nan

        # Get moon dates for the specified phase
        moon_dates = self.moon_data[self.moon_data['phase'] == moon_phase]['adjusted_date'].values

        # Get opposite phase dates
        opposite_phase = 'new' if moon_phase == 'full' else 'full'
        opposite_dates = self.moon_data[self.moon_data['phase'] == opposite_phase]['adjusted_date'].values

        # Mark signal dates and calculate days to opposite phase
        for moon_date in moon_dates:
            # Find the closest trading day to the moon date
            closest_idx = self.spy_data.index.asof(pd.Timestamp(moon_date))
            if pd.notna(closest_idx) and closest_idx in signals.index:
                signals.loc[closest_idx, 'signal'] = True

                # Find next opposite phase date
                future_opposite_dates = opposite_dates[opposite_dates > moon_date]
                if len(future_opposite_dates) > 0:
                    next_opposite = pd.Timestamp(future_opposite_dates[0])
                    # Count trading days between moon phases
                    future_trading_days = self.spy_data.loc[closest_idx:].index
                    if next_opposite in future_trading_days:
                        days_between = len(self.spy_data.loc[closest_idx:next_opposite]) - 1
                        signals.loc[closest_idx, 'days_to_opposite_phase'] = days_between

        return signals

    def calculate_forward_performance(self, signals, holding_days):
        """
        Calculate forward performance metrics
        holding_days: int for fixed days, or 'until_opposite' for dynamic holding
        """
        signal_dates = signals[signals['signal']].index
        returns = []
        winning_signals = 0
        actual_holding_days_list = []

        for signal_date in signal_dates:
            try:
                entry_price = self.spy_data.loc[signal_date, 'Close']
                future_dates = self.spy_data.loc[signal_date:].index[1:]

                # Determine holding period
                if holding_days == 'until_opposite':
                    # Use dynamic holding until opposite moon phase
                    days_to_hold = signals.loc[signal_date, 'days_to_opposite_phase']
                    if pd.isna(days_to_hold) or days_to_hold <= 0:
                        continue
                    days_to_hold = int(days_to_hold)
                else:
                    # Use fixed holding period
                    days_to_hold = int(holding_days)

                if len(future_dates) >= days_to_hold:
                    exit_date = future_dates[days_to_hold - 1]
                    exit_price = self.spy_data.loc[exit_date, 'Close']
                    forward_return = (exit_price / entry_price) - 1
                    returns.append(forward_return)
                    actual_holding_days_list.append(days_to_hold)
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
            'avg_holding_days': np.mean(actual_holding_days_list) if actual_holding_days_list else 0,
        }

        return metrics

    def run_optimization_grid(self, data_slice, moon_phase='full'):
        """
        Test different holding periods for a given moon phase
        Includes both fixed holding periods and 'until_opposite' dynamic holding
        """
        results = []

        # Generate signals for this moon phase
        signals = self.generate_moon_signals(moon_phase)

        # Filter signals to only those in the data slice
        signals_slice = signals.loc[data_slice.index]

        # Test fixed holding periods
        for holding_days in self.holding_period_range:
            # Calculate metrics for this holding period
            metrics = self.calculate_forward_performance_slice(
                signals_slice, data_slice, holding_days
            )

            score = -999

            if metrics['total_signals'] >= 3:  # Lower threshold for moon phases
                score = metrics['avg_return']

                results.append({
                    'moon_phase': moon_phase,
                    'holding_days': holding_days,
                    'holding_type': 'fixed',
                    'win_rate': metrics['win_rate'],
                    'total_signals': metrics['total_signals'],
                    'avg_return': metrics['avg_return'],
                    'median_return': metrics['median_return'],
                    'total_return': metrics['total_return'],
                    'std_return': metrics['std_return'],
                    'avg_holding_days': holding_days,
                    'score': score,
                })

        # Test dynamic holding until opposite moon phase
        metrics = self.calculate_forward_performance_slice(
            signals_slice, data_slice, 'until_opposite'
        )

        if metrics['total_signals'] >= 3:
            score = metrics['avg_return']
            opposite_phase = 'New' if moon_phase == 'full' else 'Full'

            results.append({
                'moon_phase': moon_phase,
                'holding_days': f'Until {opposite_phase}',
                'holding_type': 'dynamic',
                'win_rate': metrics['win_rate'],
                'total_signals': metrics['total_signals'],
                'avg_return': metrics['avg_return'],
                'median_return': metrics['median_return'],
                'total_return': metrics['total_return'],
                'std_return': metrics['std_return'],
                'avg_holding_days': metrics.get('avg_holding_days', 0),
                'score': score,
            })

        results_df = pd.DataFrame(results)
        return results_df.sort_values('score', ascending=False) if not results_df.empty else pd.DataFrame()

    def calculate_forward_performance_slice(self, signals, data_slice, holding_days):
        """
        Calculate forward performance for a specific data slice
        holding_days: int for fixed days, or 'until_opposite' for dynamic holding
        """
        signal_dates = signals[signals['signal']].index
        returns = []
        winning_signals = 0
        actual_holding_days_list = []

        for signal_date in signal_dates:
            try:
                if signal_date not in data_slice.index:
                    continue

                entry_price = data_slice.loc[signal_date, 'Close']
                future_dates = data_slice.loc[signal_date:].index[1:]

                # Determine holding period
                if holding_days == 'until_opposite':
                    # Use dynamic holding until opposite moon phase
                    days_to_hold = signals.loc[signal_date, 'days_to_opposite_phase']
                    if pd.isna(days_to_hold) or days_to_hold <= 0:
                        continue
                    days_to_hold = int(days_to_hold)
                else:
                    # Use fixed holding period
                    days_to_hold = int(holding_days)

                if len(future_dates) >= days_to_hold:
                    exit_date = future_dates[days_to_hold - 1]
                    exit_price = data_slice.loc[exit_date, 'Close']
                    forward_return = (exit_price / entry_price) - 1
                    returns.append(forward_return)
                    actual_holding_days_list.append(days_to_hold)
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
            'avg_holding_days': np.mean(actual_holding_days_list) if actual_holding_days_list else 0,
        }

        return metrics

    def print_grid_results(self, results_df, fold_number, moon_phase):
        """Print optimization grid results"""
        if results_df.empty:
            print(f"| No valid results for {moon_phase.upper()} moon in Fold {fold_number}.")
            return

        print("\n" + "="*100)
        print(f"| 🌙 {moon_phase.upper()} MOON OPTIMIZATION - FOLD {fold_number} |")
        print("="*100)

        display_df = results_df.copy()
        display_df['Avg Ret'] = (display_df['avg_return'] * 100).map('{:.2f}%'.format)
        display_df['Std Dev'] = (display_df['std_return'] * 100).map('{:.2f}%'.format)
        display_df['Win Rate'] = (display_df['win_rate'] * 100).map('{:.1f}%'.format)
        display_df['Total Ret'] = (display_df['total_return'] * 100).map('{:.1f}%'.format)

        display_df = display_df[['holding_days', 'avg_holding_days', 'score', 'Win Rate', 'Avg Ret',
                                 'Std Dev', 'Total Ret', 'total_signals']]
        display_df.columns = ['Hold Period', 'Avg Days', 'Score', 'WR', 'Avg Ret', 'Std Dev', 'Tot Ret', 'Signals']

        print(display_df.head(12).to_markdown(index=False, floatfmt=(".0f", ".1f", ".4f")))
        print("="*100)

    def cross_validate_parameters(self, moon_phase='full', K=5):
        """
        Perform time-series K-Fold Cross-Validation for moon phase strategy
        """
        opposite_phase = 'New' if moon_phase == 'full' else 'Full'
        print("\n" + "~"*100)
        print(f"🏁 RUNNING K={K}-FOLD CV for {moon_phase.upper()} MOON STRATEGY")
        print(f"🎯 Fixed Holding Periods: {self.holding_period_range.tolist()} days")
        print(f"🎯 Dynamic Holding: Until {opposite_phase} Moon (~14 days)")
        print("~"*100)

        data = self.spy_data
        data_length = len(data)
        fold_size = data_length // K
        oos_results = []

        for i in range(K):
            sys.stdout.write(f"\n--- Processing Fold {i+1}/{K} ---")

            # Define OOS Test Set
            oos_start_idx = i * fold_size
            oos_end_idx = (i + 1) * fold_size if i < K - 1 else data_length
            oos_data = data.iloc[oos_start_idx:oos_end_idx]

            # Define IS Optimization Set
            is_data = data.iloc[:oos_start_idx]

            min_data_required = 100  # Need at least 100 days

            if len(is_data) < min_data_required or len(oos_data) < 30:
                sys.stdout.write(f" (Skipping: Insufficient data)\n")
                continue

            sys.stdout.write(f" (IS: {len(is_data)} days, OOS: {len(oos_data)} days)\n")

            # Optimize on IS data
            is_results_df = self.run_optimization_grid(is_data, moon_phase)
            self.print_grid_results(is_results_df, i + 1, moon_phase)

            if is_results_df.empty:
                print(f"No valid signals found in IS data for Fold {i+1}. Skipping OOS test.")
                continue

            # Get best parameters
            best_params = is_results_df.iloc[0]
            H_star = best_params['holding_days']

            # Check if it's a dynamic or fixed holding period
            if isinstance(H_star, str):
                # Dynamic holding (e.g., 'Until New' or 'Until Full')
                H_star_value = 'until_opposite'
                H_star_display = H_star
            else:
                # Fixed holding period
                H_star_value = int(H_star)
                H_star_display = f"{H_star_value} days"

            # Test on OOS data
            oos_signals = self.generate_moon_signals(moon_phase)
            oos_test_signals = oos_signals.loc[oos_data.index]

            oos_metrics = self.calculate_forward_performance_slice(
                oos_test_signals, oos_data, H_star_value
            )

            oos_results.append({
                'Fold': i + 1,
                'Moon_Phase': moon_phase,
                'H_star': H_star_display,
                'Avg_Hold_Days': oos_metrics.get('avg_holding_days', H_star_value if isinstance(H_star_value, int) else 0),
                'OOS_WinRate': oos_metrics['win_rate'],
                'OOS_AvgReturn': oos_metrics['avg_return'],
                'OOS_TotalSignals': oos_metrics['total_signals'],
                'OOS_TotalReturn': oos_metrics['total_return'],
                'OOS_StdReturn': oos_metrics['std_return'],
                'OOS_Start': oos_data.iloc[0].name.date(),
                'OOS_End': oos_data.iloc[-1].name.date(),
                'IS_Score': best_params['score'],
            })

            print(f"Optimal Hold: {H_star_display} (Avg: {oos_metrics.get('avg_holding_days', 0):.1f} days) "
                  f"(OOS WR: {oos_metrics['win_rate']:.1%}, AR: {oos_metrics['avg_return']:.2%}, "
                  f"Signals: {oos_metrics['total_signals']:.0f})")

        return pd.DataFrame(oos_results)

    def print_cv_results(self, oos_results_df, moon_phase):
        """Print cross-validation summary results"""
        print("\n" + "="*120)
        print(f"📈 CV SUMMARY - {moon_phase.upper()} MOON STRATEGY")
        print("="*120)

        if oos_results_df.empty:
            print(f"No valid out-of-sample results.")
            return oos_results_df

        # Table 1: Fold Results
        print("\n## 1. Out-of-Sample Performance by Fold")
        print("-" * 130)

        sorted_oos = oos_results_df.sort_values('OOS_AvgReturn', ascending=False)

        print(f"{'Fold':>4} {'Test Period':>25} {'Hold Strategy':>15} {'Avg Days':>9} "
              f"{'IS Score':>8} {'WR':>8} {'Avg Ret':>8} {'Std Dev':>8} {'Signals':>8}")
        print("-" * 130)

        for i, row in sorted_oos.iterrows():
            h_star_str = str(row['H_star'])[:15]  # Truncate if needed
            print(f"{row['Fold']:4.0f} {str(row['OOS_Start'])} to {str(row['OOS_End']):10} "
                  f"{h_star_str:>15} {row['Avg_Hold_Days']:9.1f} "
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
        print("\n## 3. Optimal Holding Period Frequency")
        print("-" * 80)
        h_counts = oos_results_df['H_star'].value_counts(normalize=True).round(3)
        print(f"    - Holding Periods Chosen: {h_counts.to_dict()}")

        print("="*120)
        return oos_results_df


# =============================================================================
# Execution Block
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🌙 MOON PHASE BACKTEST - FULL MOON & NEW MOON STRATEGIES")
    print("="*80)
    print("Testing if moon phases have predictive power for SPY returns")
    print("Method: Time-Series K-Fold Cross-Validation")
    print("")

    # Initialize backtest
    backtest = MoonPhaseBacktest()

    # Download SPY data
    backtest.download_spy_data(start_date='2000-01-01')

    # Calculate moon phases
    backtest.calculate_moon_phases()

    # Test both moon phases
    moon_phases_to_test = ['full', 'new']
    all_results = {}

    for moon_phase in moon_phases_to_test:
        print("\n" + "#"*120)
        print(f"🌙 TESTING {moon_phase.upper()} MOON STRATEGY")
        print("#"*120)

        # Run CV
        oos_results = backtest.cross_validate_parameters(
            moon_phase=moon_phase,
            K=5
        )

        # Print results
        oos_results = backtest.print_cv_results(oos_results, moon_phase)

        if not oos_results.empty:
            all_results[moon_phase] = {
                'Mean_WinRate': oos_results['OOS_WinRate'].mean(),
                'Mean_AvgReturn': oos_results['OOS_AvgReturn'].mean(),
                'Mean_TotalSignals': oos_results['OOS_TotalSignals'].mean(),
                'Std_AvgReturn': oos_results['OOS_AvgReturn'].std(),
            }

    # Final comparison
    print("\n" + "!"*120)
    print("🌟 FINAL COMPARISON: FULL MOON vs NEW MOON")
    print("!"*120)

    if all_results:
        print(f"\n| Moon Phase | Mean Win Rate | Mean Avg Return | Std Avg Return | Avg Signals |")
        print(f"| :---: | :---: | :---: | :---: | :---: |")

        for phase, results in all_results.items():
            print(f"| {phase.upper()} | {results['Mean_WinRate']:.1%} | "
                  f"**{results['Mean_AvgReturn']:.2%}** | {results['Std_AvgReturn']:.2%} | "
                  f"{results['Mean_TotalSignals']:.1f} |")

    print("\n" + "!"*120)
    print("✓ Moon Phase Analysis Complete!")
    print("!"*120)
