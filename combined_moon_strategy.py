import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import sys
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class CombinedMoonPhaseStrategy:
    """
    Combined Moon Phase Strategy with PCALL Z-Score and MTSI Filters

    Tests if filtering moon phase signals with mean-reversion indicators
    improves win rate and reduces volatility
    """

    def __init__(self, mtsi_long_length=3, mtsi_short_length=2):
        self.spy_data = None
        self.pcall_data = None
        self.moon_data = None
        self.mtsi_data = None
        self.combined_data = None

        # Moon phase constants
        self.SYNOD_MONTH = 29.530588
        self.SYNOD_FULL_START = 105.2330205
        self.SYNOD_NEW_START = 120
        self.BASE_DAY_COUNT = 25569

        # MTSI parameters
        self.mtsi_long_length = mtsi_long_length
        self.mtsi_short_length = mtsi_short_length

        # Test parameters
        self.holding_period_range = np.array([7, 10, 14], dtype=int)
        self.pcall_z_lookback_range = np.array([50, 100, 200], dtype=int)
        self.pcall_z_threshold_range = np.array([-2.0, -1.5, -1.0])
        self.mtsi_z_lookback_range = np.array([50, 100, 200], dtype=int)
        self.mtsi_z_threshold_range = np.array([-2.0, -1.5, -1.0])

    def download_spy_data(self, start_date='2015-01-01', end_date=None):
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

        # Calculate VWAP approximation
        spy['TypicalPrice'] = (spy['High'] + spy['Low'] + spy['Close']) / 3
        spy['VWAP'] = spy['TypicalPrice']

        self.spy_data = spy
        print(f"✓ Downloaded {len(spy)} days of SPY data from {spy.index[0].date()} to {spy.index[-1].date()}")
        return spy

    def load_pcall_data(self, csv_file):
        """Load PCALL data from CSV"""
        print(f"Loading $PCALL data from {csv_file}...")

        try:
            df = pd.read_csv(csv_file, header=None, sep='\t')
        except Exception:
            df = pd.read_csv(csv_file, header=None, sep=',')

        # Load Date and Close columns (0 and 8)
        df = df[[0, 8]].copy()
        df.columns = ['DATE', 'CLOSE']

        # Clean and convert
        if df['CLOSE'].dtype == 'object':
            df['CLOSE'] = df['CLOSE'].astype(str).str.strip()
        df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')

        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE'])
        df = df.set_index('DATE').dropna(subset=['CLOSE']).copy()

        self.pcall_data = df
        print(f"✓ Loaded {len(df)} days of $PCALL data from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    def days_from_epoch(self, date):
        """Calculate days from Unix epoch"""
        epoch = datetime(1970, 1, 1)
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        delta = date - epoch
        return delta.days

    def calculate_moon_phases(self):
        """Calculate Full Moon and New Moon dates"""
        print("Calculating moon phases...")

        moon_events = []
        start_date = self.spy_data.index[0]
        end_date = self.spy_data.index[-1]

        current_date = start_date
        while current_date <= end_date:
            days_from_1970 = self.days_from_epoch(current_date)
            serial_date = days_from_1970 + self.BASE_DAY_COUNT

            # Check Full Moon
            for index in range(1237, 2470):
                full_moon_serial = round(self.SYNOD_FULL_START + index * self.SYNOD_MONTH, 0)
                if abs(serial_date - full_moon_serial) < 0.5:
                    moon_events.append({
                        'date': current_date,
                        'phase': 'full',
                        'weekday': current_date.weekday()
                    })
                    break

            # Check New Moon
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
            # Weekend adjustments
            adjusted_dates = []
            for idx, row in moon_df.iterrows():
                adj_date = row['date']
                if row['weekday'] == 5:  # Saturday
                    adj_date = row['date'] - timedelta(days=1)
                elif row['weekday'] == 6:  # Sunday
                    adj_date = row['date'] + timedelta(days=1)
                adjusted_dates.append(adj_date)

            moon_df['adjusted_date'] = adjusted_dates
            moon_df = moon_df.drop_duplicates(subset=['adjusted_date', 'phase'])

        self.moon_data = moon_df
        full_count = len(moon_df[moon_df['phase'] == 'full'])
        new_count = len(moon_df[moon_df['phase'] == 'new'])
        print(f"✓ Calculated {len(moon_df)} moon events ({full_count} Full, {new_count} New)")
        return moon_df

    def calculate_ema(self, series, length):
        """Calculate exponential moving average"""
        return series.ewm(span=length, adjust=False).mean()

    def calculate_mtsi(self):
        """Calculate MTSI indicator"""
        print("Calculating MTSI...")
        df = self.spy_data.copy()

        # MTSI calculation
        df['diff'] = np.log(df['Close'] / df['VWAP'])

        abs_diff = df['diff'].abs()
        abs_diff_smooth1 = self.calculate_ema(abs_diff, self.mtsi_long_length)
        abs_diff_smooth2 = self.calculate_ema(abs_diff_smooth1, self.mtsi_short_length)

        diff_smooth1 = self.calculate_ema(df['diff'], self.mtsi_long_length)
        diff_smooth2 = self.calculate_ema(diff_smooth1, self.mtsi_short_length)

        df['MTSI'] = np.where(
            abs_diff_smooth2 == 0,
            0,
            100 * diff_smooth2 / abs_diff_smooth2
        )

        self.mtsi_data = df
        print(f"✓ MTSI calculated")
        return df

    def combine_all_data(self):
        """Combine SPY, PCALL, and MTSI data"""
        print("Combining all datasets...")

        # Start with SPY MTSI data
        combined = self.mtsi_data[['Close', 'MTSI']].copy()

        # Merge PCALL
        combined = combined.merge(
            self.pcall_data[['CLOSE']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        combined.rename(columns={'CLOSE': 'PCALL'}, inplace=True)

        self.combined_data = combined.dropna()
        print(f"✓ Combined {len(self.combined_data)} days of data")
        return self.combined_data

    def calculate_pcall_zscore(self, data, lookback):
        """Calculate PCALL z-score"""
        pcall = data['PCALL']
        rolling_mean = pcall.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = pcall.rolling(window=lookback, min_periods=lookback).std()
        rolling_std[rolling_std.fillna(0) == 0] = np.nan
        return (pcall - rolling_mean) / rolling_std

    def calculate_mtsi_zscore(self, data, lookback):
        """Calculate MTSI z-score"""
        mtsi = data['MTSI']
        rolling_mean = mtsi.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = mtsi.rolling(window=lookback, min_periods=lookback).std()
        rolling_std[rolling_std.fillna(0) == 0] = np.nan
        return (mtsi - rolling_mean) / rolling_std

    def generate_filtered_signals(self, moon_phase='full',
                                  use_pcall_filter=False, pcall_z_lookback=100, pcall_z_threshold=-1.5,
                                  use_mtsi_filter=False, mtsi_z_lookback=100, mtsi_z_threshold=-1.5):
        """
        Generate moon phase signals with optional filters

        Filters:
        - PCALL: Only enter if PCALL z-score < threshold (oversold)
        - MTSI: Only enter if MTSI z-score < threshold (oversold, no bearish signal)
        """
        signals = pd.DataFrame(index=self.combined_data.index)
        signals['moon_signal'] = False
        signals['pcall_filter'] = True
        signals['mtsi_filter'] = True
        signals['final_signal'] = False

        # Get moon dates
        moon_dates = self.moon_data[self.moon_data['phase'] == moon_phase]['adjusted_date'].values

        # Mark moon signals
        for moon_date in moon_dates:
            closest_idx = self.combined_data.index.asof(pd.Timestamp(moon_date))
            if pd.notna(closest_idx) and closest_idx in signals.index:
                signals.loc[closest_idx, 'moon_signal'] = True

        # Apply PCALL filter if enabled
        if use_pcall_filter:
            pcall_zscore = self.calculate_pcall_zscore(self.combined_data, pcall_z_lookback)
            signals['pcall_zscore'] = pcall_zscore
            signals['pcall_filter'] = pcall_zscore < pcall_z_threshold

        # Apply MTSI filter if enabled
        if use_mtsi_filter:
            mtsi_zscore = self.calculate_mtsi_zscore(self.combined_data, mtsi_z_lookback)
            signals['mtsi_zscore'] = mtsi_zscore
            signals['mtsi_filter'] = mtsi_zscore < mtsi_z_threshold

        # Final signal: moon signal AND all filters
        signals['final_signal'] = (
            signals['moon_signal'] &
            signals['pcall_filter'] &
            signals['mtsi_filter']
        )

        return signals

    def calculate_forward_performance(self, signals, holding_days):
        """Calculate forward performance metrics"""
        signal_dates = signals[signals['final_signal']].index
        returns = []
        winning_signals = 0

        for signal_date in signal_dates:
            try:
                entry_price = self.combined_data.loc[signal_date, 'Close']
                future_dates = self.combined_data.loc[signal_date:].index[1:]

                if len(future_dates) >= holding_days:
                    exit_date = future_dates[holding_days - 1]
                    exit_price = self.combined_data.loc[exit_date, 'Close']
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

    def test_filter_combinations(self, moon_phase='full', holding_days=14):
        """
        Test different filter combinations to find the best setup
        """
        print(f"\n{'='*100}")
        print(f"TESTING FILTER COMBINATIONS FOR {moon_phase.upper()} MOON (H={holding_days} days)")
        print(f"{'='*100}\n")

        results = []

        # 1. Baseline: Moon alone (no filters)
        signals = self.generate_filtered_signals(
            moon_phase=moon_phase,
            use_pcall_filter=False,
            use_mtsi_filter=False
        )
        metrics = self.calculate_forward_performance(signals, holding_days)

        results.append({
            'Strategy': f'{moon_phase.upper()} Moon Only',
            'PCALL_Filter': 'No',
            'MTSI_Filter': 'No',
            'PCALL_Z_LB': '-',
            'PCALL_Z_Th': '-',
            'MTSI_Z_LB': '-',
            'MTSI_Z_Th': '-',
            'Win_Rate': metrics['win_rate'],
            'Avg_Return': metrics['avg_return'],
            'Std_Return': metrics['std_return'],
            'Total_Return': metrics['total_return'],
            'Total_Signals': metrics['total_signals'],
        })

        # 2. PCALL filter only
        for pcall_lb in self.pcall_z_lookback_range:
            for pcall_th in self.pcall_z_threshold_range:
                signals = self.generate_filtered_signals(
                    moon_phase=moon_phase,
                    use_pcall_filter=True,
                    pcall_z_lookback=pcall_lb,
                    pcall_z_threshold=pcall_th,
                    use_mtsi_filter=False
                )
                metrics = self.calculate_forward_performance(signals, holding_days)

                if metrics['total_signals'] >= 5:
                    results.append({
                        'Strategy': f'{moon_phase.upper()} + PCALL',
                        'PCALL_Filter': 'Yes',
                        'MTSI_Filter': 'No',
                        'PCALL_Z_LB': pcall_lb,
                        'PCALL_Z_Th': pcall_th,
                        'MTSI_Z_LB': '-',
                        'MTSI_Z_Th': '-',
                        'Win_Rate': metrics['win_rate'],
                        'Avg_Return': metrics['avg_return'],
                        'Std_Return': metrics['std_return'],
                        'Total_Return': metrics['total_return'],
                        'Total_Signals': metrics['total_signals'],
                    })

        # 3. MTSI filter only
        for mtsi_lb in self.mtsi_z_lookback_range:
            for mtsi_th in self.mtsi_z_threshold_range:
                signals = self.generate_filtered_signals(
                    moon_phase=moon_phase,
                    use_pcall_filter=False,
                    use_mtsi_filter=True,
                    mtsi_z_lookback=mtsi_lb,
                    mtsi_z_threshold=mtsi_th
                )
                metrics = self.calculate_forward_performance(signals, holding_days)

                if metrics['total_signals'] >= 5:
                    results.append({
                        'Strategy': f'{moon_phase.upper()} + MTSI',
                        'PCALL_Filter': 'No',
                        'MTSI_Filter': 'Yes',
                        'PCALL_Z_LB': '-',
                        'PCALL_Z_Th': '-',
                        'MTSI_Z_LB': mtsi_lb,
                        'MTSI_Z_Th': mtsi_th,
                        'Win_Rate': metrics['win_rate'],
                        'Avg_Return': metrics['avg_return'],
                        'Std_Return': metrics['std_return'],
                        'Total_Return': metrics['total_return'],
                        'Total_Signals': metrics['total_signals'],
                    })

        # 4. Both filters
        for pcall_lb in self.pcall_z_lookback_range:
            for pcall_th in self.pcall_z_threshold_range:
                for mtsi_lb in self.mtsi_z_lookback_range:
                    for mtsi_th in self.mtsi_z_threshold_range:
                        signals = self.generate_filtered_signals(
                            moon_phase=moon_phase,
                            use_pcall_filter=True,
                            pcall_z_lookback=pcall_lb,
                            pcall_z_threshold=pcall_th,
                            use_mtsi_filter=True,
                            mtsi_z_lookback=mtsi_lb,
                            mtsi_z_threshold=mtsi_th
                        )
                        metrics = self.calculate_forward_performance(signals, holding_days)

                        if metrics['total_signals'] >= 5:
                            results.append({
                                'Strategy': f'{moon_phase.upper()} + BOTH',
                                'PCALL_Filter': 'Yes',
                                'MTSI_Filter': 'Yes',
                                'PCALL_Z_LB': pcall_lb,
                                'PCALL_Z_Th': pcall_th,
                                'MTSI_Z_LB': mtsi_lb,
                                'MTSI_Z_Th': mtsi_th,
                                'Win_Rate': metrics['win_rate'],
                                'Avg_Return': metrics['avg_return'],
                                'Std_Return': metrics['std_return'],
                                'Total_Return': metrics['total_return'],
                                'Total_Signals': metrics['total_signals'],
                            })

        results_df = pd.DataFrame(results)
        return results_df.sort_values('Avg_Return', ascending=False)

    def print_results(self, results_df, moon_phase, holding_days):
        """Print formatted results"""
        if results_df.empty:
            print("No valid results found.")
            return

        print(f"\n{'='*120}")
        print(f"📊 RESULTS: {moon_phase.upper()} MOON STRATEGY (Holding: {holding_days} days)")
        print(f"{'='*120}\n")

        # Format for display
        display_df = results_df.copy()
        display_df['Win Rate'] = (display_df['Win_Rate'] * 100).map('{:.1f}%'.format)
        display_df['Avg Ret'] = (display_df['Avg_Return'] * 100).map('{:.2f}%'.format)
        display_df['Std Dev'] = (display_df['Std_Return'] * 100).map('{:.2f}%'.format)
        display_df['Tot Ret'] = (display_df['Total_Return'] * 100).map('{:.1f}%'.format)

        display_df = display_df[['Strategy', 'PCALL_Z_LB', 'PCALL_Z_Th', 'MTSI_Z_LB', 'MTSI_Z_Th',
                                 'Win Rate', 'Avg Ret', 'Std Dev', 'Tot Ret', 'Total_Signals']]
        display_df.columns = ['Strategy', 'P-LB', 'P-Th', 'M-LB', 'M-Th', 'WR', 'Avg Ret', 'Std', 'Tot Ret', 'Sigs']

        print("## Top 20 Strategies by Average Return")
        print("-" * 120)
        print(display_df.head(20).to_markdown(index=False))

        print("\n" + "="*120)


# =============================================================================
# Execution Block
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🌙 COMBINED MOON PHASE STRATEGY")
    print("="*80)
    print("Testing Moon Phase signals filtered by PCALL and MTSI z-scores")
    print("Goal: Improve win rate and reduce volatility")
    print("")

    # Initialize strategy
    strategy = CombinedMoonPhaseStrategy()

    # Download SPY data
    strategy.download_spy_data(start_date='2015-01-01')

    # Load PCALL data
    csv_file = input("Enter PCALL CSV filename (or press Enter for default): ").strip()
    if not csv_file:
        csv_file = "StrategyReports_$PCALL_12925.csv"

    strategy.load_pcall_data(csv_file)

    # Calculate moon phases
    strategy.calculate_moon_phases()

    # Calculate MTSI
    strategy.calculate_mtsi()

    # Combine all data
    strategy.combine_all_data()

    # Test both moon phases
    moon_phases = ['full', 'new']
    holding_periods = [7, 10, 14]

    for moon_phase in moon_phases:
        for holding_days in holding_periods:
            results = strategy.test_filter_combinations(
                moon_phase=moon_phase,
                holding_days=holding_days
            )
            strategy.print_results(results, moon_phase, holding_days)

    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)
