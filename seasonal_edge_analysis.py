"""
SPY Seasonal Edge Analysis
Analyzes SPY performance across VIX Z-Score regimes (seasons)

Seasons:
- Summer (Green): RiskZ > 0 and rising (Risk-On accelerating)
- Fall (Dark Green): RiskZ > 0 but falling (Risk-On weakening)
- Winter (Red): RiskZ <= 0 and falling (Risk-Off accelerating)
- Spring (Dark Red): RiskZ <= 0 but rising (Risk-Off weakening)

Analyzes:
1. Overnight returns (close-to-open) by season
2. Intraday returns (open-to-close) by season
3. Swing period performance (multi-day holds) by season
4. Statistical edges and optimal holding periods
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

class SeasonalEdgeAnalyzer:
    def __init__(self, start_date='2010-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.data = None
        self.analysis = None

    def download_data(self):
        """Download SPY and VIX data with intraday info"""
        print(f"Downloading data from {self.start_date} to {self.end_date}...")

        # Download SPY with full OHLC data
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)

        # Download VIX
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)

        # Combine data
        self.data = pd.DataFrame()
        self.data['SPY_Open'] = spy['Open']
        self.data['SPY_High'] = spy['High']
        self.data['SPY_Low'] = spy['Low']
        self.data['SPY_Close'] = spy['Close']
        self.data['SPY_Volume'] = spy['Volume']
        self.data['VIX_Close'] = vix['Close']

        # Drop any rows with missing data
        self.data = self.data.dropna()

        print(f"Downloaded {len(self.data)} trading days")

    def calculate_risk_z(self, vix_fast=10, vix_slow=30, z_lookback=180):
        """Calculate Risk Z-Score exactly as in ThinkScript"""
        print("Calculating Risk Z-Score...")

        # VIX moving averages (using SMA to match ThinkScript Average)
        self.data['VIX_Fast'] = self.data['VIX_Close'].rolling(window=vix_fast).mean()
        self.data['VIX_Slow'] = self.data['VIX_Close'].rolling(window=vix_slow).mean()

        # Risk Index (VIX_Slow / VIX_Fast)
        self.data['Risk_Index'] = self.data['VIX_Slow'] / self.data['VIX_Fast']

        # Z-Score normalization
        risk_mean = self.data['Risk_Index'].rolling(window=z_lookback).mean()
        risk_std = self.data['Risk_Index'].rolling(window=z_lookback).std()

        self.data['Risk_Z'] = (self.data['Risk_Index'] - risk_mean) / risk_std

        # Drop NaN rows
        self.data = self.data.dropna()

        print(f"Risk Z calculated. {len(self.data)} valid data points.")

    def classify_seasons(self):
        """Classify each day into seasons based on Risk Z"""
        print("Classifying seasons...")

        # Calculate if Risk Z is rising or falling
        self.data['Risk_Z_Change'] = self.data['Risk_Z'].diff()

        # Season classification
        conditions = [
            (self.data['Risk_Z'] > 0) & (self.data['Risk_Z_Change'] >= 0),  # Summer
            (self.data['Risk_Z'] > 0) & (self.data['Risk_Z_Change'] < 0),   # Fall
            (self.data['Risk_Z'] <= 0) & (self.data['Risk_Z_Change'] <= 0), # Winter
            (self.data['Risk_Z'] <= 0) & (self.data['Risk_Z_Change'] > 0)   # Spring
        ]

        choices = ['Summer', 'Fall', 'Winter', 'Spring']

        self.data['Season'] = np.select(conditions, choices, default='Unknown')

        # Season stats
        season_counts = self.data['Season'].value_counts()
        print("\nSeason Distribution:")
        for season, count in season_counts.items():
            pct = count / len(self.data) * 100
            print(f"  {season}: {count} days ({pct:.1f}%)")

    def calculate_returns(self):
        """Calculate various return types"""
        print("\nCalculating returns...")

        # Overnight return (prev close to current open)
        self.data['Overnight_Return'] = (
            (self.data['SPY_Open'] - self.data['SPY_Close'].shift(1)) /
            self.data['SPY_Close'].shift(1) * 100
        )

        # Intraday return (open to close)
        self.data['Intraday_Return'] = (
            (self.data['SPY_Close'] - self.data['SPY_Open']) /
            self.data['SPY_Open'] * 100
        )

        # Daily return (close to close)
        self.data['Daily_Return'] = (
            self.data['SPY_Close'].pct_change() * 100
        )

        # 2-day swing return
        self.data['Swing_2D'] = (
            (self.data['SPY_Close'].shift(-2) - self.data['SPY_Close']) /
            self.data['SPY_Close'] * 100
        )

        # 3-day swing return
        self.data['Swing_3D'] = (
            (self.data['SPY_Close'].shift(-3) - self.data['SPY_Close']) /
            self.data['SPY_Close'] * 100
        )

        # 5-day swing return
        self.data['Swing_5D'] = (
            (self.data['SPY_Close'].shift(-5) - self.data['SPY_Close']) /
            self.data['SPY_Close'] * 100
        )

        # 10-day swing return
        self.data['Swing_10D'] = (
            (self.data['SPY_Close'].shift(-10) - self.data['SPY_Close']) /
            self.data['SPY_Close'] * 100
        )

        # Drop NaN rows from forward-looking calculations
        print(f"Returns calculated.")

    def analyze_by_season(self):
        """Analyze returns by season"""
        print("\n" + "="*80)
        print("SEASONAL EDGE ANALYSIS")
        print("="*80)

        seasons = ['Summer', 'Fall', 'Winter', 'Spring']

        results = []

        for season in seasons:
            season_data = self.data[self.data['Season'] == season].copy()

            if len(season_data) == 0:
                continue

            # Calculate statistics for each return type
            stats = {
                'Season': season,
                'Days': len(season_data),

                # Overnight stats
                'Overnight_Mean': season_data['Overnight_Return'].mean(),
                'Overnight_Median': season_data['Overnight_Return'].median(),
                'Overnight_Std': season_data['Overnight_Return'].std(),
                'Overnight_Win%': (season_data['Overnight_Return'] > 0).sum() / len(season_data) * 100,
                'Overnight_Sharpe': (season_data['Overnight_Return'].mean() /
                                    season_data['Overnight_Return'].std() * np.sqrt(252)),

                # Intraday stats
                'Intraday_Mean': season_data['Intraday_Return'].mean(),
                'Intraday_Median': season_data['Intraday_Return'].median(),
                'Intraday_Std': season_data['Intraday_Return'].std(),
                'Intraday_Win%': (season_data['Intraday_Return'] > 0).sum() / len(season_data) * 100,
                'Intraday_Sharpe': (season_data['Intraday_Return'].mean() /
                                   season_data['Intraday_Return'].std() * np.sqrt(252)),

                # Daily stats
                'Daily_Mean': season_data['Daily_Return'].mean(),
                'Daily_Std': season_data['Daily_Return'].std(),
                'Daily_Win%': (season_data['Daily_Return'] > 0).sum() / len(season_data) * 100,

                # Swing stats
                'Swing_2D_Mean': season_data['Swing_2D'].mean(),
                'Swing_3D_Mean': season_data['Swing_3D'].mean(),
                'Swing_5D_Mean': season_data['Swing_5D'].mean(),
                'Swing_10D_Mean': season_data['Swing_10D'].mean(),
            }

            results.append(stats)

        self.analysis = pd.DataFrame(results)

        # Display results
        self._print_analysis()

    def _print_analysis(self):
        """Print formatted analysis results"""

        for idx, row in self.analysis.iterrows():
            season = row['Season']
            print(f"\n{'='*80}")
            print(f"{season.upper()} (Risk-{'On' if season in ['Summer', 'Fall'] else 'Off'})")
            print(f"{'='*80}")
            print(f"Total Days: {row['Days']:.0f}")

            print(f"\n--- OVERNIGHT (Close-to-Open) ---")
            print(f"  Mean Return:    {row['Overnight_Mean']:>8.3f}%")
            print(f"  Median Return:  {row['Overnight_Median']:>8.3f}%")
            print(f"  Std Dev:        {row['Overnight_Std']:>8.3f}%")
            print(f"  Win Rate:       {row['Overnight_Win%']:>8.1f}%")
            print(f"  Sharpe Ratio:   {row['Overnight_Sharpe']:>8.2f}")

            print(f"\n--- INTRADAY (Open-to-Close) ---")
            print(f"  Mean Return:    {row['Intraday_Mean']:>8.3f}%")
            print(f"  Median Return:  {row['Intraday_Median']:>8.3f}%")
            print(f"  Std Dev:        {row['Intraday_Std']:>8.3f}%")
            print(f"  Win Rate:       {row['Intraday_Win%']:>8.1f}%")
            print(f"  Sharpe Ratio:   {row['Intraday_Sharpe']:>8.2f}")

            print(f"\n--- DAILY (Close-to-Close) ---")
            print(f"  Mean Return:    {row['Daily_Mean']:>8.3f}%")
            print(f"  Std Dev:        {row['Daily_Std']:>8.3f}%")
            print(f"  Win Rate:       {row['Daily_Win%']:>8.1f}%")

            print(f"\n--- SWING PERIODS ---")
            print(f"  2-Day Mean:     {row['Swing_2D_Mean']:>8.3f}%")
            print(f"  3-Day Mean:     {row['Swing_3D_Mean']:>8.3f}%")
            print(f"  5-Day Mean:     {row['Swing_5D_Mean']:>8.3f}%")
            print(f"  10-Day Mean:    {row['Swing_10D_Mean']:>8.3f}%")

        # Summary comparisons
        print(f"\n{'='*80}")
        print("EDGE SUMMARY")
        print(f"{'='*80}")

        # Find best season for each metric
        best_overnight = self.analysis.loc[self.analysis['Overnight_Mean'].idxmax()]
        best_intraday = self.analysis.loc[self.analysis['Intraday_Mean'].idxmax()]
        best_daily = self.analysis.loc[self.analysis['Daily_Mean'].idxmax()]

        print(f"\nBest Overnight Edge:  {best_overnight['Season']} ({best_overnight['Overnight_Mean']:.3f}%/day)")
        print(f"Best Intraday Edge:   {best_intraday['Season']} ({best_intraday['Intraday_Mean']:.3f}%/day)")
        print(f"Best Daily Edge:      {best_daily['Season']} ({best_daily['Daily_Mean']:.3f}%/day)")

        # Overnight vs Intraday comparison
        print(f"\n--- Overnight vs Intraday by Season ---")
        for idx, row in self.analysis.iterrows():
            on_edge = row['Overnight_Mean']
            id_edge = row['Intraday_Mean']
            better = "OVERNIGHT" if on_edge > id_edge else "INTRADAY"
            diff = abs(on_edge - id_edge)
            print(f"{row['Season']:10s}: {better:10s} edge (+{diff:.3f}%)")

    def plot_seasonal_analysis(self, save_path='seasonal_edge_analysis.png'):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        seasons = ['Summer', 'Fall', 'Winter', 'Spring']
        colors = {'Summer': 'green', 'Fall': 'orange', 'Winter': 'red', 'Spring': 'lightcoral'}

        # Plot 1: Overnight vs Intraday Returns
        ax = axes[0, 0]
        overnight = [self.analysis[self.analysis['Season']==s]['Overnight_Mean'].values[0] for s in seasons]
        intraday = [self.analysis[self.analysis['Season']==s]['Intraday_Mean'].values[0] for s in seasons]

        x = np.arange(len(seasons))
        width = 0.35

        ax.bar(x - width/2, overnight, width, label='Overnight', alpha=0.8)
        ax.bar(x + width/2, intraday, width, label='Intraday', alpha=0.8)
        ax.set_xlabel('Season')
        ax.set_ylabel('Mean Return (%)')
        ax.set_title('Overnight vs Intraday Returns by Season', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 2: Win Rates
        ax = axes[0, 1]
        overnight_wr = [self.analysis[self.analysis['Season']==s]['Overnight_Win%'].values[0] for s in seasons]
        intraday_wr = [self.analysis[self.analysis['Season']==s]['Intraday_Win%'].values[0] for s in seasons]

        ax.bar(x - width/2, overnight_wr, width, label='Overnight', alpha=0.8)
        ax.bar(x + width/2, intraday_wr, width, label='Intraday', alpha=0.8)
        ax.set_xlabel('Season')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rates by Season', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot 3: Sharpe Ratios
        ax = axes[1, 0]
        overnight_sharpe = [self.analysis[self.analysis['Season']==s]['Overnight_Sharpe'].values[0] for s in seasons]
        intraday_sharpe = [self.analysis[self.analysis['Season']==s]['Intraday_Sharpe'].values[0] for s in seasons]

        ax.bar(x - width/2, overnight_sharpe, width, label='Overnight', alpha=0.8)
        ax.bar(x + width/2, intraday_sharpe, width, label='Intraday', alpha=0.8)
        ax.set_xlabel('Season')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 4: Swing Period Returns
        ax = axes[1, 1]
        swing_periods = ['2D', '3D', '5D', '10D']
        for season in seasons:
            row = self.analysis[self.analysis['Season'] == season].iloc[0]
            swing_returns = [row['Swing_2D_Mean'], row['Swing_3D_Mean'],
                           row['Swing_5D_Mean'], row['Swing_10D_Mean']]
            ax.plot(swing_periods, swing_returns, marker='o', label=season,
                   color=colors[season], linewidth=2)

        ax.set_xlabel('Swing Period')
        ax.set_ylabel('Mean Return (%)')
        ax.set_title('Swing Period Returns by Season', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 5: Return Distributions (Overnight)
        ax = axes[2, 0]
        for season in seasons:
            season_data = self.data[self.data['Season'] == season]['Overnight_Return'].dropna()
            ax.hist(season_data, bins=50, alpha=0.5, label=season, color=colors[season])

        ax.set_xlabel('Overnight Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Overnight Return Distributions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 6: Return Distributions (Intraday)
        ax = axes[2, 1]
        for season in seasons:
            season_data = self.data[self.data['Season'] == season]['Intraday_Return'].dropna()
            ax.hist(season_data, bins=50, alpha=0.5, label=season, color=colors[season])

        ax.set_xlabel('Intraday Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Intraday Return Distributions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCharts saved to: {save_path}")
        plt.close()

    def export_results(self, filename='seasonal_analysis.csv'):
        """Export detailed analysis to CSV"""
        self.analysis.to_csv(filename, index=False)
        print(f"Analysis exported to: {filename}")

        # Also export detailed day-by-day data
        detail_file = filename.replace('.csv', '_detail.csv')
        export_cols = ['Season', 'Risk_Z', 'SPY_Close', 'Overnight_Return',
                      'Intraday_Return', 'Daily_Return', 'Swing_2D',
                      'Swing_3D', 'Swing_5D', 'Swing_10D']
        self.data[export_cols].to_csv(detail_file)
        print(f"Detailed data exported to: {detail_file}")

    def run_full_analysis(self):
        """Run complete seasonal edge analysis"""
        self.download_data()
        self.calculate_risk_z()
        self.classify_seasons()
        self.calculate_returns()
        self.analyze_by_season()
        self.plot_seasonal_analysis()
        self.export_results()


if __name__ == "__main__":
    analyzer = SeasonalEdgeAnalyzer(start_date='2010-01-01')
    analyzer.run_full_analysis()
