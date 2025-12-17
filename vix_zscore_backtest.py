"""
SPY/VIX Z-Score Backtest
Strategy:
- Enter long SPY when VIX Z-score rises above moving average from below -1
- Exit long when VIX Z-score crosses below moving average above 1

Based on the ThetaTrend VIX Z-Score indicator
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class VIXZScoreBacktest:
    def __init__(self, start_date='2010-01-01', end_date=None, initial_capital=100000):
        """
        Initialize the backtest

        Parameters:
        -----------
        start_date : str
            Start date for backtest (YYYY-MM-DD)
        end_date : str
            End date for backtest (YYYY-MM-DD), defaults to today
        initial_capital : float
            Initial portfolio value
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.portfolio = None

    def download_data(self):
        """Download SPY and VIX data from Yahoo Finance"""
        print(f"Downloading data from {self.start_date} to {self.end_date}...")

        # Download SPY (S&P 500 ETF)
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)

        # Download VIX (CBOE Volatility Index)
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)

        # Combine data
        self.data = pd.DataFrame()
        self.data['SPY_Close'] = spy['Close']
        self.data['VIX_Close'] = vix['Close']

        # Drop any rows with missing data
        self.data = self.data.dropna()

        print(f"Downloaded {len(self.data)} trading days of data")

    def calculate_vix_zscore(self, vix_short=10, vix_long=30, zscore_period=180):
        """
        Calculate VIX Z-Score indicator based on ThetaTrend methodology

        Parameters:
        -----------
        vix_short : int
            Short-term VIX moving average period (default: 10)
        vix_long : int
            Long-term VIX moving average period (default: 30)
        zscore_period : int
            Period for Z-score calculation (default: 180)
        """
        print("Calculating VIX Z-Score indicator...")

        # Calculate VIX moving averages
        self.data['VIX_10'] = self.data['VIX_Close'].rolling(window=vix_short).mean()
        self.data['VIX_30'] = self.data['VIX_Close'].rolling(window=vix_long).mean()

        # Calculate ratio (VIX30 / VIX10)
        self.data['VIX_Ratio'] = self.data['VIX_30'] / self.data['VIX_10']

        # Calculate Z-Score of the ratio
        ratio_mean = self.data['VIX_Ratio'].rolling(window=zscore_period).mean()
        ratio_std = self.data['VIX_Ratio'].rolling(window=zscore_period).std()

        self.data['Z_Score'] = (self.data['VIX_Ratio'] - ratio_mean) / ratio_std

        # Calculate WMA(2) of Z-Score (avgZv in ThinkScript)
        # WMA with period 2: weights are [2, 1], normalized
        weights = np.array([2, 1])
        weights = weights / weights.sum()

        self.data['avgZv'] = self.data['Z_Score'].rolling(window=2).apply(
            lambda x: np.sum(weights * x), raw=True
        )

        # Calculate moving averages of Z-Score for signal generation
        # Using EMA(3) and SMA(3) as in the ThinkScript
        self.data['Z_EMA3'] = self.data['Z_Score'].ewm(span=3, adjust=False).mean()
        self.data['Z_SMA3'] = self.data['Z_Score'].rolling(window=3).mean()

        # Drop rows with NaN values from calculations
        self.data = self.data.dropna()

        print(f"Z-Score calculation complete. {len(self.data)} valid data points.")

    def generate_signals(self):
        """
        Generate trading signals based on Z-Score crossovers

        Entry: Z-score rises above moving average from below -1
        Exit: Z-score crosses below moving average above 1
        """
        print("Generating trading signals...")

        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['Z_Score'] = self.data['Z_Score']
        self.signals['Z_MA'] = self.data['Z_SMA3']  # Using SMA3 as the moving average
        self.signals['Position'] = 0

        # Track if we're in a position
        in_position = False
        positions = []

        for i in range(1, len(self.signals)):
            current_z = self.signals['Z_Score'].iloc[i]
            prev_z = self.signals['Z_Score'].iloc[i-1]
            current_ma = self.signals['Z_MA'].iloc[i]
            prev_ma = self.signals['Z_MA'].iloc[i-1]

            if not in_position:
                # Entry condition: Z-score crosses above MA from below -1
                # Previous z-score was below -1 and below MA
                # Current z-score is above MA
                if (prev_z < -1 and prev_z < prev_ma and
                    current_z > current_ma):
                    positions.append(1)
                    in_position = True
                else:
                    positions.append(0)
            else:
                # Exit condition: Z-score crosses below MA from above 1
                # Previous z-score was above 1 and above MA
                # Current z-score is below MA
                if (prev_z > 1 and prev_z > prev_ma and
                    current_z < current_ma):
                    positions.append(0)
                    in_position = False
                else:
                    positions.append(1)  # Stay in position

        # Add initial position (0) for first row
        self.signals['Position'] = [0] + positions

        # Generate trading orders (1 = buy, -1 = sell, 0 = hold)
        self.signals['Signal'] = self.signals['Position'].diff()

        num_trades = len(self.signals[self.signals['Signal'] != 0])
        num_buys = len(self.signals[self.signals['Signal'] == 1])
        num_sells = len(self.signals[self.signals['Signal'] == -1])

        print(f"Generated {num_buys} buy signals and {num_sells} sell signals")

    def run_backtest(self):
        """Run the backtest and calculate portfolio performance"""
        print("Running backtest...")

        self.portfolio = pd.DataFrame(index=self.signals.index)
        self.portfolio['SPY_Close'] = self.data['SPY_Close']
        self.portfolio['Position'] = self.signals['Position']
        self.portfolio['Signal'] = self.signals['Signal']

        # Calculate daily returns of SPY
        self.portfolio['SPY_Returns'] = self.data['SPY_Close'].pct_change()

        # Calculate strategy returns (only when in position)
        self.portfolio['Strategy_Returns'] = (
            self.portfolio['Position'].shift(1) * self.portfolio['SPY_Returns']
        )

        # Calculate cumulative returns
        self.portfolio['SPY_Cumulative'] = (1 + self.portfolio['SPY_Returns']).cumprod()
        self.portfolio['Strategy_Cumulative'] = (1 + self.portfolio['Strategy_Returns']).cumprod()

        # Calculate portfolio value
        self.portfolio['Portfolio_Value'] = self.initial_capital * self.portfolio['Strategy_Cumulative']
        self.portfolio['SPY_Value'] = self.initial_capital * self.portfolio['SPY_Cumulative']

        print("Backtest complete!")

    def calculate_metrics(self):
        """Calculate performance metrics"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        # Basic metrics
        final_value = self.portfolio['Portfolio_Value'].iloc[-1]
        spy_final_value = self.portfolio['SPY_Value'].iloc[-1]

        total_return = (final_value / self.initial_capital - 1) * 100
        spy_return = (spy_final_value / self.initial_capital - 1) * 100

        # Annualized metrics
        years = len(self.portfolio) / 252  # Approximate trading days per year

        annualized_return = (np.power(final_value / self.initial_capital, 1/years) - 1) * 100
        spy_annualized_return = (np.power(spy_final_value / self.initial_capital, 1/years) - 1) * 100

        # Volatility (annualized)
        strategy_vol = self.portfolio['Strategy_Returns'].std() * np.sqrt(252) * 100
        spy_vol = self.portfolio['SPY_Returns'].std() * np.sqrt(252) * 100

        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annualized_return / strategy_vol) if strategy_vol > 0 else 0
        spy_sharpe = (spy_annualized_return / spy_vol) if spy_vol > 0 else 0

        # Maximum Drawdown
        cumulative = self.portfolio['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        spy_cumulative = self.portfolio['SPY_Cumulative']
        spy_running_max = spy_cumulative.expanding().max()
        spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max
        spy_max_drawdown = spy_drawdown.min() * 100

        # Win rate
        winning_trades = len(self.portfolio[
            (self.portfolio['Signal'] == -1) &
            (self.portfolio['Strategy_Returns'] > 0)
        ])
        total_trades = len(self.portfolio[self.portfolio['Signal'] == -1])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Time in market
        time_in_market = (self.portfolio['Position'].sum() / len(self.portfolio)) * 100

        print(f"\nPeriod: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Trading Days: {len(self.portfolio)}")
        print(f"Years: {years:.2f}")

        print("\n" + "-"*60)
        print("STRATEGY PERFORMANCE")
        print("-"*60)
        print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"Final Value:            ${final_value:,.2f}")
        print(f"Total Return:           {total_return:.2f}%")
        print(f"Annualized Return:      {annualized_return:.2f}%")
        print(f"Annualized Volatility:  {strategy_vol:.2f}%")
        print(f"Sharpe Ratio:           {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown:       {max_drawdown:.2f}%")
        print(f"Number of Trades:       {total_trades}")
        print(f"Win Rate:               {win_rate:.2f}%")
        print(f"Time in Market:         {time_in_market:.2f}%")

        print("\n" + "-"*60)
        print("BUY & HOLD SPY PERFORMANCE")
        print("-"*60)
        print(f"Final Value:            ${spy_final_value:,.2f}")
        print(f"Total Return:           {spy_return:.2f}%")
        print(f"Annualized Return:      {spy_annualized_return:.2f}%")
        print(f"Annualized Volatility:  {spy_vol:.2f}%")
        print(f"Sharpe Ratio:           {spy_sharpe:.2f}")
        print(f"Maximum Drawdown:       {spy_max_drawdown:.2f}%")

        print("\n" + "-"*60)
        print("RELATIVE PERFORMANCE")
        print("-"*60)
        print(f"Excess Return:          {total_return - spy_return:.2f}%")
        print(f"Excess Annual Return:   {annualized_return - spy_annualized_return:.2f}%")
        print("="*60 + "\n")

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': total_trades,
            'win_rate': win_rate,
            'time_in_market': time_in_market
        }

    def plot_results(self, save_path='backtest_results.png'):
        """Plot backtest results"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Plot 1: Portfolio Value
        axes[0].plot(self.portfolio.index, self.portfolio['Portfolio_Value'],
                    label='Strategy', linewidth=2, color='blue')
        axes[0].plot(self.portfolio.index, self.portfolio['SPY_Value'],
                    label='Buy & Hold SPY', linewidth=2, color='gray', alpha=0.7)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: VIX Z-Score
        axes[1].plot(self.data.index, self.data['Z_Score'],
                    label='Z-Score', linewidth=1.5, color='purple')
        axes[1].plot(self.data.index, self.data['Z_SMA3'],
                    label='SMA(3)', linewidth=1.5, color='orange')
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='±1')
        axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Mark buy/sell signals
        buy_signals = self.signals[self.signals['Signal'] == 1]
        sell_signals = self.signals[self.signals['Signal'] == -1]

        axes[1].scatter(buy_signals.index, self.data.loc[buy_signals.index, 'Z_Score'],
                       marker='^', color='green', s=100, label='Buy', zorder=5)
        axes[1].scatter(sell_signals.index, self.data.loc[sell_signals.index, 'Z_Score'],
                       marker='v', color='red', s=100, label='Sell', zorder=5)

        axes[1].set_title('VIX Z-Score with Trading Signals', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Z-Score', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Position
        axes[2].fill_between(self.portfolio.index, 0, self.portfolio['Position'],
                            alpha=0.3, color='green', label='In Position')
        axes[2].set_title('Position Over Time', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Position (1=Long, 0=Cash)', fontsize=12)
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Drawdown
        cumulative = self.portfolio['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        axes[3].fill_between(self.portfolio.index, 0, drawdown,
                            alpha=0.3, color='red', label='Drawdown')
        axes[3].set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Drawdown (%)', fontsize=12)
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")
        plt.close()

    def export_trades(self, filename='trades.csv'):
        """Export trade log to CSV"""
        trades = self.portfolio[self.portfolio['Signal'] != 0].copy()
        trades['Action'] = trades['Signal'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
        trades['Z_Score'] = self.signals.loc[trades.index, 'Z_Score']
        trades['Z_MA'] = self.signals.loc[trades.index, 'Z_MA']

        trades_export = trades[['Action', 'SPY_Close', 'Z_Score', 'Z_MA']].copy()
        trades_export.to_csv(filename)
        print(f"Trades exported to: {filename}")

    def run_full_backtest(self):
        """Run complete backtest pipeline"""
        self.download_data()
        self.calculate_vix_zscore()
        self.generate_signals()
        self.run_backtest()
        metrics = self.calculate_metrics()
        self.plot_results()
        self.export_trades()

        return metrics


if __name__ == "__main__":
    # Run backtest from 2010 to present
    backtest = VIXZScoreBacktest(
        start_date='2010-01-01',
        initial_capital=100000
    )

    # Run full backtest
    metrics = backtest.run_full_backtest()
