# SPY/VIX Z-Score Backtest

A quantitative trading strategy backtest for SPY (S&P 500 ETF) using the VIX Z-Score indicator.

## Strategy Overview

This backtest implements a long-only strategy based on the ThetaTrend VIX Z-Score indicator:

### Entry Signal
- **Enter Long:** When the VIX Z-Score rises above its moving average **from below -1**
- This indicates that market fear (VIX) is starting to normalize from extreme low levels

### Exit Signal
- **Exit Long:** When the VIX Z-Score crosses below its moving average **from above 1**
- This indicates that market fear is starting to decline from elevated levels

## Indicator Details

### VIX Z-Score Calculation

1. **VIX Moving Averages:**
   - Short-term: 10-day SMA of VIX
   - Long-term: 30-day SMA of VIX

2. **VIX Ratio:**
   - Ratio = VIX30 / VIX10

3. **Z-Score:**
   - Calculate the Z-Score of the ratio over a 180-day period
   - Z-Score = (Ratio - Mean) / StdDev

4. **Signal Generation:**
   - Use 3-day SMA of Z-Score as the moving average for crossover signals

## Files

- **`vix_zscore_backtest.py`** - Standalone Python script for running the backtest
- **`VIX_ZScore_Backtest.ipynb`** - Jupyter notebook for Google Colab
- **`requirements.txt`** - Python dependencies

## Usage

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python vix_zscore_backtest.py
```

### Running in Google Colab

1. Upload `VIX_ZScore_Backtest.ipynb` to Google Colab
2. Run each cell sequentially
3. Optionally save results to Google Drive

## Features

### Backtest Capabilities
- Historical data download from Yahoo Finance
- VIX Z-Score indicator calculation
- Signal generation with entry/exit logic
- Portfolio simulation with position tracking
- Comprehensive performance metrics

### Performance Metrics
- Total and annualized returns
- Sharpe ratio
- Maximum drawdown
- Win rate
- Time in market percentage
- Comparison vs. Buy & Hold SPY

### Visualizations
- Portfolio value over time
- VIX Z-Score with buy/sell signals
- Position tracking
- Strategy drawdown

### Exports
- Trade log CSV
- Full portfolio data CSV
- Performance charts (PNG)

## Default Parameters

```python
START_DATE = '2010-01-01'
INITIAL_CAPITAL = 100000
VIX_SHORT = 10        # Short-term VIX MA
VIX_LONG = 30         # Long-term VIX MA
ZSCORE_PERIOD = 180   # Z-Score calculation period
```

## Customization

You can modify the backtest parameters by editing:

### In Python Script
```python
backtest = VIXZScoreBacktest(
    start_date='2015-01-01',  # Change start date
    end_date='2023-12-31',     # Change end date
    initial_capital=50000      # Change capital
)
```

### In Jupyter Notebook
Modify the configuration cell:
```python
START_DATE = '2015-01-01'
END_DATE = '2023-12-31'
INITIAL_CAPITAL = 50000
```

## Strategy Logic

The strategy is designed to:
1. **Buy SPY during fear normalization** - When extreme low volatility (Z-Score < -1) starts reversing
2. **Sell SPY during fear spikes** - When volatility becomes elevated (Z-Score > 1) and starts declining

This creates a **risk-on/risk-off** approach:
- Risk-ON: Low volatility recovering → Stay long equities
- Risk-OFF: High volatility declining → Exit to cash

## Interpretation

### Z-Score Values
- **Z < -2**: Extreme low volatility (complacency)
- **Z < -1**: Low volatility
- **Z = 0**: Normal volatility
- **Z > 1**: Elevated volatility
- **Z > 2**: Extreme high volatility (panic)

### Signal Logic
The strategy aims to capture equity uptrends while avoiding major drawdowns by:
1. Entering when fear is low but starting to normalize
2. Staying invested during normal market conditions
3. Exiting when fear spikes and starts to decline

## Credits

Based on the **MACRO Mosaic KenLeng VIX Z-Score** indicator
Provided courtesy of [ThetaTrend.com](https://thetatrend.com)

## Disclaimer

This backtest is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.

## License

The VIX Z-Score indicator is provided courtesy of ThetaTrend.com. Please provide a link back to ThetaTrend.com when sharing.
