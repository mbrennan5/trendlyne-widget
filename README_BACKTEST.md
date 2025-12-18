# SPY/VIX Z-Score Trading Analysis

Comprehensive quantitative analysis suite for SPY (S&P 500 ETF) using the VIX Z-Score indicator.

This repository contains:
1. **Backtest System** - Test trading strategies based on VIX Z-Score signals
2. **Seasonal Edge Analysis** - Discover performance edges across market regimes

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

---

## Seasonal Edge Analysis

Beyond simple buy/sell signals, we analyze **when** and **how** to trade SPY based on market regimes.

### Market Seasons (Regimes)

The VIX Z-Score defines 4 distinct market seasons:

| Season | Condition | Regime | Color |
|--------|-----------|--------|-------|
| **Summer** | RiskZ > 0 & Rising | Risk-On Accelerating | Green |
| **Fall** | RiskZ > 0 & Falling | Risk-On Weakening | Dark Green |
| **Winter** | RiskZ ≤ 0 & Falling | Risk-Off Accelerating | Red |
| **Spring** | RiskZ ≤ 0 & Rising | Risk-Off Weakening | Dark Red |

### Edges Discovered

The seasonal analysis examines:

1. **Overnight Edge** (Close-to-Open returns)
   - Does SPY gap up/down overnight differently by season?
   - Should you hold overnight or close before the bell?

2. **Intraday Edge** (Open-to-Close returns)
   - Does SPY trend better intraday in certain seasons?
   - Should you day-trade or swing-trade?

3. **Swing Period Performance**
   - Optimal holding periods (2D, 3D, 5D, 10D) by season
   - Mean returns, win rates, and Sharpe ratios

4. **Statistical Significance**
   - Win rates by season and timeframe
   - Return distributions
   - Risk-adjusted performance (Sharpe ratios)

### Key Questions Answered

- **"Should I hold SPY overnight during Summer?"**
- **"What's the best holding period during Winter?"**
- **"Does SPY have stronger intraday or overnight edge in Spring?"**
- **"Which season offers the best risk-adjusted returns?"**

---

## Credit Spread Analysis

In addition to VIX-based signals, this repository includes comprehensive analysis of the **IEI/HYG credit spread ratio** for predicting forward SPY returns.

### Credit Spread Indicator

**IEI/HYG Ratio = 3-7yr Treasuries / High Yield Corporate Bonds**

- **Rising ratio** → Spreads widening → Flight to safety → Risk-off
- **Falling ratio** → Spreads tightening → Risk appetite → Risk-on

### Top Signals Discovered

Based on testing 200+ signals across various methodologies:

1. **Extreme Wide → Large Tightening** (Sharpe 5.52, +2.41% mean, 85% win rate)
   - Z-Score [1.0-1.5] + 1-day ROC < -0.3%
   - Best for 5-day holds

2. **Z-Score Neutral Zone [0.0-0.25]** (Sharpe 3.06, +1.60% mean, 75% win rate)
   - Spreads slightly elevated but normalizing
   - Best for 10-day holds

3. **Moderate Tight + Accelerating** (Sharpe 1.77, +0.53% mean, 68% win rate)
   - Z-Score [-1.5 to -0.5] + 5-day ROC > 0.3% + MA10 > MA50
   - Best for 5-day holds

See **`CREDIT_SPREAD_GUIDE.md`** for complete signal documentation and trading guidelines.

### Robustness Testing

The credit spread signals have been validated across multiple parameter sets to ensure they are not curve-fitted:

- **MA Smoothing Periods:** 5, 10, 20, 30, 50 days
- **Z-Score Lookbacks:** 60, 90, 120, 180, 252, 360 days
- **ROC Periods:** 1, 2, 3, 5, 10 days

Run the robustness test to verify signal stability across different parameter choices.

---

## Files

### Backtest System
- **`vix_zscore_backtest.py`** - Standalone Python script for running the backtest
- **`VIX_ZScore_Backtest.ipynb`** - Jupyter notebook for Google Colab

### Seasonal Edge Analysis
- **`seasonal_edge_analysis.py`** - Standalone Python script for edge analysis
- **`Seasonal_Edge_Analysis.ipynb`** - Jupyter notebook for Google Colab

### Credit Spread Analysis
- **`credit_spread_comprehensive_test.py`** - Tests 200+ credit spread signals
- **`credit_spread_robustness_test.py`** - Parameter robustness validation
- **`Credit_Spread_Robustness_Test.ipynb`** - Robustness test for Google Colab
- **`CREDIT_SPREAD_GUIDE.md`** - Complete trading guide with top signals

### Shared
- **`requirements.txt`** - Python dependencies
- **`EDGE_PATTERNS.md`** - Expected seasonal patterns reference

## Usage

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run VIX backtest
python vix_zscore_backtest.py

# Run seasonal edge analysis
python seasonal_edge_analysis.py

# Run credit spread comprehensive test
python credit_spread_comprehensive_test.py

# Run robustness validation
python credit_spread_robustness_test.py
```

### Running in Google Colab

**For VIX Backtest:**
1. Upload `VIX_ZScore_Backtest.ipynb` to Google Colab
2. Run each cell sequentially
3. Optionally save results to Google Drive

**For Seasonal Edge Analysis:**
1. Upload `Seasonal_Edge_Analysis.ipynb` to Google Colab
2. Run each cell sequentially
3. View comprehensive edge analysis and charts
4. Export results to Google Drive

**For Credit Spread Robustness Test:**
1. Upload `Credit_Spread_Robustness_Test.ipynb` to Google Colab
2. Run each cell sequentially
3. View parameter sensitivity analysis
4. Verify signal stability across different lookback periods
5. Export robustness metrics and visualizations

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

### Seasonal Edge Analysis Capabilities
- Automatic season classification (Summer, Fall, Winter, Spring)
- Overnight vs intraday return comparison
- Swing period analysis (2D, 3D, 5D, 10D)
- Win rate and Sharpe ratio by season
- Return distribution analysis
- Current market regime identification

### Edge Analysis Visualizations
- Overnight vs Intraday returns by season (bar chart)
- Win rates comparison (bar chart)
- Sharpe ratios by season (bar chart)
- Swing period performance (line chart)
- Return distributions (histograms)

### Edge Analysis Exports
- Seasonal statistics summary CSV
- Detailed day-by-day data CSV
- Comprehensive charts (PNG)

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
