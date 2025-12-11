# trendlyne-widget

## Hourly Day Type Analysis

A Python tool that classifies trading days based on hourly price action using Yahoo Finance data.

### Day Type Classifications

- **Range Day**: Price remains confined to a range, with 4+ hours overlapping the opening price OR directional movement that reverses back to opening after 11:00 AM
- **DWP (Directional With Pullbacks)**: Price moves directionally away from opening but experiences hourly pullbacks
- **DNP (Directional No Pullbacks)**: Price moves directionally away from opening without significant pullbacks
- **N/A**: No clear directional bias established

### Features

- Uses 1-hour bars from Yahoo Finance
- Opening price treated as the opening range reference (both high and low)
- Analyzes Regular Trading Hours (RTH): 9:30 AM - 4:15 PM EST
- Tracks hourly pullbacks and range overlaps
- Exports results to CSV
- Configurable lookback period

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```python
from hourly_daytype_analysis import HourlyDayTypeAnalyzer

# Analyze symbols for the last 20 trading days
symbols = ['SPY', 'QQQ', 'AAPL']
analyzer = HourlyDayTypeAnalyzer(symbols=symbols, lookback_days=20)
analyzer.analyze_all()
analyzer.print_results()
analyzer.export_to_csv("results.csv")
```

### Usage Examples

Run the main analysis:
```bash
python hourly_daytype_analysis.py
```

Or see more examples:
```bash
python example_usage.py
```

### How It Works

1. **Downloads 1-hour data** from Yahoo Finance for specified symbols
2. **Filters to RTH hours** (9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30)
3. **For each trading day**:
   - Uses opening price as the reference point
   - Tracks if price moves above (Dir_Up) or below (Dir_Down) opening
   - Detects hourly pullbacks (when current hour reverses previous hour's direction)
   - Counts hours that overlap with opening price
   - Checks for late reversals back to opening price
4. **Classifies the day** based on directional bias, pullbacks, and range behavior

### Output Format

Results include:
- Date
- Day Type (RANGE DAY, DWP, DNP, or N/A)
- Opening price
- Directional bias (Up/Down/None)
- Whether pullbacks occurred
- Number of hours tested at opening price
- Total hours in the trading day

### Configuration

Edit the main() function in `hourly_daytype_analysis.py`:
```python
SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']  # Your symbols
LOOKBACK_DAYS = 20  # Number of trading days
```