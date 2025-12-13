# trendlyne-widget

## Hourly Day Type Analysis

A Python tool that classifies trading days based on hourly price action.

**Two versions available:**
- **Yahoo Finance Version**: Downloads 1-hour bars automatically
- **30-Minute Data Version**: Uses your own 30-minute CSV files (Google Drive compatible)

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

---

## 30-Minute Data Version (Google Colab)

### For Use with Your Own Data Files

If you have 30-minute CSV files (e.g., from your Google Drive), use this version.

#### Google Colab Usage

1. **Open Google Colab**: https://colab.research.google.com/
2. **Create new notebook**
3. **Copy entire contents** of `colab_daytype_classifier.py`
4. **Paste into a cell and run**
5. **Follow the prompts**:
   - Select symbols (e.g., `SPY,QQQ` or `ALL`)
   - Select date range (e.g., `2024-01-01` or `ALL`)

#### Data Requirements

Your 30-minute CSV files should be:
- **Location**: `/content/drive/MyDrive/StockData/`
- **Naming**: `SYMBOL_30Min.csv` (e.g., `SPY_30Min.csv`)
- **Columns**: `t` (datetime), `Open`, `High`, `Low`, `Close`, `Volume`

#### How It Works

1. **Mounts Google Drive** automatically
2. **Reads 30-minute CSV files** from your StockData folder
3. **Aggregates into hourly sessions**:
   - 9:30-10:00 + 10:00-10:30 → 9:30 hourly session
   - 10:30-11:00 + 11:00-11:30 → 10:30 hourly session
   - And so on...
4. **Classifies each day** using the same logic as ThinkScript
5. **Exports results** to CSV and auto-downloads

#### Key Implementation Details

Based on extensive debugging to match ThinkScript exactly:
- ✅ **Opening price** (9:30 open) used as both H930 and L930
- ✅ **First bar (9:30) skipped** for directional detection (matches `isNewDay` reset)
- ✅ **Last bar (3:30 PM) skipped** for pullback detection
- ✅ **Two 30-min bars** aggregated into each hourly session

#### Standalone Python Usage

```python
from daytype_classifier_30min import DayTypeClassifier30Min

# Configure
DATA_DIR = "/path/to/your/30min/data"
SYMBOLS = ['SPY', 'QQQ']  # Or None for all
START_DATE = "2024-01-01"  # Or None for all

# Analyze
classifier = DayTypeClassifier30Min(data_directory=DATA_DIR)
classifier.analyze_all(symbols=SYMBOLS, start_date=START_DATE)
classifier.print_results()
classifier.export_to_csv("results.csv")
```

---

## Files in This Repository

- **`hourly_daytype_analysis.py`**: Yahoo Finance version (1-hour bars)
- **`daytype_classifier_30min.py`**: Standalone 30-minute version
- **`colab_daytype_classifier.py`**: Google Colab version with Drive integration
- **`example_usage.py`**: Usage examples for Yahoo Finance version
- **`debug_pullback.py`**: Debug script for testing classification logic
- **`requirements.txt`**: Python dependencies
- **`COLAB_INSTRUCTIONS.md`**: Detailed Google Colab instructions