# Google Colab Instructions

## Quick Start (3 Steps)

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/)

### Step 2: Create Two Cells

**Cell 1 - Install Dependencies:**
```python
!pip install yfinance pandas numpy pytz
```

**Cell 2 - Copy the entire content from `colab_hourly_daytype_analysis.py`**

### Step 3: Customize and Run

Modify these variables in the code:
```python
# List of symbols to analyze
SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']

# Number of trading days to analyze
LOOKBACK_DAYS = 20
```

Then run the cell!

## What You'll Get

The script will:
1. Download 1-hour bar data from Yahoo Finance
2. Analyze each trading day for the specified symbols
3. Classify each day as:
   - **RANGE DAY** - Price stayed near opening or reversed back
   - **DWP** - Directional with pullbacks
   - **DNP** - Directional without pullbacks
   - **N/A** - No clear pattern

4. Display results in a formatted table
5. Export to CSV and auto-download it

## Example Output

```
================================================================================
Symbol: SPY
================================================================================
       Date                      DayType  Opening Directional  HasPullback  TestCount  NumHours
 2024-11-22                    RANGE DAY   589.50        None        False          5         7
 2024-11-25                    RANGE DAY   593.25        None        False          6         7
 2024-11-26  DWP (Directional w/ Pullbacks)   595.80          Up         True          2         7
 2024-11-27  DNP (Directional No Pullbacks)   597.50          Up        False          1         7
```

## Customization Examples

### Analyze just one symbol for 5 days:
```python
SYMBOLS = ['SPY']
LOOKBACK_DAYS = 5
```

### Analyze multiple stocks for last 30 days:
```python
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
LOOKBACK_DAYS = 30
```

### Analyze sector ETFs:
```python
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV']
LOOKBACK_DAYS = 20
```

## Troubleshooting

**If you get "No data returned":**
- Check that the symbol is correct
- Some symbols may not have hourly data available
- Try a different lookback period

**If you get timezone warnings:**
- This is normal and can be ignored

**To see more details:**
- The exported CSV file contains additional columns with debug information
