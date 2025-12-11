#!/usr/bin/env python3
"""
Example usage of the Hourly Day Type Analyzer
"""

from hourly_daytype_analysis import HourlyDayTypeAnalyzer

# Example 1: Analyze a single symbol
print("Example 1: Single Symbol Analysis")
print("-" * 80)
analyzer = HourlyDayTypeAnalyzer(symbols=['SPY'], lookback_days=10)
analyzer.analyze_all()
analyzer.print_results()

print("\n" * 2)

# Example 2: Analyze multiple symbols
print("Example 2: Multiple Symbols Analysis")
print("-" * 80)
symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
analyzer = HourlyDayTypeAnalyzer(symbols=symbols, lookback_days=20)
analyzer.analyze_all()
analyzer.print_results()
analyzer.export_to_csv("multi_symbol_analysis.csv")

print("\n" * 2)

# Example 3: Analyze individual stocks
print("Example 3: Individual Stocks")
print("-" * 80)
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
analyzer = HourlyDayTypeAnalyzer(symbols=tech_stocks, lookback_days=15)
results = analyzer.analyze_all()

# Print only recent DNP days (Directional No Pullbacks)
for symbol, df in results.items():
    dnp_days = df[df['DayType'] == 'DNP (Directional No Pullbacks)']
    if len(dnp_days) > 0:
        print(f"\n{symbol} - DNP Days:")
        print(dnp_days[['Date', 'DayType', 'Directional', 'Opening']].to_string(index=False))

print("\n" * 2)

# Example 4: Custom analysis with filtering
print("Example 4: Filter for specific day types")
print("-" * 80)
analyzer = HourlyDayTypeAnalyzer(symbols=['SPY'], lookback_days=30)
analyzer.analyze_all()

for symbol, df in analyzer.results.items():
    print(f"\n{symbol} Analysis:")
    print(f"Total days analyzed: {len(df)}")
    print(f"\nDay Type Distribution:")
    print(df['DayType'].value_counts())

    print(f"\nRange Days:")
    range_days = df[df['DayType'] == 'RANGE DAY']
    if len(range_days) > 0:
        print(range_days[['Date', 'TestCount', 'NumHours']].to_string(index=False))

    print(f"\nDWP Days (Directional with Pullbacks):")
    dwp_days = df[df['DayType'] == 'DWP (Directional w/ Pullbacks)']
    if len(dwp_days) > 0:
        print(dwp_days[['Date', 'Directional', 'Opening']].to_string(index=False))
