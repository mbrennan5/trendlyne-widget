"""
============================================================================
DAY TYPE RESULTS ANALYZER - GOOGLE COLAB
============================================================================
Analyzes your completed daytype_classification_results.csv
Shows trendiness metrics, temporal patterns, and symbol comparisons
============================================================================
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configuration
RESULTS_FILE = '/content/drive/MyDrive/backtest_results/daytype_classification_results.csv'
OUTPUT_DIR = '/content/drive/MyDrive/backtest_results'

print("="*80)
print("DAY TYPE RESULTS ANALYZER")
print("="*80)

# Load results
print(f"\nLoading results from: {RESULTS_FILE}")
df = pd.read_csv(RESULTS_FILE)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

print(f"✓ Loaded {len(df):,} trading days")
print(f"  Symbols: {df['Symbol'].nunique()}")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"  Columns: {', '.join(df.columns.tolist())}")

# ============================================================================
# OVERALL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

total_days = len(df)
day_type_counts = df['DayType'].value_counts()

print("\nDay Type Distribution (All Symbols):")
for day_type, count in day_type_counts.items():
    pct = (count / total_days) * 100
    print(f"  {day_type:40s}: {count:6,} days ({pct:5.1f}%)")

# ============================================================================
# PER-SYMBOL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PER-SYMBOL ANALYSIS")
print("="*80)

symbol_stats = []

for symbol in sorted(df['Symbol'].unique()):
    symbol_data = df[df['Symbol'] == symbol]

    total = len(symbol_data)
    dnp = len(symbol_data[symbol_data['DayType'] == 'DNP (Directional No Pullbacks)'])
    dwp = len(symbol_data[symbol_data['DayType'] == 'DWP (Directional w/ Pullbacks)'])
    range_day = len(symbol_data[symbol_data['DayType'] == 'RANGE DAY'])
    na = len(symbol_data[symbol_data['DayType'] == 'N/A'])

    directional = dnp + dwp

    symbol_stats.append({
        'Symbol': symbol,
        'Total_Days': total,
        'DNP': dnp,
        'DWP': dwp,
        'Range': range_day,
        'N/A': na,
        'Directional': directional,
        'DNP%': (dnp / total * 100) if total > 0 else 0,
        'DWP%': (dwp / total * 100) if total > 0 else 0,
        'Directional%': (directional / total * 100) if total > 0 else 0,
        'Range%': (range_day / total * 100) if total > 0 else 0
    })

stats_df = pd.DataFrame(symbol_stats)

print("\nSymbol Statistics:")
print(stats_df.to_string(index=False))

# ============================================================================
# TRENDINESS RANKING
# ============================================================================
print("\n" + "="*80)
print("TRENDINESS RANKING (sorted by Directional%)")
print("="*80)

ranking_df = stats_df[['Symbol', 'Total_Days', 'Directional%', 'DNP%', 'DWP%', 'Range%']].copy()
ranking_df = ranking_df.sort_values('Directional%', ascending=False)

print("\n" + ranking_df.to_string(index=False))

print("\n" + "-"*80)
print("Most Directional Symbols:")
top_5 = ranking_df.head(5)
for idx, row in top_5.iterrows():
    print(f"  {row['Symbol']:8s}: {row['Directional%']:5.1f}% directional ({row['DNP%']:4.1f}% DNP, {row['DWP%']:4.1f}% DWP)")

print("\nLeast Directional Symbols:")
bottom_5 = ranking_df.tail(5)
for idx, row in bottom_5.iterrows():
    print(f"  {row['Symbol']:8s}: {row['Directional%']:5.1f}% directional ({row['Range%']:4.1f}% Range)")

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("TEMPORAL ANALYSIS")
print("="*80)

# Add year/quarter/month columns
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.to_period('Q')
df['YearMonth'] = df['Date'].dt.to_period('M')

# Check if we have multi-year data
years = df['Year'].unique()
print(f"\nYears in dataset: {sorted(years)}")

if len(years) > 1:
    print("\nDirectional% by Year (All Symbols):")
    for year in sorted(years):
        year_data = df[df['Year'] == year]
        directional = len(year_data[year_data['DayType'].str.contains('Directional', na=False)])
        total = len(year_data)
        pct = (directional / total * 100) if total > 0 else 0
        print(f"  {year}: {pct:5.1f}% ({directional}/{total} days)")

# Monthly breakdown for most recent year
recent_year = max(years)
print(f"\nMonthly Breakdown for {recent_year}:")

recent_data = df[df['Year'] == recent_year]
monthly_stats = []

for month in sorted(recent_data['YearMonth'].unique()):
    month_data = recent_data[recent_data['YearMonth'] == month]
    total = len(month_data)
    directional = len(month_data[month_data['DayType'].str.contains('Directional', na=False)])
    dnp = len(month_data[month_data['DayType'] == 'DNP (Directional No Pullbacks)'])
    dwp = len(month_data[month_data['DayType'] == 'DWP (Directional w/ Pullbacks)'])
    range_day = len(month_data[month_data['DayType'] == 'RANGE DAY'])

    monthly_stats.append({
        'Month': str(month),
        'Total_Days': total,
        'Directional%': (directional / total * 100) if total > 0 else 0,
        'DNP%': (dnp / total * 100) if total > 0 else 0,
        'DWP%': (dwp / total * 100) if total > 0 else 0,
        'Range%': (range_day / total * 100) if total > 0 else 0
    })

monthly_df = pd.DataFrame(monthly_stats)
print(monthly_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Day Type Analysis', fontsize=16, fontweight='bold')

# 1. Overall Distribution (Pie Chart)
ax1 = axes[0, 0]
day_type_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
ax1.set_ylabel('')
ax1.set_title('Overall Day Type Distribution')

# 2. Directional% by Symbol (Bar Chart)
ax2 = axes[0, 1]
ranking_df_sorted = ranking_df.sort_values('Directional%', ascending=True)
ranking_df_sorted.plot(x='Symbol', y='Directional%', kind='barh', ax=ax2, legend=False, color='steelblue')
ax2.set_xlabel('Directional %')
ax2.set_title('Directional % by Symbol')
ax2.grid(axis='x', alpha=0.3)

# 3. Day Type Breakdown by Symbol (Stacked Bar)
ax3 = axes[1, 0]
breakdown_data = stats_df[['Symbol', 'DNP%', 'DWP%', 'Range%', 'N/A']].set_index('Symbol')
breakdown_data = breakdown_data.sort_values('DNP%', ascending=False)
breakdown_data.plot(kind='bar', stacked=True, ax=ax3,
                    color=['#2ecc71', '#3498db', '#e74c3c', '#95a5a6'])
ax3.set_ylabel('Percentage')
ax3.set_title('Day Type Breakdown by Symbol')
ax3.legend(['DNP', 'DWP', 'Range', 'N/A'])
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

# 4. DNP vs DWP Scatter
ax4 = axes[1, 1]
ax4.scatter(stats_df['DNP%'], stats_df['DWP%'], s=100, alpha=0.6, color='steelblue')
for idx, row in stats_df.iterrows():
    ax4.annotate(row['Symbol'], (row['DNP%'], row['DWP%']),
                fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
ax4.set_xlabel('DNP %')
ax4.set_ylabel('DWP %')
ax4.set_title('DNP vs DWP Distribution')
ax4.grid(alpha=0.3)

plt.tight_layout()

# Save figure
viz_file = os.path.join(OUTPUT_DIR, 'daytype_analysis_charts.png')
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualizations saved to: {viz_file}")
plt.show()

# ============================================================================
# TIME SERIES VISUALIZATION (if multi-month data)
# ============================================================================
if len(df['YearMonth'].unique()) > 3:
    print("\nGenerating time series analysis...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate directional% by month for each symbol
    time_series_data = []
    for symbol in df['Symbol'].unique():
        symbol_data = df[df['Symbol'] == symbol]
        for month in symbol_data['YearMonth'].unique():
            month_data = symbol_data[symbol_data['YearMonth'] == month]
            total = len(month_data)
            directional = len(month_data[month_data['DayType'].str.contains('Directional', na=False)])
            pct = (directional / total * 100) if total > 0 else 0

            time_series_data.append({
                'Symbol': symbol,
                'Month': str(month),
                'Directional%': pct
            })

    ts_df = pd.DataFrame(time_series_data)

    # Plot top 5 most directional symbols over time
    top_symbols = ranking_df.head(5)['Symbol'].tolist()

    for symbol in top_symbols:
        symbol_ts = ts_df[ts_df['Symbol'] == symbol].sort_values('Month')
        ax.plot(symbol_ts['Month'], symbol_ts['Directional%'], marker='o', label=symbol, linewidth=2)

    ax.set_xlabel('Month')
    ax.set_ylabel('Directional %')
    ax.set_title('Directional % Over Time (Top 5 Symbols)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    ts_file = os.path.join(OUTPUT_DIR, 'daytype_timeseries.png')
    plt.savefig(ts_file, dpi=150, bbox_inches='tight')
    print(f"✓ Time series chart saved to: {ts_file}")
    plt.show()

# ============================================================================
# EXPORT SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("EXPORTING SUMMARY STATISTICS")
print("="*80)

# Save summary stats
stats_file = os.path.join(OUTPUT_DIR, 'daytype_summary_statistics.csv')
stats_df.to_csv(stats_file, index=False)
print(f"✓ Summary statistics saved to: {stats_file}")

# Save ranking
ranking_file = os.path.join(OUTPUT_DIR, 'daytype_trendiness_ranking.csv')
ranking_df.to_csv(ranking_file, index=False)
print(f"✓ Trendiness ranking saved to: {ranking_file}")

# Download files
try:
    from google.colab import files
    print("\nDownloading files...")
    files.download(viz_file)
    files.download(stats_file)
    files.download(ranking_file)
    if len(df['YearMonth'].unique()) > 3:
        files.download(ts_file)
    print("✓ Files downloaded!")
except:
    print("Not in Colab - files saved to Drive only")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find most/least directional
most_directional = ranking_df.iloc[0]
least_directional = ranking_df.iloc[-1]

print(f"\n1. MOST DIRECTIONAL: {most_directional['Symbol']}")
print(f"   - {most_directional['Directional%']:.1f}% directional days")
print(f"   - {most_directional['DNP%']:.1f}% DNP (clean trends)")
print(f"   - {most_directional['DWP%']:.1f}% DWP (trends with pullbacks)")

print(f"\n2. LEAST DIRECTIONAL: {least_directional['Symbol']}")
print(f"   - {least_directional['Directional%']:.1f}% directional days")
print(f"   - {least_directional['Range%']:.1f}% range days")

# Calculate variance in directional%
directional_variance = stats_df['Directional%'].std()
print(f"\n3. CONSISTENCY:")
print(f"   - Standard deviation of Directional% across symbols: {directional_variance:.1f}%")
if directional_variance < 5:
    print("   - LOW variance: Most symbols behave similarly")
elif directional_variance > 10:
    print("   - HIGH variance: Significant differences between symbols")
else:
    print("   - MODERATE variance: Some variation between symbols")

# DNP vs DWP ratio
avg_dnp = stats_df['DNP%'].mean()
avg_dwp = stats_df['DWP%'].mean()
print(f"\n4. TREND QUALITY:")
print(f"   - Average DNP%: {avg_dnp:.1f}% (clean trends)")
print(f"   - Average DWP%: {avg_dwp:.1f}% (trends with pullbacks)")
print(f"   - Ratio: {avg_dnp/avg_dwp:.2f}:1" if avg_dwp > 0 else "   - Ratio: N/A")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print(f"  1. {viz_file}")
print(f"  2. {stats_file}")
print(f"  3. {ranking_file}")
if len(df['YearMonth'].unique()) > 3:
    print(f"  4. {ts_file}")
print("\nAll files saved to Google Drive and downloaded to your computer.")
print("="*80)
