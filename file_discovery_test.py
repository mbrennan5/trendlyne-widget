"""
============================================================================
FILE DISCOVERY DIAGNOSTIC SCRIPT
============================================================================
This script helps diagnose file discovery issues by showing:
- All files in your StockData directory
- Which files match the 30-minute pattern
- Symbol mapping
- File sizes and row counts
- Date ranges in each file

USAGE (Google Colab):
1. Copy this entire code block
2. Paste into a Google Colab cell
3. Run the cell
4. Review the detailed output

============================================================================
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import pandas as pd
from pathlib import Path
import os

# Configuration
DATA_DIR = '/content/drive/MyDrive/StockData'

print("=" * 80)
print("FILE DISCOVERY DIAGNOSTIC")
print("=" * 80)
print(f"\nSearching in: {DATA_DIR}\n")

# Check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: Directory does not exist: {DATA_DIR}")
    print("\nPlease verify the path to your StockData folder.")
else:
    print(f"✓ Directory exists: {DATA_DIR}\n")

    # List ALL files in directory
    print("=" * 80)
    print("STEP 1: ALL FILES IN DIRECTORY")
    print("=" * 80)

    all_files = list(Path(DATA_DIR).glob('*'))

    if not all_files:
        print("❌ No files found in directory!")
    else:
        print(f"Found {len(all_files)} total files/folders:\n")
        for i, f in enumerate(sorted(all_files), 1):
            file_type = "DIR" if f.is_dir() else "FILE"
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"  {i:3d}. [{file_type:4s}] {f.name:50s} ({size_mb:.2f} MB)")

    # Find files matching 30Min pattern
    print("\n" + "=" * 80)
    print("STEP 2: FILES MATCHING '*30Min*.csv' PATTERN")
    print("=" * 80)

    matching_files = list(Path(DATA_DIR).glob('*30Min*.csv'))

    if not matching_files:
        print("❌ No files matching '*30Min*.csv' pattern found!")
        print("\nThis means the file naming pattern may be different.")
        print("Check the file names above and update the pattern if needed.")
    else:
        print(f"Found {len(matching_files)} files matching pattern:\n")

        symbol_map = {}

        for i, file_path in enumerate(sorted(matching_files), 1):
            filename = file_path.stem
            # Extract symbol (assumes format: SYMBOL_30Min or SYMBOL-30Min)
            symbol = filename.split('_')[0].split('-')[0].upper()

            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            print(f"  {i:3d}. {file_path.name:50s} → Symbol: {symbol:6s} ({file_size_mb:.2f} MB)")

            symbol_map[symbol] = file_path

        # Symbol summary
        print("\n" + "=" * 80)
        print("STEP 3: SYMBOL MAPPING")
        print("=" * 80)
        print(f"\n{len(symbol_map)} unique symbols found:\n")
        print(f"  {', '.join(sorted(symbol_map.keys()))}")

        # Detailed file analysis
        print("\n" + "=" * 80)
        print("STEP 4: DETAILED FILE ANALYSIS")
        print("=" * 80)

        for symbol, file_path in sorted(symbol_map.items()):
            print(f"\n{'─' * 80}")
            print(f"Symbol: {symbol}")
            print(f"File: {file_path.name}")
            print(f"{'─' * 80}")

            try:
                # Quick read to get row count and date range
                df = pd.read_csv(file_path)

                print(f"  Total rows: {len(df):,}")
                print(f"  Columns: {', '.join(df.columns.tolist())}")

                # Try to parse datetime
                if 't' in df.columns:
                    df['datetime'] = pd.to_datetime(df['t'], utc=True)
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                else:
                    df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)

                min_date = df['datetime'].min()
                max_date = df['datetime'].max()

                print(f"  Date range: {min_date.date()} to {max_date.date()}")
                print(f"  Duration: {(max_date - min_date).days} days")

                # Count unique dates
                unique_dates = df['datetime'].dt.date.nunique()
                print(f"  Unique dates: {unique_dates:,}")

                # Show first few rows of datetime column
                print(f"\n  First 3 timestamps:")
                for idx, dt in enumerate(df['datetime'].head(3), 1):
                    print(f"    {idx}. {dt}")

                print(f"\n  Last 3 timestamps:")
                for idx, dt in enumerate(df['datetime'].tail(3), 1):
                    print(f"    {idx}. {dt}")

            except Exception as e:
                print(f"  ❌ ERROR reading file: {e}")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nThis diagnostic shows:")
print("  1. All files in your StockData directory")
print("  2. Which files match the 30-minute pattern")
print("  3. Symbol-to-file mapping")
print("  4. Row counts and date ranges for each file")
print("\nUse this information to verify:")
print("  • File naming matches expected pattern")
print("  • Symbols are being extracted correctly")
print("  • CSV files contain expected date ranges")
print("  • No duplicate symbols with different files")
print("=" * 80)
