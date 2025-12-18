"""
Credit Spread Signal Robustness Test

Tests the stability of credit spread signals across different parameter choices:
- MA smoothing periods
- Z-score lookback windows
- ROC calculation periods

Goal: Determine if findings are robust or curve-fitted
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT SPREAD ROBUSTNESS TEST")
print("=" * 80)

start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

print("\nDownloading data...")

raw = yf.download(['IEI', 'HYG', 'SPY'], start=start_date, end=end_date, progress=False)

# Extract data using the actual column structure: ('Price', 'Ticker')
data = pd.DataFrame({
    'IEI': raw[('Close', 'IEI')],
    'HYG': raw[('Close', 'HYG')],
    'SPY_Close': raw[('Close', 'SPY')],
    'SPY_Open': raw[('Open', 'SPY')]
})

data = data.dropna()
data['Spread'] = data['IEI'] / data['HYG']
data['Fwd3D'] = data['SPY_Close'].pct_change(3).shift(-3) * 100
data['Fwd5D'] = data['SPY_Close'].pct_change(5).shift(-5) * 100
data['Fwd10D'] = data['SPY_Close'].pct_change(10).shift(-10) * 100

print(f"✅ {data.index[0].date()} to {data.index[-1].date()} ({len(data)} days)\n")

# PARAMETERS
SPREAD_MA_PERIODS = [5, 10, 20, 30, 50]
Z_LOOKBACK_PERIODS = [60, 90, 120, 180, 252, 360]
MA_SLOW_PERIODS = [30, 50]
ROC_PERIODS = [1, 2, 3, 5, 10]

print("PARAMETER SPACE")
print(f"  MA: {SPREAD_MA_PERIODS}")
print(f"  Z: {Z_LOOKBACK_PERIODS}")
print(f"  Slow MA: {MA_SLOW_PERIODS}")
print(f"  ROC: {ROC_PERIODS}\n")

# INDICATORS
print("Calculating indicators...")

for ma in SPREAD_MA_PERIODS:
    data[f'SpreadMA{ma}'] = data['Spread'].rolling(ma).mean()

for ma in SPREAD_MA_PERIODS:
    for z in Z_LOOKBACK_PERIODS:
        col = f'SpreadMA{ma}'
        mean = data[col].rolling(z).mean()
        std = data[col].rolling(z).std()
        data[f'Z_MA{ma}_LB{z}'] = (data[col] - mean) / std

for roc in ROC_PERIODS:
    data[f'ROC{roc}D'] = data['Spread'].pct_change(roc) * 100

for slow in MA_SLOW_PERIODS:
    data[f'MA10_Above_MA{slow}'] = (data['SpreadMA10'] > data[f'SpreadMA{slow}']).astype(int)

print(f"✅ Done\n")

# TEST SIGNALS
results = []

print("TESTING SIGNALS\n")

# Test 1
print("[1/4] Z-Score [0.0-0.25]...", end=' ')
c = 0
for ma in SPREAD_MA_PERIODS:
    for z in Z_LOOKBACK_PERIODS:
        mask = (data[f'Z_MA{ma}_LB{z}'] >= 0.0) & (data[f'Z_MA{ma}_LB{z}'] < 0.25)
        if mask.sum() >= 30:
            for fwd in ['Fwd3D', 'Fwd5D', 'Fwd10D']:
                rets = data.loc[mask, fwd].dropna()
                if len(rets) >= 30:
                    c += 1
                    results.append({
                        'Signal': 'ZScore[0-0.25]',
                        'MA': ma, 'Z': z, 'ROC': None,
                        'Hold': fwd[3:4],
                        'Ret': rets.mean(),
                        'Sharpe': rets.mean()/rets.std() if rets.std()>0 else 0,
                        'Win': (rets>0).mean()*100,
                        'N': len(rets)
                    })
print(f"{c} combos")

# Test 2
print("[2/4] Extreme Wide → Tight...", end=' ')
c = 0
for ma in SPREAD_MA_PERIODS:
    for z in Z_LOOKBACK_PERIODS:
        for roc in ROC_PERIODS:
            mask = (data[f'Z_MA{ma}_LB{z}']>=1.0) & (data[f'Z_MA{ma}_LB{z}']<1.5) & (data[f'ROC{roc}D']<-0.3)
            if mask.sum() >= 10:
                for fwd in ['Fwd3D', 'Fwd5D', 'Fwd10D']:
                    rets = data.loc[mask, fwd].dropna()
                    if len(rets) >= 10:
                        c += 1
                        results.append({
                            'Signal': 'ExtremeWide→Tight',
                            'MA': ma, 'Z': z, 'ROC': roc,
                            'Hold': fwd[3:4],
                            'Ret': rets.mean(),
                            'Sharpe': rets.mean()/rets.std() if rets.std()>0 else 0,
                            'Win': (rets>0).mean()*100,
                            'N': len(rets)
                        })
print(f"{c} combos")

# Test 3
print("[3/4] Strong ROC [1.0-2.0]...", end=' ')
c = 0
for roc in ROC_PERIODS:
    mask = (data[f'ROC{roc}D']>=1.0) & (data[f'ROC{roc}D']<2.0)
    if mask.sum() >= 20:
        for fwd in ['Fwd3D', 'Fwd5D', 'Fwd10D']:
            rets = data.loc[mask, fwd].dropna()
            if len(rets) >= 20:
                c += 1
                results.append({
                    'Signal': 'StrongROC[1.0-2.0]',
                    'MA': None, 'Z': None, 'ROC': roc,
                    'Hold': fwd[3:4],
                    'Ret': rets.mean(),
                    'Sharpe': rets.mean()/rets.std() if rets.std()>0 else 0,
                    'Win': (rets>0).mean()*100,
                    'N': len(rets)
                })
print(f"{c} combos")

# Test 4
print("[4/4] MA Cross + Tight...", end=' ')
c = 0
for slow in MA_SLOW_PERIODS:
    for z in Z_LOOKBACK_PERIODS:
        data['FC'] = (data[f'MA10_Above_MA{slow}']==1) & (data[f'MA10_Above_MA{slow}'].shift(1)==0)
        data['CA'] = sum(data['FC'].shift(i) for i in range(1,6))
        mask = (data['CA']>0) & (data[f'Z_MA10_LB{z}']<0)
        if mask.sum() >= 20:
            for fwd in ['Fwd5D', 'Fwd10D']:
                rets = data.loc[mask, fwd].dropna()
                if len(rets) >= 20:
                    c += 1
                    results.append({
                        'Signal': 'MACross+Tight',
                        'MA': slow, 'Z': z, 'ROC': None,
                        'Hold': fwd[3:5],
                        'Ret': rets.mean(),
                        'Sharpe': rets.mean()/rets.std() if rets.std()>0 else 0,
                        'Win': (rets>0).mean()*100,
                        'N': len(rets)
                    })
print(f"{c} combos\n")

# ANALYZE
df = pd.DataFrame(results)

print("="*80)
print(f"RESULTS: {len(df)} total combinations\n")

for sig in df['Signal'].unique():
    d = df[df['Signal']==sig]
    med = d['Sharpe'].median()
    pct = (d['Sharpe']>1.5).mean()*100

    print(f"{sig}")
    print(f"  Combinations: {len(d)}")
    print(f"  Sharpe: mean={d['Sharpe'].mean():.2f}, med={med:.2f}, std={d['Sharpe'].std():.2f}")
    print(f"  Robustness: {(d['Sharpe']>0).mean()*100:.0f}% positive, {pct:.0f}% good (>1.5)")

    if med>2.0 and pct>60:
        verdict = "✅ HIGHLY ROBUST"
    elif med>1.0 and pct>40:
        verdict = "✅ ROBUST"
    elif med>0.5 and pct>20:
        verdict = "⚠️ MODERATE"
    else:
        verdict = "❌ WEAK"
    print(f"  Verdict: {verdict}")

    print(f"\n  Top 3:")
    top = d.nlargest(3,'Sharpe')[['MA','Z','ROC','Hold','Sharpe','Ret','Win']]
    for _, r in top.iterrows():
        print(f"    MA={r['MA']}, Z={r['Z']}, ROC={r['ROC']}, Hold={r['Hold']}D: Sharpe={r['Sharpe']:.2f}, Ret={r['Ret']:.2f}%, Win={r['Win']:.0f}%")
    print()

df.to_csv('credit_spread_robustness_results.csv', index=False)
print("="*80)
print("✅ Saved: credit_spread_robustness_results.csv")
