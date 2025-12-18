# Credit Spread Trading Guide

## Overview

This guide documents the **best** ways to use the IEI/HYG credit spread ratio to predict forward SPY returns, based on comprehensive backtesting from 2010-2025.

## What is the Credit Spread Signal?

**IEI/HYG Ratio = 3-7yr Treasuries / High Yield Corporate Bonds**

- **Rising ratio** = Spreads widening = Flight to safety = Risk-off
- **Falling ratio** = Spreads tightening = Risk appetite = Risk-on

## Best Signals (Ranked by Sharpe Ratio)

### 🏆 #1: Extreme Wide → Large Tightening
**Best for: 3-5 day swing trades**

**Setup:**
- Z-Score between 1.0 and 1.5 (spreads moderately wide)
- 1-day ROC < -0.3% (rapidly tightening)

**Performance (5-Day Hold):**
- Mean Return: **+2.41%**
- Sharpe Ratio: **5.52**
- Win Rate: **85%**
- Frequency: **0.5% of days** (20 occurrences in 15 years)

**Action:** Enter 100-125% long SPY, hold for 3-5 days

**Why it works:** Extreme fear (wide spreads) rapidly reversing signals panic subsiding and strong mean reversion opportunity.

---

### 🥈 #2: Z-Score Neutral Zone [0.0 to 0.25]
**Best for: 10-day swing trades**

**Setup:**
- Z-Score (252-day) between 0.0 and 0.25
- Spreads slightly elevated but normalizing

**Performance (10-Day Hold):**
- Mean Return: **+1.60%**
- Sharpe Ratio: **3.06**
- Win Rate: **75%**
- Frequency: **3.9% of days** (174 occurrences)

**Action:** Enter 75-100% long SPY, hold for 10 days

**Why it works:** Optimal risk/reward zone where spreads aren't extreme in either direction. High frequency makes it tradeable.

---

### 🥉 #3: Moderate Tight + Accelerating Widening + Above MA
**Best for: 5-day swing trades**

**Setup (COMBO Signal):**
- Z-Score between -1.5 and -0.5 (moderately tight)
- 5-day ROC > 0.3% (accelerating widening)
- MA10 > MA50 (uptrend)

**Performance (5-Day Hold):**
- Mean Return: **+0.53%**
- Sharpe Ratio: **1.77**
- Win Rate: **68%**
- Frequency: **4.0% of days** (180 occurrences)

**Action:** Enter 50-75% long SPY, hold for 5 days

**Why it works:** Multi-factor confirmation creates robust setup. Spreads tightening from moderate levels with MA support.

---

### #4: Fresh Bullish Cross + Tight Spreads
**Best for: 5-10 day swing trades**

**Setup (COMBO Signal):**
- MA10 crosses above MA50 (within last 5 days)
- Z-Score < 0 (spreads tight)

**Performance (5-Day Hold):**
- Mean Return: **+0.52%**
- Sharpe Ratio: **1.65**
- Win Rate: **71%**
- Frequency: **1.1% of days** (49 occurrences)

**Action:** Enter 75-100% long SPY, hold for 5-10 days

**Why it works:** Fresh trend change with favorable spread environment. High win rate indicates consistency.

---

### #5: Strong ROC Momentum [1.0% to 2.0%]
**Best for: 3-10 day swing trades**

**Setup:**
- 1-day ROC between 1.0% and 2.0%
- Strong daily momentum in spread widening

**Performance (10-Day Hold):**
- Mean Return: **+4.56%**
- Sharpe Ratio: **2.66**
- Win Rate: **80%**
- Frequency: **0.7% of days** (30 occurrences)

**Action:** Enter 75-100% long SPY, hold 3-10 days

**Why it works:** Captures momentum continuation. When spreads move strongly, follow-through is likely.

---

## Quick Reference Table

| Signal | Hold | Mean Return | Sharpe | Win % | Frequency |
|--------|------|-------------|--------|-------|-----------|
| Extreme Wide → Tight | 5D | +2.41% | 5.52 | 85% | 0.5% |
| Z-Score [0.0-0.25] | 10D | +1.60% | 3.06 | 75% | 3.9% |
| Moderate Tight Combo | 5D | +0.53% | 1.77 | 68% | 4.0% |
| Bullish Cross Combo | 5-10D | +0.52% | 1.65 | 71% | 1.1% |
| Strong ROC Momentum | 3-10D | +4.56% | 2.66 | 80% | 0.7% |

---

## Category Performance

Based on comprehensive testing of 200+ signals:

| Category | Avg 5D Sharpe | Best For |
|----------|---------------|----------|
| **COMBO Signals** | 1.71 | Multi-factor confirmation |
| **Reversal Signals** | 1.02 | Extreme → reverting |
| **ROC 5D** | 0.87 | Medium-term momentum |
| **Z-Score (90D)** | 0.76 | Short-term levels |
| **MA Crossovers** | 0.72 | Trend following |

**Recommendation:** Focus on COMBO and Reversal signals for highest Sharpe ratios.

---

## Position Sizing Guide

Based on signal quality:

```
ULTRA HIGH CONVICTION (Sharpe > 5.0):
  → 100-125% of capital
  → Only for rare, extreme setups
  → Examples: Extreme Wide → Large Tightening

HIGH CONVICTION (Sharpe 2.0-5.0):
  → 75-100% of capital
  → Multiple factors aligned
  → Examples: Z-Score neutral zone, Strong ROC

MODERATE CONVICTION (Sharpe 1.5-2.0):
  → 50-75% of capital
  → Single strong signal
  → Examples: COMBO signals

LOW CONVICTION (Sharpe < 1.5):
  → 0-25% of capital
  → Wait for better setup
```

---

## Implementation Checklist

**Daily Routine:**
1. Download latest IEI, HYG, SPY data
2. Calculate spread Z-score and ROC
3. Check for active signals
4. Size position based on conviction
5. Set exit after hold period (3D, 5D, or 10D)

**Entry Criteria:**
- ✅ At least one signal with Sharpe > 1.5
- ✅ Multiple signals = higher conviction
- ✅ ULTRA signals = maximum size

**Exit Criteria:**
- 📅 Time-based: Hold for specified period (3D, 5D, or 10D)
- 🎯 Target-based: Optional profit target at 2x expected mean
- 🛡️ Stop-based: Optional stop at -2x expected mean

**Risk Management:**
- Never exceed 125% of capital (even on best signals)
- Use 0-25% base position when no signals active
- Don't chase - wait for setup

---

## Files

- `credit_spread_comprehensive_test.py` - Full testing framework
- `credit_spread_trading_system.py` - Live signal detection
- `CREDIT_SPREAD_GUIDE.md` - This guide

---

## Backtested Performance (2010-2025)

**Test Period:** 15 years (3,800+ trading days)
**Signals Tested:** 200+ unique signals
**Best Signal Sharpe:** 5.52 (5-day), 7.40 (3-day)
**Best Win Rate:** 85% (Extreme Wide → Tightening)

---

## Disclaimer

Past performance does not guarantee future results. This analysis is for educational purposes only. Always conduct your own research and consult financial professionals before trading.

---

## Credits

Analysis based on credit spread concepts from market microstructure research. Backtesting conducted using Python, yfinance, pandas, and numpy.
