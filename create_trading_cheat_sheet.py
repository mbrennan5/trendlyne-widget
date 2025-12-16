"""
GDT TRADING CHEAT SHEET - PDF GENERATOR
Creates a printable PDF cheat sheet with all trading rules for the top 3 setups
Run this in Google Colab to generate the PDF
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Create PDF with multiple pages
pdf_filename = 'GDT_Trading_Cheat_Sheet.pdf'

try:
    pdf_path = f'/content/drive/MyDrive/backtest_results/{pdf_filename}'
except:
    pdf_path = pdf_filename

pdf = PdfPages(pdf_path)

# Color scheme
GREEN = '#2ecc71'
RED = '#e74c3c'
ORANGE = '#f39c12'
BLUE = '#3498db'
DARK_GRAY = '#34495e'
LIGHT_GRAY = '#ecf0f1'

# ============================================================================
# PAGE 1: OVERVIEW & GDT CYCLE
# ============================================================================
fig1 = plt.figure(figsize=(8.5, 11))
fig1.suptitle('GDT TRADING CHEAT SHEET', fontsize=24, fontweight='bold', y=0.97)

# Add subtitle
fig1.text(0.5, 0.93, 'Momentum vs Exhaustion: Same-Day Prediction Using GDT Day# + Gap Behavior',
          ha='center', fontsize=11, style='italic')

ax1 = fig1.add_axes([0.1, 0.75, 0.8, 0.15])
ax1.axis('off')

# GDT Cycle Overview
ax1.text(0.5, 0.9, 'GDT DAY# CYCLE OVERVIEW', ha='center', fontsize=16, fontweight='bold',
         transform=ax1.transAxes)

# Buy Cycle
buy_cycle_text = """
BULLISH CYCLE:
• Buy_Day1: Momentum flip (neutral +2-4% edge)
• Buy_Day2: PEAK MOMENTUM (+8-12% edge) ⭐ TRADE HERE
• Buy_Day3: Continuation (neutral +3-5% edge)
• Buy_Day4: EXHAUSTION (-5-8% edge) ❌ FADE/AVOID
"""

sell_cycle_text = """
BEARISH CYCLE:
• Sell_Day1-3: Neutral edges (-2-4%)
• Sell_Day4: EXHAUSTION ❌ AVOID
"""

ax1.text(0.05, 0.65, buy_cycle_text, fontsize=10, va='top', family='monospace',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.1, edgecolor=GREEN, linewidth=2))

ax1.text(0.55, 0.65, sell_cycle_text, fontsize=10, va='top', family='monospace',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=RED, alpha=0.1, edgecolor=RED, linewidth=2))

# Key Concepts
ax2 = fig1.add_axes([0.1, 0.50, 0.8, 0.22])
ax2.axis('off')

ax2.text(0.5, 0.95, 'KEY CONCEPTS', ha='center', fontsize=14, fontweight='bold',
         transform=ax2.transAxes)

concepts_text = """
SAME-DAY PREDICTION LOGIC (All predictors known at this morning's open):
├─ YESTERDAY's GDT Day# → Where are we in the momentum cycle?
├─ YESTERDAY's Range Size → Was it narrow/wide/very wide? (vs 20-day ADR)
├─ YESTERDAY's Closing Position → Did it close in top/bottom quarter of range?
└─ THIS MORNING's Gap → GapStat size, direction vs trend (with/against)
   → Predicts: TODAY's directional outcome (by close)

GAP BEHAVIOR:
• Gap AGAINST Trend = Shakeout reversal (works BEST on Buy_Day2)
  Example: 5-day SMA up, stock gaps DOWN → reversal setup
• Gap WITH Trend = Momentum continuation (works early cycle, FAILS on Day 4)

CLOSING RANGE %:
• 0% = closed at low of day
• 100% = closed at high of day
• Best setups average 70-75% (closes near highs)
"""

ax2.text(0.05, 0.80, concepts_text, fontsize=9, va='top', family='monospace',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor=LIGHT_GRAY, edgecolor=DARK_GRAY, linewidth=1))

# Top 3 Setups Summary
ax3 = fig1.add_axes([0.1, 0.10, 0.8, 0.38])
ax3.axis('off')

ax3.text(0.5, 0.98, 'TOP 3 TRADE SETUPS (DETAILED RULES ON NEXT PAGE)',
         ha='center', fontsize=14, fontweight='bold', transform=ax3.transAxes)

# Setup table
table_data = [
    ['Setup', 'Requirements', 'Edge', 'Avg Move', 'Action'],
    ['#1: Shakeout\nReversal',
     'Buy_Day2 + Gap Against Trend\n+ Yest Close Top Quarter',
     '+10-12%', '+2.0-2.5%', '🟢 LONG'],
    ['#2: Big Gap\nStrength',
     'Buy_Day2 + GapStat >2.0\n+ Yest Close Top Quarter',
     '+10-11%', '+1.8-2.3%', '🟢 LONG'],
    ['#3: Volatility\nExpansion',
     'Buy_Day2 + GapStat >2.0\n+ Yest Wide Range',
     '+8-10%', '+1.6-2.0%', '🟢 LONG'],
    ['FADE Signal',
     'Buy_Day4 + Gap With Trend',
     '-5-8%', '-0.5-+0.2%', '🔴 FADE'],
]

table = ax3.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.18, 0.40, 0.12, 0.12, 0.10],
                 bbox=[0.02, 0.02, 0.96, 0.88])

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.8)

# Style header
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor(DARK_GRAY)
    cell.set_text_props(weight='bold', color='white')
    cell.set_height(0.08)

# Color rows
for i in [1, 2, 3]:
    for j in range(5):
        table[(i, j)].set_facecolor('#d5f4e6')

for j in range(5):
    table[(4, j)].set_facecolor('#fadbd8')

# Add footer
fig1.text(0.5, 0.04, '⭐ Focus on Buy_Day2 | ⚠️ Avoid Buy_Day4 | 📊 All stats from same-day prediction backtest',
          ha='center', fontsize=9, style='italic',
          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

fig1.text(0.5, 0.01, 'Page 1 of 3', ha='center', fontsize=8, color='gray')

pdf.savefig(fig1, bbox_inches='tight')
plt.close()

# ============================================================================
# PAGE 2: DETAILED TRADING RULES
# ============================================================================
fig2 = plt.figure(figsize=(8.5, 11))
fig2.suptitle('DETAILED TRADING RULES', fontsize=20, fontweight='bold', y=0.97)

# Setup #1
ax1 = fig2.add_axes([0.08, 0.72, 0.84, 0.22])
ax1.axis('off')

setup1_text = """
SETUP #1: "THE SHAKEOUT REVERSAL"
Gap Against Trend + Yesterday Close Top Quarter (Buy_Day2)

PRE-MARKET CHECKLIST (Before 9:30am):
✓ Yesterday was Buy_Day1 (GDT momentum flip to bullish)
✓ Yesterday closed in top 25% of its range (Close_Range% ≥ 75%)
✓ Today is Buy_Day2 (2nd day of bullish GDT cycle)
✓ Stock has gap AGAINST trend (5-SMA up → gap DOWN, or 5-SMA down → gap UP)
✓ Gap size ≥ 0.5%

ENTRY:
• Window: First 30-60 minutes (9:30-10:30am)
• Trigger Options:
  - Aggressive: First 5-min green candle after open + price above VWAP
  - Conservative: Wait for gap fill attempt to fail, enter on reversal
  - VWAP Bounce: Wait for pullback to VWAP, enter on bounce with volume
• Position Size: 1-2% account risk (2-3% if GapStat also >2.0)

STOPS:
• Initial: Open - (1.5 × ATR) OR 0.3% below gap low OR VWAP - 0.5%
• Maximum: -1.5% from entry
• Move to breakeven at +1% profit
• Trail at breakeven +0.5% once at +2% profit

PROFIT TARGETS:
• Target 1 (50%): +1.5% from open
• Target 2 (30%): +2.5% from open
• Target 3 (20%): Hold to close (expect 70-75% closing range)
• Time Stop: Exit 50% by 11:30am if no follow-through

STATS: Edge +10-12% | Avg Move +2.0-2.5% | Close Range 70-75%
"""

ax1.text(0.02, 0.98, setup1_text, fontsize=8, va='top', family='monospace',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.1, edgecolor=GREEN, linewidth=2))

# Setup #2
ax2 = fig2.add_axes([0.08, 0.47, 0.84, 0.22])
ax2.axis('off')

setup2_text = """
SETUP #2: "THE BIG GAP STRENGTH"
GapStat >2.0 + Yesterday Close Top Quarter (Buy_Day2)

PRE-MARKET CHECKLIST:
✓ Yesterday was Buy_Day1
✓ Yesterday closed in top 25% of its range
✓ Today is Buy_Day2
✓ GapStat > 2.0 (normalized gap statistic) - use absolute gap >1.5-2% as proxy

ENTRY:
• Window: First 15-45 minutes (9:30-10:15am)
• Trigger:
  - Gap UP >2.0: Wait for first pullback, enter when price reclaims VWAP
  - Gap DOWN >2.0: Enter on first reversal candle (green after red), must hold gap low
• Volume confirmation: Entry volume > 5-min average
• Position Size: 1.5-2.5% risk (reduce to 1-1.5% if GapStat >2.5)

STOPS:
• Gap Up: 1% below entry or VWAP (whichever tighter)
• Gap Down: 0.5% below gap low
• Maximum: -2% from entry (big gaps = big swings)
• Breakeven at +1.2%, trail +0.3% below each 15-min higher low after +2%

TARGETS:
• Target 1 (40%): +1.5%
• Target 2 (40%): +2.3% (avg expected move)
• Target 3 (20%): Hold to close
• Exit 50% by 12pm if no movement, exit all by 3pm if profit <1%

STATS: Edge +10-11% | Avg Move +1.8-2.3% | Close Range 68-72%
"""

ax2.text(0.02, 0.98, setup2_text, fontsize=8, va='top', family='monospace',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.1, edgecolor=GREEN, linewidth=2))

# Setup #3
ax3 = fig2.add_axes([0.08, 0.22, 0.84, 0.22])
ax3.axis('off')

setup3_text = """
SETUP #3: "THE VOLATILITY EXPANSION"
GapStat >2.0 + Yesterday Wide Range (Buy_Day2)

PRE-MARKET CHECKLIST:
✓ Yesterday was Buy_Day1
✓ Yesterday had Wide or Very Wide range (Range_vs_ADR > 1.3)
  Calculate: Yesterday's Range% / 20-day ADR
✓ Today is Buy_Day2
✓ GapStat > 2.0

ENTRY:
• Window: First 60 minutes (9:30-10:30am)
• Wait for opening range (first 15 min) to establish
• Enter on OR-high breakout with volume >1.5x average
• Entry: Limit at OR-high + $0.15, must hold for 5 minutes
• Position Size: 1-1.5% risk (REDUCED - high volatility setup)

STOPS (WIDER - volatility setup):
• OR-low - 0.5% OR Entry - (2.0 × ATR)
• Maximum: -2.5% from entry
• Breakeven at +1.5%, trail 1 ATR below price after +2.5%

TARGETS:
• Target 1 (30%): +1.5%
• Target 2 (30%): +2.0%
• Target 3 (20%): +3.0% (volatility extension)
• Target 4 (20%): Runner to close
• If choppy by 11am, exit 50%; if stalling at +1%, exit 70% and trail

STATS: Edge +8-10% | Avg Move +1.6-2.0% | Close Range 66-70%
"""

ax3.text(0.02, 0.98, setup3_text, fontsize=8, va='top', family='monospace',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor=BLUE, alpha=0.1, edgecolor=BLUE, linewidth=2))

# Do Not Trade / Fade Setup
ax4 = fig2.add_axes([0.08, 0.08, 0.84, 0.12])
ax4.axis('off')

fade_text = """
⚠️ FADE SETUP (DO NOT LONG - Consider Fading/Shorting):
BUY_DAY4 + GAP WITH TREND = EXHAUSTION

• This is LATE in the bullish cycle (Day 4 of 4)
• Gap WITH trend (5-SMA up + gap UP) = failed breakout setup
• Edge: -5 to -8% (NEGATIVE for longs!)
• Price action: Choppy, closes mid-range (45-50%), weak/negative move from open
• Action: AVOID longs, consider fading or shorting the gap
"""

ax4.text(0.02, 0.98, fade_text, fontsize=9, va='top', family='monospace',
         transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=RED, alpha=0.15, edgecolor=RED, linewidth=3))

fig2.text(0.5, 0.01, 'Page 2 of 3', ha='center', fontsize=8, color='gray')

pdf.savefig(fig2, bbox_inches='tight')
plt.close()

# ============================================================================
# PAGE 3: RISK MANAGEMENT & QUICK REFERENCE
# ============================================================================
fig3 = plt.figure(figsize=(8.5, 11))
fig3.suptitle('RISK MANAGEMENT & QUICK REFERENCE', fontsize=20, fontweight='bold', y=0.97)

# Risk Management Rules
ax1 = fig3.add_axes([0.08, 0.72, 0.84, 0.22])
ax1.axis('off')

risk_text = """
RISK MANAGEMENT RULES

GENERAL RULES FOR ALL SETUPS:
❌ DO NOT TRADE IF:
  • SPY trending AGAINST your direction (check SPY first!)
  • Market choppy/no trend (VIX >25 and rising)
  • Stock has major news/earnings (unless experienced)
  • Missed entry window (don't chase after 11am)
  • Sample size <30 in your backtest

✅ ENHANCE EDGE:
  • All setups work BEST on Buy_Day2 specifically
  • Combine filters (Gap Against + GapStat >2 + Yest Close Top Qtr = STACK EDGE)
  • Trade in direction of SPY trend
  • Higher volume at entry = better probability
  • Focus on symbols with most samples (NVDA, TSLA, AMD, SMCI)

POSITION SIZING & LIMITS:
  • Max 3 positions from these setups per day
  • Max 5% total account risk across all positions
  • Daily stop: -2% account = DONE for the day
  • Scale into winners, cut losers FAST

CORRELATION WARNING:
  • High Beta Tech stocks are correlated
  • Don't take 3 positions all in NVDA, AMD, TSLA on same setup
  • Diversify across different symbols or different setup types
"""

ax1.text(0.02, 0.98, risk_text, fontsize=9, va='top', family='monospace',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=ORANGE, alpha=0.1, edgecolor=ORANGE, linewidth=2))

# Quick Reference Table
ax2 = fig3.add_axes([0.08, 0.42, 0.84, 0.27])
ax2.axis('off')

ax2.text(0.5, 0.98, 'QUICK REFERENCE TABLE', ha='center', fontsize=14, fontweight='bold',
         transform=ax2.transAxes)

quick_ref = [
    ['Setup', 'Best Entry', 'Stop', 'Target 1', 'Target 2', 'Runner'],
    ['#1 Shakeout', '1st green/VWAP', '-1.5%', '+1.5% (50%)', '+2.5% (30%)', 'Close (20%)'],
    ['#2 Big Gap', '1st pullback', '-2%', '+1.5% (40%)', '+2.3% (40%)', 'Close (20%)'],
    ['#3 Vol Expand', 'OR-high break', '-2.5%', '+1.5% (30%)', '+2% (30%)', '+3% (40%)'],
]

table2 = ax2.table(cellText=quick_ref, cellLoc='center', loc='center',
                  bbox=[0.02, 0.02, 0.96, 0.88])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2.2)

for i in range(6):
    cell = table2[(0, i)]
    cell.set_facecolor(DARK_GRAY)
    cell.set_text_props(weight='bold', color='white')

for i in [1, 2, 3]:
    for j in range(6):
        table2[(i, j)].set_facecolor(LIGHT_GRAY)

# Pre-Market Checklist
ax3 = fig3.add_axes([0.08, 0.18, 0.40, 0.22])
ax3.axis('off')

ax3.text(0.5, 0.98, '🌅 PRE-MARKET CHECKLIST', ha='center', fontsize=12, fontweight='bold',
         transform=ax3.transAxes)

checklist_text = """
EVERY MORNING (8:00-9:30am):

□ Calculate GDT Day# for watchlist
  (Based on yesterday's close)

□ Identify Buy_Day2 symbols

□ Check yesterday's stats for each:
  • Closing range % (top quarter?)
  • Range size (wide/narrow?)

□ Check pre-market gaps:
  • Gap size and direction
  • Gap vs 5-day SMA trend

□ Shortlist Buy_Day2 + qualifying gaps

□ Check SPY trend direction

□ Prepare entry orders
"""

ax3.text(0.05, 0.88, checklist_text, fontsize=8, va='top', family='monospace',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor=LIGHT_GRAY, edgecolor=DARK_GRAY, linewidth=1))

# Trade Log Template
ax4 = fig3.add_axes([0.52, 0.18, 0.40, 0.22])
ax4.axis('off')

ax4.text(0.5, 0.98, '📝 TRADE LOG (Track Each)', ha='center', fontsize=12, fontweight='bold',
         transform=ax4.transAxes)

log_text = """
RECORD KEEPING:

For each trade, log:
□ Symbol & Date
□ Setup used (#1, #2, or #3)
□ Entry price & time
□ Stop price
□ Exit price(s) & time(s)
□ % change from open at exit
□ Closing range % (if held to close)
□ Did it match expected behavior?
□ What went right/wrong?

REVIEW WEEKLY:
• Win rate by setup type
• Avg R:R by setup
• Best/worst symbols
• Time of day patterns
"""

ax4.text(0.05, 0.88, log_text, fontsize=8, va='top', family='monospace',
         transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=LIGHT_GRAY, edgecolor=DARK_GRAY, linewidth=1))

# Final Notes
ax5 = fig3.add_axes([0.08, 0.04, 0.84, 0.12])
ax5.axis('off')

notes_text = """
📊 FINAL NOTES:
• These edges come from same-day prediction backtests (all predictors known at open)
• Focus on Buy_Day2 for best results (+8-12% edge vs baseline)
• Avoid Buy_Day4 - late cycle exhaustion leads to fade/chop
• Gap AGAINST trend works better than gap WITH trend (counter-intuitive but proven)
• High closing range % (70-75%) indicates strong conviction moves
• Trade smaller size on high volatility setups (Setup #3)
• BE PATIENT - wait for the right setups, don't force trades on non-Buy_Day2 days
"""

ax5.text(0.02, 0.98, notes_text, fontsize=9, va='top', family='monospace',
         transform=ax5.transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2, edgecolor=DARK_GRAY, linewidth=1))

fig3.text(0.5, 0.01, 'Page 3 of 3 | © GDT Trading System', ha='center', fontsize=8, color='gray')

pdf.savefig(fig3, bbox_inches='tight')
plt.close()

# Close PDF
pdf.close()

print("="*80)
print("✓ PDF CHEAT SHEET CREATED!")
print("="*80)
print(f"Saved to: {pdf_path}")
print("\nContents:")
print("  Page 1: GDT Cycle Overview & Top 3 Setups Summary")
print("  Page 2: Detailed Trading Rules for Each Setup")
print("  Page 3: Risk Management & Quick Reference")
print("\n📄 Print this and keep it by your trading desk!")
print("="*80)
