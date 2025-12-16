"""
GDT CYCLE VISUALIZATION - MOMENTUM vs EXHAUSTION
Shows which day types are continuation (high edge) vs exhaustion (fade signals)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.2, 1], hspace=0.3, wspace=0.3)

# Color scheme
MOMENTUM_COLOR = '#2ecc71'  # Green
EXHAUSTION_COLOR = '#e74c3c'  # Red
NEUTRAL_COLOR = '#f39c12'    # Orange
BUY_CYCLE_BG = '#e8f8f5'     # Light green
SELL_CYCLE_BG = '#fadbd8'    # Light red

# ============================================================================
# SUBPLOT 1: GDT CYCLE TIMELINE (Top - spans both columns)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)
ax1.axis('off')
ax1.set_title('GDT DAY# CYCLE: MOMENTUM vs EXHAUSTION PHASES',
              fontsize=20, fontweight='bold', pad=20)

# Buy Cycle Background
buy_cycle_box = Rectangle((0.3, 1.8), 4.2, 2.5,
                          facecolor=BUY_CYCLE_BG, edgecolor='#27ae60',
                          linewidth=3, alpha=0.3)
ax1.add_patch(buy_cycle_box)
ax1.text(2.4, 4.5, 'BULLISH CYCLE', fontsize=14, fontweight='bold',
         ha='center', color='#27ae60')

# Sell Cycle Background
sell_cycle_box = Rectangle((5.5, 1.8), 4.2, 2.5,
                           facecolor=SELL_CYCLE_BG, edgecolor='#c0392b',
                           linewidth=3, alpha=0.3)
ax1.add_patch(sell_cycle_box)
ax1.text(7.6, 4.5, 'BEARISH CYCLE', fontsize=14, fontweight='bold',
         ha='center', color='#c0392b')

# Day boxes with edges
day_positions = {
    'Buy_Day1': (0.5, 2.8, NEUTRAL_COLOR, '+2%'),
    'Buy_Day2': (1.5, 3.5, MOMENTUM_COLOR, '+8-12%'),  # HIGHEST
    'Buy_Day3': (2.5, 2.5, NEUTRAL_COLOR, '+3%'),
    'Buy_Day4': (3.5, 2.0, EXHAUSTION_COLOR, '-5-8%'),  # EXHAUSTION
    'Sell_Day1': (5.7, 2.8, NEUTRAL_COLOR, '-2%'),
    'Sell_Day2': (6.7, 2.5, NEUTRAL_COLOR, '-3%'),
    'Sell_Day3': (7.7, 2.3, NEUTRAL_COLOR, '-2%'),
    'Sell_Day4': (8.7, 2.0, EXHAUSTION_COLOR, 'Fade'),
}

# Draw day boxes
for day_name, (x, y, color, edge) in day_positions.items():
    # Box height represents edge strength
    height = 0.6
    box = FancyBboxPatch((x, y), 0.8, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black',
                         linewidth=2, alpha=0.8)
    ax1.add_patch(box)

    # Day label
    day_num = day_name.split('_')[1]
    ax1.text(x + 0.4, y + height/2, day_num,
            fontsize=12, fontweight='bold', ha='center', va='center')

    # Edge annotation
    ax1.text(x + 0.4, y - 0.3, edge,
            fontsize=10, ha='center', fontweight='bold',
            color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Arrows between days
arrow_props = dict(arrowstyle='->', lw=2, color='gray')
for i in range(3):
    start_x = 0.5 + i + 0.8
    ax1.annotate('', xy=(start_x + 0.2, 3.1), xytext=(start_x, 3.1),
                arrowprops=arrow_props)

# Cycle flip arrow
flip_arrow = FancyArrowPatch((4.5, 2.5), (5.5, 2.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='purple')
ax1.add_patch(flip_arrow)
ax1.text(5.0, 2.8, 'FLIP', fontsize=11, fontweight='bold',
        ha='center', color='purple',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', linewidth=2))

for i in range(3):
    start_x = 5.7 + i + 0.8
    ax1.annotate('', xy=(start_x + 0.2, 2.6), xytext=(start_x, 2.6),
                arrowprops=arrow_props)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=MOMENTUM_COLOR, edgecolor='black', label='MOMENTUM (Best Edge)'),
    mpatches.Patch(facecolor=NEUTRAL_COLOR, edgecolor='black', label='Neutral (Modest Edge)'),
    mpatches.Patch(facecolor=EXHAUSTION_COLOR, edgecolor='black', label='EXHAUSTION (Fade Signal)')
]
ax1.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=True)

# Key insight boxes
ax1.text(2.4, 0.8, '🎯 SWEET SPOT\nBuy_Day2 = Strongest Edges\nGap Against Trend works best',
        fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=MOMENTUM_COLOR, alpha=0.2, edgecolor='green', linewidth=2))

ax1.text(7.6, 0.8, '⚠️ LATE CYCLE\nDay 4 = Exhaustion\nFade gap-with-trend setups',
        fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=EXHAUSTION_COLOR, alpha=0.2, edgecolor='red', linewidth=2))

# ============================================================================
# SUBPLOT 2: BUY_DAY2 - MOMENTUM CONTINUATION (Bottom Left)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlim(0, 10)
ax2.set_ylim(95, 105)
ax2.set_title('BUY_DAY2: MOMENTUM CONTINUATION', fontsize=14, fontweight='bold', color='green')
ax2.set_xlabel('Time of Day', fontsize=11)
ax2.set_ylabel('Price', fontsize=11)
ax2.grid(True, alpha=0.3)

# Simulate typical Buy_Day2 price action
time_points = np.linspace(0, 10, 100)

# Scenario 1: Gap Down (against trend) then recovery
gap_down_scenario = 100 + np.concatenate([
    np.array([-2]),  # Gap down
    -2 + 0.3 * np.arange(20),  # Morning recovery
    1.8 + 0.8 * np.sin(np.linspace(0, np.pi/2, 40)) + np.linspace(0, 1.5, 40),  # Grind higher
    3.3 + 0.2 * np.sin(np.linspace(0, np.pi, 39))  # Consolidate near highs
])

ax2.plot(time_points, gap_down_scenario, linewidth=3, color='green', label='Gap Against Trend (Typical)')
ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Previous Close', alpha=0.7)
ax2.axhline(y=98, color='red', linestyle=':', linewidth=1, label='Gap Open', alpha=0.7)

# Annotations
ax2.annotate('Gap DOWN\n(Against Trend)', xy=(0, 98), xytext=(1, 96.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red')

ax2.annotate('Shakeout\nReversal', xy=(3, 100.5), xytext=(4.5, 98),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green')

ax2.annotate('Momentum\nContinuation', xy=(7, 103), xytext=(8.5, 104),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green')

ax2.text(9, 103.5, 'Close: +3.3%\n70-75% of range',
        fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax2.set_xticks([0, 2.5, 5, 7.5, 10])
ax2.set_xticklabels(['9:30am', '10:30am', '12:00pm', '2:00pm', '4:00pm'])
ax2.legend(loc='upper left', fontsize=9)

# Add stats box
stats_text = 'Gap Against + Yest Close Top Qtr\n+10-12% Edge | +2.0-2.5% Avg Move\n70-75% Closing Range'
ax2.text(0.5, 97, stats_text, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

# ============================================================================
# SUBPLOT 3: BUY_DAY4 - EXHAUSTION (Bottom Right)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlim(0, 10)
ax3.set_ylim(95, 105)
ax3.set_title('BUY_DAY4: EXHAUSTION / FADE', fontsize=14, fontweight='bold', color='red')
ax3.set_xlabel('Time of Day', fontsize=11)
ax3.set_ylabel('Price', fontsize=11)
ax3.grid(True, alpha=0.3)

# Scenario: Gap Up (with trend) then fade/chop
gap_up_fade = 100 + np.concatenate([
    np.array([2.5]),  # Gap up
    2.5 + 0.3 * np.sin(np.linspace(0, np.pi, 20)),  # Morning pump
    2.3 - 1.5 * np.linspace(0, 1, 40),  # Fade back
    0.8 + 0.5 * np.sin(np.linspace(0, 2*np.pi, 39))  # Chop/consolidate
])

ax3.plot(time_points, gap_up_fade, linewidth=3, color='red', label='Gap With Trend (Fade)', alpha=0.8)
ax3.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Previous Close', alpha=0.7)
ax3.axhline(y=102.5, color='green', linestyle=':', linewidth=1, label='Gap Open', alpha=0.7)

# Annotations
ax3.annotate('Gap UP\n(With Trend)', xy=(0, 102.5), xytext=(1.5, 104),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green')

ax3.annotate('Failed\nBreakout', xy=(2, 102.7), xytext=(3.5, 104),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=10, fontweight='bold', color='orange')

ax3.annotate('Exhaustion\nFade', xy=(5, 101), xytext=(6.5, 103),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red')

ax3.text(9, 101.5, 'Close: +0.8%\n45-50% of range\nChoppy',
        fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

ax3.set_xticks([0, 2.5, 5, 7.5, 10])
ax3.set_xticklabels(['9:30am', '10:30am', '12:00pm', '2:00pm', '4:00pm'])
ax3.legend(loc='upper right', fontsize=9)

# Add warning box
warning_text = 'Gap With Trend on Day 4\n-5 to -8% Edge | FADE SETUP\nWeak/Choppy Action'
ax3.text(0.5, 97, warning_text, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))

# ============================================================================
# SUBPLOT 4: EDGE COMPARISON BY DAY (Bottom Left)
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

days = ['Buy_Day1', 'Buy_Day2', 'Buy_Day3', 'Buy_Day4']
gap_against_edges = [2, 10, 3, -1]  # Example edges for Gap Against Trend
gap_with_edges = [1, 4, -2, -7]     # Example edges for Gap With Trend

x = np.arange(len(days))
width = 0.35

bars1 = ax4.bar(x - width/2, gap_against_edges, width, label='Gap Against Trend',
               color=[NEUTRAL_COLOR, MOMENTUM_COLOR, NEUTRAL_COLOR, EXHAUSTION_COLOR],
               edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, gap_with_edges, width, label='Gap With Trend',
               color=[NEUTRAL_COLOR, NEUTRAL_COLOR, EXHAUSTION_COLOR, EXHAUSTION_COLOR],
               edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Edge (%)', fontsize=11, fontweight='bold')
ax4.set_title('Gap Behavior Edge by GDT Day (Buy Cycle)', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(days, rotation=0, fontsize=10)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.0f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

# ============================================================================
# SUBPLOT 5: SETUP SUMMARY TABLE (Bottom Right)
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
ax5.set_title('TRADE SETUP SUMMARY', fontsize=13, fontweight='bold', pad=10)

# Create table data
table_data = [
    ['Day Type', 'Best Setup', 'Edge', 'Action'],
    ['Buy_Day1', 'GapStat >2', '+2-4%', '🟡 Monitor'],
    ['Buy_Day2', 'Gap Against + Top Qtr', '+10-12%', '🟢 LONG'],
    ['Buy_Day3', 'Gap Against', '+3-5%', '🟡 Monitor'],
    ['Buy_Day4', 'Gap With Trend', '-5-8%', '🔴 FADE'],
    ['Sell_Day1', 'Gap Down', '-2-3%', '🟡 Monitor'],
    ['Sell_Day2', 'Gap Down', '-3-4%', '🟡 Monitor'],
    ['Sell_Day3', 'Any Gap', '-2-3%', '🟡 Monitor'],
    ['Sell_Day4', 'Large Gaps', 'Fade', '🔴 AVOID'],
]

# Create table
table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.2, 0.35, 0.15, 0.15],
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Color code rows
row_colors = {
    1: '#f8f9fa',  # Day1 - neutral
    2: '#d5f4e6',  # Day2 - green (best)
    3: '#f8f9fa',  # Day3 - neutral
    4: '#fadbd8',  # Day4 - red (fade)
    5: '#f8f9fa',  # Sell days
    6: '#f8f9fa',
    7: '#f8f9fa',
    8: '#fadbd8',  # Sell_Day4 - red
}

for row_idx, color in row_colors.items():
    for col_idx in range(4):
        table[(row_idx, col_idx)].set_facecolor(color)

# Make specific cells bold
for row in [2, 4]:  # Buy_Day2 and Buy_Day4
    for col in range(4):
        table[(row, col)].set_text_props(weight='bold')

# Add note
ax5.text(0.5, -0.05, '🎯 Focus on Buy_Day2 | ⚠️ Avoid/Fade Buy_Day4 & Sell_Day4',
        ha='center', fontsize=11, fontweight='bold', style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# Main title and annotations
# ============================================================================
fig.suptitle('GDT CYCLE ANALYSIS: MOMENTUM vs EXHAUSTION PATTERNS',
            fontsize=22, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('/home/user/trendlyne-widget/gdt_cycle_momentum_exhaustion.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: gdt_cycle_momentum_exhaustion.png")

plt.show()
