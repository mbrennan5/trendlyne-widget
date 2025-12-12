#!/usr/bin/env python3
"""
Debug script to trace pullback detection logic step-by-step
"""

def debug_classify_day(bars_data):
    """
    Debug version that shows step-by-step pullback detection

    Args:
        bars_data: List of dicts with 'hour', 'high', 'low' keys
    """
    opening = bars_data[0]['open']
    H930 = L930 = opening

    print(f"Opening Price (H930=L930): {opening}")
    print(f"{'='*80}")

    dir_up = False
    dir_down = False
    has_pullback = False

    prev_high = None
    prev_low = None

    for i, bar in enumerate(bars_data):
        hour_time = bar['hour']
        hour_high = bar['high']
        hour_low = bar['low']

        print(f"\n{hour_time}: High={hour_high}, Low={hour_low}")

        # Check directional bias (skip first bar - matches isNewDay reset in ThinkScript)
        if i > 0:  # Skip first bar (9:30) - directional flags reset by isNewDay
            if hour_high > H930:
                if not dir_up:
                    print(f"  → Dir_Up SET (high {hour_high} > opening {H930})")
                dir_up = True
            if hour_low < L930:
                if not dir_down:
                    print(f"  → Dir_Down SET (low {hour_low} < opening {L930})")
                dir_down = True

        # Check for pullbacks (skip first hour at index 0 AND last hour)
        # Last bar (3:30 PM) might be excluded from pullback detection
        is_last_bar = i == len(bars_data) - 1

        if i > 0 and prev_high is not None and prev_low is not None and not is_last_bar:
            print(f"  Pullback Check:")
            print(f"    Previous hour: High={prev_high}, Low={prev_low}")
            print(f"    Current hour:  High={hour_high}, Low={hour_low}")

            if dir_up and hour_low < prev_low:
                print(f"    ✓ PULLBACK DETECTED (Dir_Up and {hour_low} < {prev_low})")
                has_pullback = True
            elif dir_down and hour_high > prev_high:
                print(f"    ✓ PULLBACK DETECTED (Dir_Down and {hour_high} > {prev_high})")
                has_pullback = True
            else:
                if dir_up:
                    print(f"    No pullback (Dir_Up but {hour_low} >= {prev_low})")
                elif dir_down:
                    print(f"    No pullback (Dir_Down but {hour_high} <= {prev_high})")
                else:
                    print(f"    No pullback (no directional bias yet)")
        elif i == 0:
            print(f"  (First hour - directional and pullback checks skipped, matches isNewDay reset)")
        elif is_last_bar:
            print(f"  (Last hour - pullback check SKIPPED - testing if this matches ThinkScript)")

        # Store for next iteration
        prev_high = hour_high
        prev_low = hour_low

    print(f"\n{'='*80}")
    print(f"RESULT:")
    print(f"  Directional: {'Up' if dir_up else ''}{'Down' if dir_down else ''}{'None' if not (dir_up or dir_down) else ''}")
    print(f"  Has Pullback: {has_pullback}")
    print(f"  Classification: {'DWP' if has_pullback else 'DNP' if (dir_up or dir_down) else 'N/A'}")
    print(f"{'='*80}\n")

    return has_pullback


# 12/3/25 Data from ThinkScript
print("SPY 12/3/25")
print("="*80)
day1 = [
    {'hour': '9:30',  'open': 680.57, 'high': 682.34, 'low': 679.69},
    {'hour': '10:30', 'open': 681.65, 'high': 682.99, 'low': 681.3299},
    {'hour': '11:30', 'open': 682.385, 'high': 683.7, 'low': 681.61},
    {'hour': '12:30', 'open': 683.55, 'high': 684.185, 'low': 683.25},
    {'hour': '1:30',  'open': 683.93, 'high': 684.5, 'low': 683.7866},
    {'hour': '2:30',  'open': 684.33, 'high': 684.91, 'low': 684.165},
    {'hour': '3:30',  'open': 684.525, 'high': 684.62, 'low': 683.55},
]
debug_classify_day(day1)

# 12/4/25 Data from ThinkScript
print("\nSPY 12/4/25")
print("="*80)
day2 = [
    {'hour': '9:30',  'open': 685.3, 'high': 685.37, 'low': 682.17},
    {'hour': '10:30', 'open': 683.59, 'high': 684.56, 'low': 682.21},
    {'hour': '11:30', 'open': 683.31, 'high': 684.58, 'low': 682.86},
    {'hour': '12:30', 'open': 684.03, 'high': 684.77, 'low': 683.465},
    {'hour': '1:30',  'open': 684.67, 'high': 684.75, 'low': 681.34},
    {'hour': '2:30',  'open': 682.19, 'high': 683.96, 'low': 681.99},
    {'hour': '3:30',  'open': 682.76, 'high': 684.58, 'low': 682.7},
]
debug_classify_day(day2)
