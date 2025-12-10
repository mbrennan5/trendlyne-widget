# Trading Day Classification Study
# Classifies trading days as Range Day, DWP (Directional With Pullbacks), or DNP (Directional No Pullbacks)
# Based on 9:30 AM range and hourly behavior during RTH (09:30 - 16:15 EST)

declare upper;

# ==============================================================================
# PHASE 1: FOUNDATION AND CONSTANT VARIABLES
# ==============================================================================

# Define RTH session times (09:30 - 16:15 EST)
def marketOpen = 0930;
def marketClose = 1615;
def currentTime = SecondsFromTime(marketOpen);
def closeTime = SecondsFromTime(marketClose);

# isRTH: Check if we're in Regular Trading Hours
def isRTH = currentTime >= 0 and closeTime <= 0;

# Capture 9:30 AM High and Low (Primary Range Reference)
def isFirstBar = SecondsFromTime(marketOpen) == 0;
def H930 = if isFirstBar then high else H930[1];
def L930 = if isFirstBar then low else L930[1];

# Reset tracking variables at start of new day
def isNewDay = GetDay() != GetDay()[1];

# ==============================================================================
# HOURLY SESSION TRACKING
# ==============================================================================

# NewHour: Triggers at the start of each hour (09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30)
# Using SecondsFromTime to detect first bar of each hour
def at0930 = SecondsFromTime(0930) == 0;
def at1030 = SecondsFromTime(1030) == 0;
def at1130 = SecondsFromTime(1130) == 0;
def at1230 = SecondsFromTime(1230) == 0;
def at1330 = SecondsFromTime(1330) == 0;
def at1430 = SecondsFromTime(1430) == 0;
def at1530 = SecondsFromTime(1530) == 0;

def NewHour = at0930 or at1030 or at1130 or at1230 or at1330 or at1430 or at1530;

# Track which hourly session we're in (0-6)
def hourlySession = if isNewDay then 0
                   else if NewHour then hourlySession[1] + 1
                   else hourlySession[1];

# Track hourly High and Low (resets at each NewHour)
def hourlyHigh = if isNewDay then high
                 else if NewHour then high
                 else if isRTH then Max(high, hourlyHigh[1])
                 else hourlyHigh[1];

def hourlyLow = if isNewDay then low
                else if NewHour then low
                else if isRTH then Min(low, hourlyLow[1])
                else hourlyLow[1];

# Store previous hour's high and low for pullback comparison
def prevHourHigh = if NewHour then hourlyHigh[1] else prevHourHigh[1];
def prevHourLow = if NewHour then hourlyLow[1] else prevHourLow[1];

# ==============================================================================
# PHASE 2: DIRECTIONAL STATE AND PULLBACK TRACKING
# ==============================================================================

# Dir_Up: Set to True (and held) when market high first moves above H930
def Dir_Up = if isNewDay then 0
             else if high > H930 then 1
             else Dir_Up[1];

# Dir_Down: Set to True (and held) when market low first moves below L930
def Dir_Down = if isNewDay then 0
               else if low < L930 then 1
               else Dir_Down[1];

# isDirectional: True if either directional bias is established
def isDirectional = Dir_Up or Dir_Down;

# HourlyPullback: Check for pullbacks at each hour
# - If Dir_Up: Current hourlyLow < Previous hourlyLow (pullback down toward 930)
# - If Dir_Down: Current hourlyHigh > Previous hourlyHigh (pullback up toward 930)
def HourlyPullback = if !NewHour then 0
                     else if Dir_Up and hourlyLow < prevHourLow then 1
                     else if Dir_Down and hourlyHigh > prevHourHigh then 1
                     else 0;

# HasPullbackOccurred: Persistent flag set if any pullback occurs during RTH
def HasPullbackOccurred = if isNewDay then 0
                          else if HourlyPullback then 1
                          else HasPullbackOccurred[1];

# ==============================================================================
# PHASE 3: RANGE DAY CONDITIONS
# ==============================================================================

# Range 1 Condition: 4+ Hours Test (overlap with 930 range)
# Test930: Does current hourly range overlap the 930 range?
def Test930 = hourlyLow <= H930 and hourlyHigh >= L930;

# TestCount: Count number of hours that tested the 930 range
def TestCount = if isNewDay then 0
                else if NewHour and Test930 then TestCount[1] + 1
                else TestCount[1];

# Range1: True if 4 or more hours tested the 930 range
def Range1 = TestCount >= 4;

# Range 2 Condition: Late Reversal (directional, then reverses back to 930 range between 11:00-16:15)
def isAfter1100 = SecondsFromTime(1100) >= 0;
def touchedRangeAfter1100 = high >= L930 and low <= H930 and isAfter1100;

# ReversedTo930: Persistent flag if price touches 930 range after 11:00
def ReversedTo930 = if isNewDay then 0
                    else if touchedRangeAfter1100 then 1
                    else ReversedTo930[1];

# Range2: Directional day that reversed back to 930 range
def Range2 = isDirectional and ReversedTo930;

# Final Range Day Condition
def isRangeDay = Range1 or Range2;

# ==============================================================================
# PHASE 4: FINAL CLASSIFICATION (at end of RTH session)
# ==============================================================================

# Classification values:
# 0 = N/A (Inconclusive/No Breakout)
# 1 = Range Day
# 2 = DWP (Directional with Pullbacks)
# 3 = DNP (Directional No Pullbacks)

def classification = if isRangeDay then 1
                     else if isDirectional and HasPullbackOccurred then 2
                     else if isDirectional and !HasPullbackOccurred then 3
                     else 0;

# ==============================================================================
# VISUALIZATION AND LABELS
# ==============================================================================

# Plot the 930 range levels
plot High930 = if isRTH then H930 else Double.NaN;
plot Low930 = if isRTH then L930 else Double.NaN;

High930.SetDefaultColor(Color.CYAN);
High930.SetStyle(Curve.SHORT_DASH);
High930.SetLineWeight(2);

Low930.SetDefaultColor(Color.CYAN);
Low930.SetStyle(Curve.SHORT_DASH);
Low930.SetLineWeight(2);

# Add classification label
AddLabel(yes,
         "Day Type: " +
         (if classification == 1 then "RANGE DAY"
          else if classification == 2 then "DWP (Directional w/ Pullbacks)"
          else if classification == 3 then "DNP (Directional No Pullbacks)"
          else "N/A"),
         (if classification == 1 then Color.YELLOW
          else if classification == 2 then Color.ORANGE
          else if classification == 3 then Color.GREEN
          else Color.GRAY));

# Additional debug labels (optional - can be commented out)
AddLabel(yes, "930 Range: " + Round(H930, 2) + " - " + Round(L930, 2), Color.CYAN);
AddLabel(yes, "Dir: " + (if Dir_Up then "UP" else if Dir_Down then "DOWN" else "NONE"),
         if isDirectional then Color.WHITE else Color.GRAY);
AddLabel(yes, "Hours Tested 930: " + TestCount, Color.LIGHT_GRAY);
AddLabel(yes, "Pullbacks: " + (if HasPullbackOccurred then "YES" else "NO"),
         if HasPullbackOccurred then Color.ORANGE else Color.GRAY);
AddLabel(yes, "Range1: " + (if Range1 then "YES" else "NO"),
         if Range1 then Color.YELLOW else Color.GRAY);
AddLabel(yes, "Range2: " + (if Range2 then "YES" else "NO"),
         if Range2 then Color.YELLOW else Color.GRAY);

# ==============================================================================
# CLOUD VISUALIZATION (Optional - visualize the 930 range as a cloud)
# ==============================================================================

plot CloudHigh = if isRTH then H930 else Double.NaN;
plot CloudLow = if isRTH then L930 else Double.NaN;
CloudHigh.SetDefaultColor(Color.DARK_GRAY);
CloudLow.SetDefaultColor(Color.DARK_GRAY);
CloudHigh.Hide();
CloudLow.Hide();
AddCloud(CloudHigh, CloudLow, Color.DARK_GRAY, Color.DARK_GRAY);

# ==============================================================================
# BACKGROUND COLOR (Optional - color the background based on classification)
# ==============================================================================

AssignBackgroundColor(
    if !isRTH then Color.BLACK
    else if classification == 1 then CreateColor(40, 40, 0)    # Dark Yellow for Range
    else if classification == 2 then CreateColor(40, 20, 0)    # Dark Orange for DWP
    else if classification == 3 then CreateColor(0, 30, 0)     # Dark Green for DNP
    else Color.BLACK                                            # Black for N/A
);
