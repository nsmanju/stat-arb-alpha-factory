import numpy as np
import pandas as pd

# --- Generate fake cointegrated pair like your pdb_backtest.py ---
np.random.seed(42)
x = pd.Series(np.cumsum(np.random.randn(200)) + 100) # INFY
beta = 1.52
y = beta * x + np.random.randn(200) # TCS = beta*INFY + noise

# --- Build spread ---
spread = y - beta * x # distance between twins

# --- Rolling mean and std ---
mean = spread.rolling(20).mean()
std = spread.rolling(20).std()

# src/backtest.py - Vectorized Backtest Explained

# --- Step 1: Build Z-score for all 200 days at once ---
# Spread = y - beta*x (distance between twins)
# mean = rolling 20-day average of spread (normal distance)
# std = rolling 20-day volatility (how much it normally wiggles)
# z = how many std away from normal today?
# Example: z=2 means spread is 2 std above normal = twins too far apart
z = (spread - mean) / std # Vector: 200 values in one shot, no loop

# --- Step 2: Create position series ---
# Start flat: 0 = no trade
# This will hold -1, 0, +1 for each day
pos = pd.Series(0, index=z.index) # 200 zeros

# --- Step 3: Entry Rule - Short when too high ---
# If z > 2: spread is abnormally high, expect it to fall
# So we SHORT spread: sell y, buy beta*x
# pos = -1 means short 1 unit of spread
# Example: z=2.5 on day 45 -> pos[45] = -1
pos[z > 2] = -1 # Find all days where z>2, set to -1 (short signal)

# --- Step 4: Entry Rule - Long when too low ---
# If z < -2: spread is abnormally low, expect it to rise
# So we LONG spread: buy y, sell beta*x
# pos = +1 means long 1 unit of spread
# Example: z=-2.3 on day 102 -> pos[102] = 1
pos[z < -2] = 1 # Find all days where z<-2, set to +1 (long signal)

# --- Step 5: Hold Position (ffill = forward fill) ---
# After entry, we hold until exit
# ffill copies last non-zero pos forward
# Example: Enter -1 on day 45, day 46,47,48... stay -1 until exit signal
# Without ffill, you would trade only 1 day and go flat
pos = pos.ffill().fillna(0) # Fill NaN with 0 at start

# --- Step 6: Exit Rule ---
# In full version: if abs(z) < 0.5 -> pos = 0
pos[abs(z) < 0.5] = 0
pos = pos.ffill().fillna(0)

# --- Step 7: Calculate Returns - MOST IMPORTANT LINE ---
# pos.shift(1) = yesterday's position
# Why shift? To avoid lookahead bias / cheating
# In real life: You see signal at 3:59pm close, you can only trade tomorrow open
returns = pos.shift(1) * spread.diff() # Vector of 200 daily PnLs

# --- Step 8: Measure ---
sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std()!=0 else 0
pnl = returns.sum()
print(f"pos head:\n{pos.head(10)}")
print(f"\nreturns head:\n{returns.head(10)}")
print(f"\nSharpe: {sharpe:.2f} | Total PnL: {pnl:.2f}")
