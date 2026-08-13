import sys
sys.path.insert(0,'src')
import pandas as pd, numpy as np
from cointegration import CointegrationTest
from ou_process import OUProcess
from backtest import Backtest

np.random.seed(42)
x = pd.Series(np.cumsum(np.random.randn(200)))
y = pd.Series(1.52*x + np.random.randn(200)*0.5)

hedge = CointegrationTest(x,y).get_hedge_ratio()
spread = hedge['spread']
z = CointegrationTest(x,y).get_spread_zscore(window=20)

print(f"beta={hedge['beta']:.2f}")


bt = Backtest(spread, z)
print(bt.summary())
