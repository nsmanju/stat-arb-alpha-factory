import numpy as np
import pandas as pd

class Backtest:
    """
    Z-score mean-reversion backtest
    Entry: z > 2 short spread, z < -2 long spread
    Exit: z crosses 0
    """
    def __init__(self, spread: pd.Series, zscore: pd.Series = None):
        self.spread = spread.dropna()
        self.zscore = zscore

    def calc_zscore(self, window=20):
        mean = self.spread.rolling(window).mean()
        std = self.spread.rolling(window).std()
        return (self.spread - mean) / std

    def run(self, entry=2.0, exit_level=0.0):
        if self.zscore is None:
            z = self.calc_zscore()
        else:
            z = self.zscore
        
        pos = pd.Series(0, index=z.index)
        pos[z > entry] = -1
        pos[z < -entry] = 1
        # exit when crosses 0
        exit_mask = (z * pos.shift(1)) < 0
        pos[exit_mask] = 0
        pos = pos.ffill().fillna(0)
        
        ret = pos.shift(1) * self.spread.diff()
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std()!=0 else 0
        return {
            'positions': pos,
            'returns': ret,
            'sharpe': sharpe,
            'cumulative': ret.cumsum()
        }

    def summary(self):
        r = self.run()
        cum = r['cumulative'].iloc[-1] if len(r['cumulative'])>0 else 0
        return f"Sharpe: {r['sharpe']:.2f} | Total PnL: {cum:.2f}"
