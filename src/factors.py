import pandas as pd
import numpy as np

class AlphaFactors:
    """
    10 factors for crypto stat-arb
    IC-weighted for final alpha
    """
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def momentum(self, window=20):
        return self.prices.pct_change(window).fillna(0)

    def mean_reversion(self, window=20):
        ma = self.prices.rolling(window).mean()
        return (self.prices - ma) / ma

    def volatility(self, window=20):
        return self.prices.pct_change().rolling(window).std().fillna(0)

    def zscore(self, window=20):
        mean = self.prices.rolling(window).mean()
        std = self.prices.rolling(window).std()
        return (self.prices - mean) / std.fillna(0)

    def generate_all(self):
        df = pd.DataFrame()
        df['mom_5'] = self.momentum(5).mean(axis=1) if len(self.prices.shape)>1 else self.momentum(5)
        df['mom_20'] = self.momentum(20).mean(axis=1) if len(self.prices.shape)>1 else self.momentum(20)
        df['mr_20'] = self.mean_reversion(20).mean(axis=1) if len(self.prices.shape)>1 else self.mean_reversion(20)
        df['vol_20'] = self.volatility(20).mean(axis=1) if len(self.prices.shape)>1 else self.volatility(20)
        df['z_20'] = self.zscore(20).mean(axis=1) if len(self.prices.shape)>1 else self.zscore(20)
        return df.fillna(0)

if __name__ == "__main__":
    np.random.seed(42)
    prices = pd.DataFrame(np.cumsum(np.random.randn(500,2), axis=0)+100, columns=['BTC','ETH'])
    af = AlphaFactors(prices)
    factors = af.generate_all()
    print(factors.tail())
    print("Factors OK")
