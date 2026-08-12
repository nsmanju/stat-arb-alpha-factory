import numpy as np
import pandas as pd

class AlphaFactors:
    """
    10-factor alpha factory for stat-arb
    Target: Combine to Sharpe 1.5 walk-forward
    """
    def __init__(self, price: pd.Series):
        self.price = price
        self.ret = price.pct_change()

    def factor_1_momentum_5d(self):
        """5-day momentum"""
        return self.price.pct_change(5)

    def factor_2_momentum_20d(self):
        """20-day momentum"""
        return self.price.pct_change(20)

    def factor_3_mean_reversion_5d(self):
        """5d mean reversion: -zscore(5d ret)"""
        r = self.ret.rolling(5).mean()
        return -(r - r.rolling(20).mean()) / r.rolling(20).std()

    def factor_4_mean_reversion_20d(self):
        """20d mean reversion"""
        r = self.ret.rolling(20).mean()
        return -(r - r.rolling(60).mean()) / r.rolling(60).std()

    def factor_5_rsi_14d(self):
        """RSI mean-rev: 50 - RSI"""
        delta = self.price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return 50 - rsi

    def factor_6_vol_imbalance(self):
        """Volatility regime: short vol when high"""
        vol_short = self.ret.rolling(5).std()
        vol_long = self.ret.rolling(20).std()
        return -(vol_short / vol_long - 1)

    def factor_7_funding_proxy(self):
        """Funding proxy: high ret = high funding = short"""
        return -self.ret.rolling(8).mean() * 100

    def factor_8_skew_20d(self):
        """Skewness factor"""
        return self.ret.rolling(20).skew()

    def factor_9_kurtosis_signal(self):
        """Low kurtosis = stable = long"""
        kurt = self.ret.rolling(20).kurt()
        return -kurt

    def factor_10_trend_vs_reversion(self):
        """Trend strength"""
        ma5 = self.price.rolling(5).mean()
        ma20 = self.price.rolling(20).mean()
        return (ma5 / ma20 - 1)

    def get_all_factors(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'mom_5d': self.factor_1_momentum_5d(),
            'mom_20d': self.factor_2_momentum_20d(),
            'mean_rev_5d': self.factor_3_mean_reversion_5d(),
            'mean_rev_20d': self.factor_4_mean_reversion_20d(),
            'rsi': self.factor_5_rsi_14d(),
            'vol_imb': self.factor_6_vol_imbalance(),
            'funding': self.factor_7_funding_proxy(),
            'skew': self.factor_8_skew_20d(),
            'kurt': self.factor_9_kurtosis_signal(),
            'trend': self.factor_10_trend_vs_reversion(),
        })
        return df.fillna(0)

# Quick test
if __name__ == "__main__":
    price = pd.Series(np.cumsum(np.random.randn(100)) + 100)
    af = AlphaFactors(price)
    print(af.get_all_factors().tail())
