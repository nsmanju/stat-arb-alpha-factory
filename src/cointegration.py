import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class CointegrationTest:
    def __init__(self, price_x: pd.Series, price_y: pd.Series):
        self.price_x = price_x
        self.price_y = price_y

    def engle_granger_test(self):
        coint_stat, p_value, crit_values = coint(self.price_x, self.price_y)
        return {
            'coint_stat': coint_stat,
            'p_value': p_value,
            'critical_values': crit_values,
            'is_cointegrated': p_value < 0.05,
        }

    def get_hedge_ratio(self):
        X = sm.add_constant(self.price_x)
        model = sm.OLS(self.price_y, X).fit()
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]
        spread = self.price_y - (alpha + beta * self.price_x)
        return {'alpha': alpha, 'beta': beta, 'spread': spread, 'model_summary': model.summary()}

    def get_spread_zscore(self, window=20):
        hedge = self.get_hedge_ratio()
        spread = hedge['spread']
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        return ((spread - mean) / std).fillna(0)

if __name__ == "__main__":
    np.random.seed(7)
    n = 1000

    # BTC: random walk starting at 0, large variance
    btc = pd.Series(np.cumsum(np.random.randn(n)))

    # Spread: stationary AR(1) rho=0.85
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.85 * spread[i-1] + np.random.randn() * 0.5
    spread = pd.Series(spread)

    # ETH: 1.5 * BTC + spread -> COINTEGRATED by definition
    eth = 1.5 * btc + spread

    test = CointegrationTest(btc, eth)
    result = test.engle_granger_test()
    hedge = test.get_hedge_ratio()
    zscore = test.get_spread_zscore()

    print(f"BTC var: {btc.var():.1f}, Spread var: {spread.var():.1f}")
    print(f"Coint p-value: {result['p_value']:.10f} -> Cointegrated? {result['is_cointegrated']}")
    print(f"Hedge ratio (beta): {hedge['beta']:.4f} (True 1.5)")
    print(f"Alpha: {hedge['alpha']:.4f} (True 0)")
    print(f"Last z-score: {zscore.iloc[-1]:.2f}")

    if result['is_cointegrated']:
        print("SUCCESS!")
    else:
        print("Still failing - try seed 7, n=1000 already set")
