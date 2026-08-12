import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class CointegrationTest:
    """
    Engle-Granger 2-step cointegration for stat-arb pairs
     "How do you test if BTC and ETH are cointegrated?"
    Answer: This file
    """

    def __init__(self, price_x: pd.Series, price_y: pd.Series):
        """
        price_x, price_y: e.g., BTC and ETH, or BTC spot vs BTC perp
        Both must be same length, log prices preferred
        """
        self.price_x = price_x
        self.price_y = price_y

    def engle_granger_test(self):
        """
        Step 1: Test if two price series are cointegrated
        Returns: coint_stat, p_value, is_cointegrated (p<0.05)

        Logic: If prices are cointegrated -> spread is stationary -> mean reverts -> tradable
        If p > 0.05 -> random walk, don't trade as pair
        """
        # statsmodels coint test
        coint_stat, p_value, crit_values = coint(self.price_x, self.price_y)

        is_cointegrated = p_value < 0.05

        return {
            'coint_stat': coint_stat,
            'p_value': p_value,
            'critical_values': crit_values,
            'is_cointegrated': is_cointegrated,
            # Interview: p<0.05 means reject null (no cointegration) -> Yes, cointegrated
        }

    def get_hedge_ratio(self):
        """
        Step 2: If cointegrated, get hedge ratio (beta)
        Model: y = alpha + beta * x + residual
        Residual = spread = tradable

        Example: BTC = 1.5 * ETH + spread. Hedge: Long 1 BTC, Short 1.5 ETH
        """
        X = sm.add_constant(self.price_x) # Add intercept
        model = sm.OLS(self.price_y, X).fit()

        alpha = model.params[0] # Intercept
        beta = model.params[1] # Hedge ratio

        # Spread = y - (alpha + beta*x) = stationary if cointegrated
        spread = self.price_y - (alpha + beta * self.price_x)

        return {
            'alpha': alpha,
            'beta': beta, # THIS is hedge ratio
            'spread': spread,
            'model_summary': model.summary() # For debugging
        }

    def get_spread_zscore(self, window=20):
        """
        Convert spread to z-score for trading signals
        z > 2 = spread too high = short spread (short y, long x)
        z < -2 = spread too low = long spread (long y, short x)

        window=20: 20-day rolling mean/std, typical for crypto
        """
        hedge = self.get_hedge_ratio()
        spread = hedge['spread']

        # Rolling z-score
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        zscore = (spread - mean) / std

        return zscore.fillna(0)

# --- LOCAL TEST: BTC/ETH cointegration example ---
if __name__ == "__main__":
    # Simulate cointegrated prices: ETH = 0.05*BTC + noise
    np.random.seed(42)
    btc = pd.Series(np.cumsum(np.random.randn(200)) + 50000)
    eth = pd.Series(0.05 * btc + np.random.randn(200) * 10 + 100)

    test = CointegrationTest(btc, eth)
    result = test.engle_granger_test()
    hedge = test.get_hedge_ratio()
    zscore = test.get_spread_zscore()

    print(f"Coint p-value: {result['p_value']:.4f} -> Cointegrated? {result['is_cointegrated']}")
    print(f"Hedge ratio (beta): {hedge['beta']:.4f}")
    print(f"Last z-score: {zscore.iloc[-1]:.2f} (Trade if |z|>2)")
