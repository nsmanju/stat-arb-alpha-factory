import numpy as np
import pandas as pd
import statsmodels.api as sm

class OUProcess:
    """
    Ornstein-Uhlenbeck for spread half-life
    Formula: dX = theta*(mu - X)*dt + sigma*dW
    Discrete: delta_X = alpha + beta*X_lag + error, beta = -theta*dt
    Half-life = -ln(2)/beta

    HK use: If HL < 1 day -> too fast (costs kill you)
            If HL > 30 days -> too slow (capital tie up)
            Sweet spot: 2-10 days
    """

    def __init__(self, spread: pd.Series):
        """
        spread: cointegrated spread from cointegration.py
        e.g., spread = ETH - beta*BTC
        """
        self.spread = spread.dropna()

    def calc_half_life(self):
        """
        Calculate half-life via OLS regression

        Regression: delta_spread = alpha + beta*spread_lag
        If beta < 0: mean-reverting (good)
        If beta >=0: random walk / momentum (bad for stat-arb)

        Returns: half_life, beta, is_mean_reverting
        """
        # Lagged spread and delta
        spread_lag = self.spread.shift(1).dropna()
        delta_spread = self.spread.diff().dropna()

        # Align lengths
        # delta_spread is spread[t] - spread[t-1], so its index starts at 1
        # spread_lag index also starts at 1
        # Ensure same index
        common_idx = spread_lag.index.intersection(delta_spread.index)
        spread_lag = spread_lag.loc[common_idx]
        delta_spread = delta_spread.loc[common_idx]

        X = sm.add_constant(spread_lag) # const + lag
        model = sm.OLS(delta_spread, X).fit()

        # FIXED for pandas 2.x
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1] # Should be negative for mean-reversion

        # Half-life formula
        if beta < 0:
            half_life = -np.log(2) / beta
            is_mean_reverting = True
        else:
            half_life = np.inf # No mean reversion
            is_mean_reverting = False

        return {
            'half_life': half_life,
            'beta': beta,
            'alpha': alpha,
            'is_mean_reverting': is_mean_reverting,
            'model': model # For t-stat check
        }

    def calc_ou_params(self):
        """
        Full OU params: mu, theta, sigma
        For interview: mu=long-term mean, theta=speed, sigma=vol
        """
        hl_info = self.calc_half_life()
        beta = hl_info['beta']
        alpha = hl_info['alpha']

        theta = -beta # Mean reversion speed
        mu = -alpha / beta if beta!= 0 else self.spread.mean() # Long-term mean

        # Sigma from residuals
        spread_lag = self.spread.shift(1).dropna()
        delta_spread = self.spread.diff().dropna()
        common_idx = spread_lag.index.intersection(delta_spread.index)
        spread_lag = spread_lag.loc[common_idx]
        delta_spread = delta_spread.loc[common_idx]

        X = sm.add_constant(spread_lag)
        model = sm.OLS(delta_spread, X).fit()
        resid_std = model.resid.std()
        sigma = resid_std # Approx

        return {
            'mu': mu,
            'theta': theta,
            'sigma': sigma,
            'half_life': hl_info['half_life']
        }

# --- TEST ---
if __name__ == "__main__":
    # Simulate OU spread: mean=0, half-life ~ 5 days
    np.random.seed(42)
    n = 500
    spread = [0]
    theta = 0.13 # -> HL = ln2/theta ~5.3 days
    for i in range(1, n):
        spread.append(spread[-1] + theta*(0-spread[-1]) + np.random.randn()*0.5)

    spread = pd.Series(spread)

    ou = OUProcess(spread)
    info = ou.calc_half_life()
    params = ou.calc_ou_params()

    print(f"Half-life: {info['half_life']:.2f} days (Target 2-10)")
    print(f"Beta: {info['beta']:.4f} (should be <0)")
    print(f"Mean-reverting? {info['is_mean_reverting']}")
    print(f"OU Params: mu={params['mu']:.3f}, theta={params['theta']:.3f}, sigma={params['sigma']:.3f}")
