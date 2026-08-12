import numpy as np
import pandas as pd
import statsmodels.api as sm

class OUProcess:
    """
    Ornstein-Uhlenbeck process for mean-reversion speed
    Answers: How fast does spread come back to mean?
    Half-life = time to revert 50% back

   "What's typical half-life for crypto pairs?"
    Answer: 2-10 days is tradable. <1 day = noise. >30 days = too slow.
    """
    def __init__(self, spread: pd.Series):
        self.spread = spread.dropna()

    def calc_half_life(self):
        """
        OU: dS = theta*(mu - S)*dt + sigma*dW
        We estimate theta from regression:
        delta_spread = alpha + beta*spread_lag + error

        beta = -theta*dt
        half-life = -ln(2) / beta

        Returns: half-life in periods (if daily data -> days)
        """
        # Lagged spread
        spread_lag = self.spread.shift(1).dropna()
        spread_ret = self.spread.diff().dropna()

        # Align
        spread_lag = spread_lag.iloc[1:]
        spread_ret = spread_ret.iloc[1:]

        # Regression: spread_ret = alpha + beta*spread_lag
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, X).fit()

        beta = model.params[1] # Should be negative for mean-reversion

        # If beta >=0 : not mean reverting!
        if beta >= 0:
            return {
                'half_life': np.inf,
                'beta': beta,
                'is_mean_reverting': False,
                'msg': 'Not mean-reverting (beta>=0)'
            }

        half_life = -np.log(2) / beta

        return {
            'half_life': half_life, # e.g., 5.2 means 5.2 days to revert 50%
            'beta': beta,
            'alpha': model.params[0],
            'is_mean_reverting': True,
            'theta': -beta, # OU theta
        }

    def get_trading_rule(self, entry_z=2.0, exit_z=0.5):
        """
        Simple rule based on OU
        entry_z=2: Enter when |z|>2
        exit_z=0.5: Exit when |z|<0.5 (back to mean)

        Half-life used for holding period estimate
        """
        hl_info = self.calc_half_life()
        hl = hl_info['half_life']

        # Z-score
        mean = self.spread.rolling(20).mean()
        std = self.spread.rolling(20).std()
        z = (self.spread - mean) / std

        # Signals
        long_entry = z < -entry_z # Spread too low -> long spread
        short_entry = z > entry_z # Spread too high -> short spread
        exit_signal = abs(z) < exit_z

        return pd.DataFrame({
            'spread': self.spread,
            'zscore': z,
            'long_entry': long_entry,
            'short_entry': short_entry,
            'exit': exit_signal,
            'half_life': hl
        })

# --- TEST ---
if __name__ == "__main__":
    # Simulate mean-reverting spread with hl ~ 5 days
    np.random.seed(42)
    n = 200
    spread = [0]
    theta = 0.15 # ~ hl 4.6 days
    for i in range(1, n):
        spread.append(spread[-1] + theta*(0-spread[-1]) + np.random.randn()*0.5)

    spread = pd.Series(spread)
    ou = OUProcess(spread)
    info = ou.calc_half_life()
    print(f"Half-life: {info['half_life']:.2f} periods")
    print(f"Mean-reverting? {info['is_mean_reverting']}")
    # Tradable if 1 < hl < 20
