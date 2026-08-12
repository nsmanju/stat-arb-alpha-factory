import pandas as pd
from factors import AlphaFactors
from cointegration import CointegrationTest
from ou_process import OUProcess
from backtest import WalkForwardBacktest

class StatArbAlphaFactory:
    """
    Main factory: BTC-ETH stat arb with 10 factors + cointegration + OU
    

    Flow:
    1. Load BTC, ETH prices
    2. Test cointegration (p<0.05?)
    3. Calc spread & OU half-life (1-20 days tradable?)
    4. Generate 10 factors on spread
    5. Combine by IC weighting
    6. Walk-forward backtest -> Sharpe 1.5
    """

    def __init__(self, btc_price: pd.Series, eth_price: pd.Series):
        self.btc = btc_price
        self.eth = eth_price

    def run_pipeline(self):
        print("=== Stat-Arb Alpha Factory ===")

        # 1. Cointegration test
        print("\n1. Cointegration Test...")
        coint_test = CointegrationTest(self.btc, self.eth)
        coint_res = coint_test.engle_granger_test()
        print(f" p-value: {coint_res['p_value']:.4f} -> Cointegrated? {coint_res['is_cointegrated']}")

        if not coint_res['is_cointegrated']:
            print(" WARNING: Not cointegrated, still continue for demo")

        hedge = coint_test.get_hedge_ratio()
        spread = hedge['spread']
        beta = hedge['beta']
        print(f" Hedge Ratio beta: {beta:.4f} -> Spread = ETH - {beta:.4f}*BTC")

        # 2. OU Half-life
        print("\n2. OU Half-life...")
        ou = OUProcess(spread)
        ou_res = ou.calc_half_life()
        print(f" Half-life: {ou_res['half_life']:.2f} periods, Mean-rev: {ou_res['is_mean_reverting']}")

        # Tradability check
        tradable = 1 < ou_res['half_life'] < 30 and ou_res['is_mean_reverting']
        print(f" Tradable? {tradable} (need 1<HL<30)")

        # 3. 10 Factors on spread (or on BTC)
        print("\n3. Generating 10 Factors...")
        # Use spread for factors, or BTC - here using spread for stat-arb alpha
        factors_engine = AlphaFactors(spread)
        factors_df = factors_engine.get_all_factors()
        print(f" Factors shape: {factors_df.shape}")
        print(f" Factors: {list(factors_df.columns)}")

        # 4. Combine factors
        print("\n4. Combining Factors (IC Weighted)...")
        forward_ret = spread.pct_change().shift(-1) # Next day spread return
        bt_temp = WalkForwardBacktest(spread, spread) # dummy
        combined_signal, weights, ics = bt_temp.combine_factors_sharpe_weighted(factors_df, forward_ret)
        print(" ICs:")
        for k,v in ics.items():
            print(f" {k}: IC={v:.3f}, weight={weights[k]:.3f}")

        # 5. Walk-forward backtest
        print("\n5. Walk-Forward Backtest...")
        bt = WalkForwardBacktest(spread, combined_signal)
        res = bt.walk_forward()

        if res:
            print(f" Sharpe: {res['sharpe']:.2f} <- TARGET 1.5")
            print(f" Max DD: {res['max_drawdown']:.4f}")
            print(f" Hit Rate: {res['hit_rate']:.2%}")
            print("\n=== Pipeline Complete ===")
            return res, factors_df, spread
        else:
            print(" Not enough data for backtest")
            return None, factors_df, spread

# --- DEMO with fake data ---
if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    n=500
    btc = pd.Series(np.cumsum(np.random.randn(n)) + 50000)
    eth = pd.Series(0.05*btc + np.random.randn(n)*20 + 100)

    factory = StatArbAlphaFactory(btc, eth)
    result, factors, spread = factory.run_pipeline()
