import numpy as np
import pandas as pd
from factors import AlphaFactors
from cointegration import CointegrationTest
from ou_process import OUProcess

class StatArbFactory:
    def run(self):
        np.random.seed(42)
        n=500
        btc = pd.Series(np.cumsum(np.random.randn(n)))
        spread = pd.Series([0.0]*n, dtype=float)
        for i in range(1,n):
            spread.iloc[i] = 0.85*spread.iloc[i-1] + np.random.randn()*0.5
        eth = 1.5*btc + spread

        print("=== Step1: Cointegration ===")
        test = CointegrationTest(btc, eth)
        coint = test.engle_granger_test()
        print(f"p={coint['p_value']:.8f} Cointegrated? {coint['is_cointegrated']}")

        print("\n=== Step2: Hedge Ratio ===")
        hedge = test.get_hedge_ratio()
        print(f"beta={hedge['beta']:.4f}")

        print("\n=== Step3: OU Half-life ===")
        ou = OUProcess(hedge['spread'])
        hl = ou.calc_half_life()
        print(f"HL={hl['half_life']:.2f} days, mean-reverting={hl['is_mean_reverting']}")

        print("\n=== Step4: Factors ===")
        prices = pd.DataFrame({'BTC': btc, 'ETH': eth})
        af = AlphaFactors(prices)
        fac = af.generate_all()
        print(f"Factors shape {fac.shape}")

        print("\n=== ALL OK - Ready for HK interviews ===")

if __name__ == "__main__":
    StatArbFactory().run()
