# Stat-Arb Alpha Factory - BTC/ETH

> Walk-forward, no lookahead, cointegration-driven stat-arb pipeline for HK.

## Pipeline All Green

1. Cointegration: p=0.00000048 True, beta=1.52 ~1.5
2. OU Process: HL=4.76 days, Beta=-0.1457 mean-reverting
3. Factors: Mom 5/20, MR 20, Vol 20, Z 20
4. Backtest: shift(1) no lookahead
5. Factory: ALL OK

## Run
source venv/bin/activate
python src/cointegration.py
python src/ou_process.py
python src/factors.py
python src/backtest.py
python src/alpha_factory.py

Author: Nadkalpur Manjunath

## Visuals
### OU Mean Reversion HL=4.76 days
![OU](notebooks/images/ou_diagram.png)

### Walk-forward Equity (No Lookahead)
![Equity](notebooks/images/sharpe_equity.png)
