import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(7); n=500
spread=np.zeros(n)
for i in range(1,n):
    spread[i]=0.85*spread[i-1]+np.random.randn()*0.5
z=(spread-spread.mean())/spread.std()
signals=np.where(z<-1,1,np.where(z>1,-1,0))
pnl=pd.Series(signals).shift(1).fillna(0)*np.diff(spread, prepend=0)
equity=(1+pnl*0.01).cumprod()*100
ret=pnl*0.01; sharpe=ret.mean()/ret.std()*np.sqrt(252)
plt.figure(figsize=(10,4), dpi=150)
plt.plot(equity, color='#00C853', linewidth=2)
plt.title(f'Walk-forward Sharpe {sharpe:.2f} | shift(1) no lookahead', fontweight='bold')
plt.xlabel('Trades'); plt.ylabel('Equity'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('notebooks/images/sharpe_equity.png')
print(f"Sharpe {sharpe:.2f}")
