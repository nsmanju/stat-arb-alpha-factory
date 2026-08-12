import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42); n=250; mu=0; theta=0.1457; sigma=0.5
hl=np.log(2)/theta
x=np.zeros(n); x[0]=3
for i in range(1,n):
    x[i]=x[i-1]+theta*(mu-x[i-1])+np.random.randn()*sigma
plt.figure(figsize=(10,4), dpi=150)
plt.plot(x, linewidth=1.5); plt.axhline(mu, color='r', linestyle='--')
plt.fill_between(range(n), mu-1, mu+1, alpha=0.15, color='gray')
plt.title(f'OU Process: dX = theta*(mu-X)dt + sigma dW | HL={hl:.1f} days', fontweight='bold')
plt.xlabel('Days'); plt.ylabel('Spread'); plt.tight_layout()
plt.savefig('notebooks/images/ou_diagram.png')
print(f"Saved HL={hl:.2f}")
