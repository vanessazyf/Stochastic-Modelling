import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Given:
S_0 = 100
sigma = 0.2
r = 0.05 
K = 100 
T = 1/12

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r-sigma**2/2)*T)/(sigma * np.sqrt(T))
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call

def PhiT(S, r, T, sigma):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
    return norm.cdf(d1)


def PsiBT(S, K, r, T, sigma):
    d2 = (np.log(S/K) + (r-sigma**2/2)*T)/(sigma * np.sqrt(T))
    return K*np.exp(-r*T)*norm.cdf(d2)


def hedge_error(S_0, K, r, T, sigma, N):
    s=0
    S_Next=S_0
    for i in range(1,N):
        TTM=T*(1-i/N)
        S = S_Next
        phit = PhiT(S, r, TTM, sigma)
        psibt = PsiBT(S, K, r, TTM, sigma)
        S_Next = S*np.exp((r-sigma**2/2)*T/N+sigma*np.random.randn()*np.sqrt(T/N))
        err = np.exp(r*abs(TTM-T/N))*((S_Next*phit-psibt*np.exp(r*T/N))-BlackScholesCall(S_Next, K, r, sigma,abs(TTM-T/N)))
        s=s+err
    return s

#loop for path=50000 when N = x, x is definded by yourself
N_x = []
for i in range(50000):
    ERR=hedge_error(S_0, K, r, T, sigma, x)
    N_x.append(ERR)

#plot hist when N = x
plt.figure(figsize=(10,8))
plt.xticks(np.arange(-1.5,1.5,0.5))
plt.xlim(-1.5,1.5)
plt.title('Histogram of P&L when N = x',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xlabel('Profit&Loss',fontsize=15)
plt.hist(N_x,color='g',bins=55)
plt.grid(which='major')
plt.show()

N_x=pd.DataFrame(N_21)
print('The mean of Hedging Error when N=x:',N_x.mean())
print('The Standard Deviation of Hedging Error when N=x:',N_x.std())
