import numpy as np
from scipy.stats import norm

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r-sigma**2/2)*T)/(sigma * np.sqrt(T))
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call

def BlackScholesPut(S,K,r,sigma,T):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r-sigma**2/2)*T)/(sigma * np.sqrt(T))
    put = - S * norm.cdf(-d1) + K * np.exp(-r*T) * norm.cdf(-d2)
    return put

def BS_cash_call(S,K,r,sigma,T):
    d2 = (np.log(S/K) + (r-sigma**2/2)*T)/(sigma * np.sqrt(T))
    call = np.exp(-r*T) * norm.cdf(d2)
    return call

def BS_cash_put(S,K,r,sigma,T):
    d2 = (np.log(S/K) + (r-sigma**2/2)*T)/(sigma * np.sqrt(T))
    put = np.exp(-r*T) * norm.cdf(-d2)
    return put

def BS_asset_call(S,K,r,sigma,T):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
    call = S * norm.cdf(d1)
    return call

def BS_asset_put(S,K,r,sigma,T):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
    put = S * norm.cdf(-d1)
    return put

def B_vanilla_call(S,K,r,sigma,T):
    d1 = (S - K)/(S * sigma * np.sqrt(T)) 
    call = (S - K) * norm.cdf(d1) + S * sigma * np.sqrt(T) * norm.pdf(d1)
    return call

def B_vanilla_put(S,K,r,sigma,T):
    d1 = (S - K)/(S * sigma * np.sqrt(T)) 
    put = - (S - K) * norm.cdf(-d1) - S * sigma * np.sqrt(T) * norm.pdf(-d1)
    return put

def B76_call_normal(S, K, r, sigma, T):
    F=S*np.exp(r*T)
    D=np.exp(-r*T)
    d1 = (K - F) / (sigma*F*np.sqrt(T))
    call=(F - K)*norm.cdf(-d1)+ (sigma*F*np.sqrt(T))*norm.pdf(d1)
    return D*call

def B76_put_normal(S, K, r, sigma, T):
    F=S*np.exp(r*T)
    D=np.exp(-r*T)
    d1 = (K - F) / (sigma*F*np.sqrt(T))
    put=-(F - K)*norm.cdf(d1) + (sigma*F*np.sqrt(T))*norm.pdf(-d1)
    return D*put

def B_cash_call(S,K,r,sigma,T):
    d1 = (S - K)/(S * sigma * np.sqrt(T)) 
    call = norm.cdf(d1) 
    return call

def B_cash_put(S,K,r,sigma,T):
    d1 = (S - K)/(S * sigma * np.sqrt(T)) 
    put = norm.cdf(-d1) 
    return put

def B_asset_call(S,K,r,sigma,T):
    d1 = (S - K)/(S * sigma * np.sqrt(T)) 
    call = S * norm.cdf(d1) + S * sigma * np.sqrt(T) * norm.pdf(d1)
    return call

def B_asset_put(S,K,r,sigma,T):
    d1 = (S - K)/(S * sigma * np.sqrt(T)) 
    put = S * norm.cdf(-d1) + S * sigma * np.sqrt(T) * norm.pdf(-d1)
    return put

def B76_vanilla_call(S, K, r, sigma, T):
    F=S*np.exp(r*T)
    D=np.exp(-r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = (np.log(F/K)-(sigma**2/2)*T) / (sigma*np.sqrt(T))
    temp=F*norm.cdf(d1)-K*norm.cdf(d2)
    return D*temp

def B76_vanilla_put(S, K, r, sigma, T):
    F=S*np.exp(r*T)
    D=np.exp(-r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = (np.log(F/K)-(sigma**2/2)*T) / (sigma*np.sqrt(T))
    temp=K*norm.cdf(-d2)-F*norm.cdf(-d1)
    return D*temp

def B76_cash_call(S, K, r, sigma, T):
    d1=np.log(S/K)+(-sigma**2/2)*T
    d2=d1/(sigma*np.sqrt(T))
    temp=np.exp(-r*T)*norm.cdf(d2)
    return temp

def B76_cash_put(S, K, r, sigma, T):
    d1=np.log(S/K)+(-sigma**2/2)*T
    d2=d1/(sigma*np.sqrt(T))
    temp=np.exp(-r*T)*norm.cdf(-d2)
    return temp

def B76_asset_call(S, K, r, sigma, T):
    d1=np.log(S/K)+(+sigma**2/2)*T
    d2=d1/(sigma*np.sqrt(T))
    temp=S*norm.cdf(d2)
    return temp

def B76_asset_put(S, K, r, sigma, T):
    d1=np.log(S/K)+(+sigma**2/2)*T
    d2=d1/(sigma*np.sqrt(T))
    temp=S*norm.cdf(-d2)
    return temp

def DD_vanilla_call(S, K, r, sigma, T, beta):
    F = S * np.exp(r * T)
    d1 = (np.log((F / beta) / (K + (1 - beta) * F/ beta)) + ((sigma * beta)** 2 * T / 2)) / (sigma * beta * np.sqrt(T))
    d2 = (np.log((F / beta) / (K + (1 - beta) * F/ beta)) - ((sigma * beta)** 2 * T / 2)) / (sigma * beta * np.sqrt(T))
    call = np.exp(-r * T) * (F / beta * norm.cdf(d1) - (K + (1 - beta) * F/ beta) * norm.cdf(d2))
    return call

def DD_vanilla_put(S, K, r, sigma, T, beta):
    F = S * np.exp(r * T)
    d1 = (np.log((F / beta) / (K + (1 - beta) * F/ beta)) + ((sigma * beta)** 2 * T / 2)) / (sigma * beta * np.sqrt(T))
    d2 = (np.log((F / beta) / (K + (1 - beta) * F/ beta)) - ((sigma * beta)** 2 * T / 2)) / (sigma * beta * np.sqrt(T))
    put = np.exp(-r * T) * (-F / beta * norm.cdf(-d1) + (K + (1 - beta) * F/ beta) * norm.cdf(-d2))
    return put

    
def DD_cash_call(F0, K, r, sigma, beta, T):
    d2 = (np.log((F0 / beta) / (K + (1 - beta) / beta * F0)) - (sigma ** 2 * beta ** 2 * T / 2)) / (
                sigma * beta * np.sqrt(T))
    call = np.exp(-r * T) * norm.cdf(d2)
    return call

def DD_cash_put(F0, K, r, sigma, beta, T):
    d2 = (np.log((F0 / beta) / (K + (1 - beta) / beta * F0)) - (sigma ** 2 * beta ** 2 * T / 2)) / (
                sigma * beta * np.sqrt(T))
    put = np.exp(-r * T) * norm.cdf(-d2)
    return put

def DD_asset_call(F0, K, r, sigma, beta, T):
    d1 = (np.log((F0 / beta) / (K + (1 - beta) / beta * F0)) + (sigma ** 2 * beta ** 2 * T / 2)) / (
                sigma * beta * np.sqrt(T))
    call = np.exp(-r * T) * F0 / beta * norm.cdf(d1)
    return call

def DD_asset_put(F0, K, r, sigma, beta, T):
    d1 = (np.log((F0 / beta) / (K + (1 - beta) / beta * F0)) + (sigma ** 2 * beta ** 2 * T / 2)) / (
                sigma * beta * np.sqrt(T))
    put = np.exp(-r * T) * F0 / beta * norm.cdf(-d1)
    return put
