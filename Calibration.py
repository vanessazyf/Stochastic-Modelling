import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import datetime as dt
from part1 import *
from scipy import interpolate
from scipy.optimize import least_squares
call_df = pd.read_csv("stock_call.csv")
put_df = pd.read_csv("stock_put.csv")
discount = pd.read_csv('discount.csv')

def impliedCallVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -
                        BlackScholesCall(S, K, r, x, T),
                        1e-6, 1)
    return impliedVol
def impliedPutVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -
                        BlackScholesPut(S, K, r, x, T),
                        1e-6, 1)
    return impliedVol
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma
today = dt.date(2013, 8, 30)
expiry = dt.date(2015, 1, 17)
S =  #today's stock prive
T = (expiry-today).days/365.0
f1 = interpolate.interp1d(discount.iloc[:, 0], discount.iloc[:, 1], kind='linear')
r = float(f1(T*365))/100
F = S * np.exp(r * T)
call_df['midprice'] = (call_df['best_bid'] + call_df['best_offer'])/2
put_df['midprice'] = (put_df['best_bid'] + put_df['best_offer'])/2  

calls = list(call_df.midprice[call_df.strike > F])
call_strike = list(call_df.strike[call_df.strike > F])
puts = list(put_df.midprice[put_df.strike < F])
put_strike = list(put_df.strike[put_df.strike < F])

'''----------------------------------Market(BS)------------------------------------'''
summary = []
for i in range(len(call_strike)):
    impliedvol = impliedCallVolatility(S, call_strike[i], r, calls[i], T)    
    summary.append([call_strike[i], impliedvol])  
for i in range(len(put_strike)):
    impliedvol = impliedPutVolatility(S, put_strike[i], r, puts[i], T)    
    summary.append([put_strike[i], impliedvol])  
df = pd.DataFrame(summary, columns=['strike', 'vol'])
#plt.plot(df['strike'], df['vol'],'g*',label = 'Market')
f2 = interpolate.interp1d(df.strike, df.vol, kind='zero')
sigma_log = f2(F)
sigma = sigma_log

'''----------------------------------Displaced Diffusion------------------------------------'''
def ddrcalibration(beta):
    call_DD = []
    [call_DD.append(DD_vanilla_call(S, K, r, sigma, T, beta)) for K in call_strike]
    put_DD = []
    [put_DD.append(DD_vanilla_put(S, K, r, sigma, T, beta)) for K in put_strike]       
    summary1 = []
    for i in range(len(call_strike)):
        impliedvol = impliedCallVolatility(S, call_strike[i], r, call_DD[i], T)    
        summary1.append([call_strike[i], impliedvol])
    for i in range(len(put_strike)):
        impliedvol = impliedPutVolatility(S, put_strike[i], r, put_DD[i], T)    
        summary1.append([put_strike[i], impliedvol]) 
    summary1 = np.array(summary1)
    error= np.sum((summary1[:,1] -  df['vol'].values)**2)
    return error    
res = least_squares(lambda x: ddrcalibration(x),0.001)
beta_dd=res.x[0]
call_DD = []
[call_DD.append(DD_vanilla_call(S, K, r, sigma, T, beta_dd)) for K in call_strike]
put_DD = []
[put_DD.append(DD_vanilla_put(S, K, r, sigma, T, beta_dd)) for K in put_strike] 
summary1 = []
for i in range(len(put_strike)):
    impliedvol = impliedPutVolatility(S, put_strike[i], r, put_DD[i], T)    
    summary1.append([put_strike[i], impliedvol]) 
for i in range(len(call_strike)):
    impliedvol = impliedCallVolatility(S, call_strike[i], r, call_DD[i], T)    
    summary1.append([call_strike[i], impliedvol])
df1 = pd.DataFrame(summary1, columns=['strike', 'vol'])
#plt.plot(df1['strike'], df1['vol'],'--',color = 'coral',label = 'Displaced Diffusion(beta=%.4f)'%beta_dd)

'''---------------------------------Normal------------------------------------'''
call_B76 = []
[call_B76.append(B76_call_normal(S, K, r, sigma, T)) for K in call_strike]
put_B76 = []
[put_B76.append(B76_put_normal(S, K, r, sigma, T)) for K in put_strike]   
summary2 = []
for i in range(len(put_strike)):
    impliedvol = impliedPutVolatility(S, put_strike[i], r, put_B76[i], T)    
    summary2.append([put_strike[i], impliedvol]) 
for i in range(len(call_strike)):
    impliedvol = impliedCallVolatility(S, call_strike[i], r, call_B76[i], T)    
    summary2.append([call_strike[i], impliedvol])   
df2 = pd.DataFrame(summary2, columns=['strike', 'vol'])
plt.plot(df2['strike'], df2['vol'],'b--',label = 'Bachelier model') 

'''-----------------------------------------BS(Lognormal)--------------------------------------''' 
call_BS = []
[call_BS.append(BlackScholesCall(S, K, r, sigma, T)) for K in call_strike]
put_BS = []
[put_BS.append(BlackScholesPut(S,K,r,sigma,T)) for K in put_strike]   
summary3 = []
for i in range(len(put_strike)):
    impliedvol = impliedPutVolatility(S, put_strike[i], r, put_BS[i], T)    
    summary3.append([put_strike[i], impliedvol]) 
for i in range(len(call_strike)):
    impliedvol = impliedCallVolatility(S, call_strike[i], r, call_BS[i], T)    
    summary3.append([call_strike[i], impliedvol])   
df3 = pd.DataFrame(summary3, columns=['strike', 'vol'])
plt.plot(df3['strike'], df3['vol'],'k--',label = 'Black model') 

'''------------------------------------------SABR--------------------------------------------------'''
beta = #the value of fixed beta
def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T, x[0], 0.8, x[1], x[2]))**2
    return err
initialGuess = [0.02, 0.2, 0.1]
res = least_squares(lambda x: sabrcalibration(x,df['strike'].values,df['vol'].values,F,T),initialGuess)
alpha = res.x[0]
rho = res.x[1]
nu = res.x[2] 
summary4 = []
for i in range(len(put_strike)):
    impliedvol = SABR(F, put_strike[i], T, alpha, beta, rho, nu)    
    summary4.append([put_strike[i], impliedvol]) 
for i in range(len(call_strike)):
    impliedvol = SABR(F, call_strike[i], T, alpha, beta, rho, nu)   
    summary4.append([call_strike[i], impliedvol])   
df4 = pd.DataFrame(summary4, columns=['strike', 'vol'])
#plt.plot(df4['strike'], df4['vol'],'--',color = 'orange',label = 'SABR(alpha=%.4f,rho=%.4f,nu=%.4f)'%(alpha,rho,nu)) 

plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.legend()
plt.show()
