import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Bernoulli,generate_arm_Finite
from utils import plot_mean_arms, traj_arms,save_data
from param import *


T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times

# Keep the distribution of arms consistent each run
seed=0
arm_start,param_start,chg_dist=generate_arm_Bernoulli(T,K,m,seed)

# arm_start, param_start =['G', 'G', 'G'], [[0.9,0.5], [0.5,0.5], [0.4,0.5]]
# chg_dist = {'2500': [['G', 'G', 'G'], [[0.4,0.5], [0.8,0.5], [0.5,0.5]]],
#             '4500': [['G', 'G', 'G'], [[0.3,0.5], [0.2,0.5], [0.7,0.5]]],
#             '7000': [['G', 'G', 'G'], [[0.9,0.5], [0.8,0.5], [0.4,0.5] ]]
#            }

mab = GMAB(arm_start, param_start, chg_dist)
store_step=1
#mydata0=np.load('abrupt/a10b10/DS_TS_data0.npy',allow_pickle=True)
#mydata1=np.load('abrupt/a10b10/DS_TS_data1.npy',allow_pickle=True)
EXP3S_data = mab.MC_regret("EXP3S", N, T, param_exp3s, store_step=store_step)
DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=store_step)
SW_TS_data = mab.MC_regret('SW_TS', N, T, {'tau':int(tau_theorique)},store_step=store_step)
DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=store_step)
###DS_TS_data = mab.MC_regret('DTS', N, T, param_dsts_b,store_step=store_step)
TS_data = mab.MC_regret('DTS', N, T,{'gamma':1},store_step=store_step)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=store_step)
CUSUM_data = mab.MC_regret('CUSUM', N, T, param_cumsum,store_step=store_step)
M_UCB_data = mab.MC_regret('M_UCB', N, T, param_mucb,store_step=store_step)

# np.save('abrupt/a20b10/DS_TS_data0.npy',DS_TS_data[0])
# np.save('abrupt/a20b10/DS_TS_data1.npy',DS_TS_data[1])



rr=np.zeros(8)
L=['EXP3S_data','CUSUM_data','DS_UCB_data','SW_TS_data','TS_data','LBSDA_data','M_UCB_data','DS_TS_data']
ii=0
for i in L:
    print(i+":",eval(i)[0][-1])
    rr[ii]=eval(i)[0][-1]
    ii += 1
#np.save("b_a50b20.npy",rr)

print("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
x=np.arange(T)
d = int(T / 20)
dd=int(T/1000)
xx = np.arange(0, T, d)
xxx=np.arange(0,T,dd)
alpha=0.05
plt.figure(2)
EXP3S_data1=EXP3S_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(EXP3S_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, EXP3S_data[0][xx], '-g^', markerfacecolor='none', label='EXP3S')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='g')

LBSDA_data1=LBSDA_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(LBSDA_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='SW-LB-SDA')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='c')

CUSUM_data1=CUSUM_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(CUSUM_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, CUSUM_data[0][xx], '-k^', markerfacecolor='none', label='CUSUM')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='k')

DS_UCB_data1=DS_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_UCB_data[0][xx], '-bd', markerfacecolor='none', label='DS-UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='b')

M_UCB_data1=M_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(M_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, M_UCB_data[0][xx], '-ms', markerfacecolor='none', label='M-UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='m')

TS_data1=TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, TS_data[0][xx], color='brown',marker='*', markerfacecolor='none', label='TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='brown')

SW_TS_data1=SW_TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(SW_TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, SW_TS_data[0][xx], '-y^', markerfacecolor='none', label='SW-TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='y')

DS_TS_data1=DS_TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_TS_data[0][xx], '-ro', markerfacecolor='none', label='DS-TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='r')

# mydata1=mydata1.T[:,xxx]
# low_bound, high_bound = sms.DescrStatsW(mydata1).tconfint_mean(alpha=alpha)
# plt.plot(xx, mydata0[xx], '-rd', markerfacecolor='none', label=r'DS-TS(known $\mu_{max}$)')
# plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='r')

plt.legend()
#plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
plt.xlabel('Round t')
plt.ylabel('Regret')
plt.show()






import scipy.stats as st


def kl(a,b):
    return a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))

def f11(x,y,k,alpha_0,beta_0):
    return st.beta.cdf((k*x+alpha_0)/(k+alpha_0+beta_0)-y,x*k+alpha_0,(1-x)*k+beta_0)

def f1(x,y,k,alpha_0,beta_0):   
    return 1-st.beta.cdf(y,x*k+alpha_0,(1-x)*k+beta_0)

def f2(x,y,k,alpha_0,beta_0): 
    return np.exp(-(k+alpha_0+beta_0-1)*kl((k*x+alpha_0-1)/(k+alpha_0+beta_0-1),y))

def f3(x,y,k,alpha_0,beta_0):   
     return 1/(2*(k+alpha_0+beta_0-1))**0.5*np.exp(-(k+alpha_0+beta_0-1)*kl((k*x+alpha_0-1)/(k+alpha_0+beta_0-1),y))

def test(T,k,alpha_0,beta_0):
    x=np.random.uniform(0,1,T)
    y=np.random.uniform(0,1,T)
    xx=x.copy()
    x=np.minimum(x,y)
    y=np.maximum(xx,y)
    x1=np.sum(f1(x,y,k,alpha_0,beta_0)<f2(x,y,k,alpha_0,beta_0)+1e-10)
    x2=np.sum(f2(x,y,k,alpha_0,beta_0)+1e-10>f3(x,y,k,alpha_0,beta_0))

    return x1,x2,x,y