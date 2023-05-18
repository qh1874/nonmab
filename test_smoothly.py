import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Bernoulli_smoothly
from utils import plot_mean_arms, traj_arms,save_data
from param_s import *

T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times
sigma=param['sigma']
# Keep the distribution of arms consistent each run
seed=0
arm_start,param_start,chg_dist=generate_arm_Bernoulli_smoothly(T,K,sigma,seed)


mab = GMAB(arm_start, param_start, chg_dist)

store_step=1
#mydata0=np.load('smooth/a5b1e4/DS_TS_data0.npy',allow_pickle=True)
#mydata1=np.load('smooth/a5b1e4/DS_TS_data1.npy',allow_pickle=True)
EXP3S_data = mab.MC_regret("EXP3S", N, T, param_exp3s, store_step=store_step)
DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=store_step)
SW_TS_data = mab.MC_regret('SW_TS', N, T, {'tau':int(tau_theorique)},store_step=store_step)
DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=store_step)
##DS_TS_data = mab.MC_regret('DTS', N, T, param_dsts_b,store_step=store_step)
TS_data = mab.MC_regret('DTS', N, T,{'gamma':1},store_step=store_step)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=store_step)

# np.save('smooth/a5b1e4/DS_TS_data0.npy',DS_TS_data[0])
# np.save('smooth/a5b1e4/DS_TS_data1.npy',DS_TS_data[1])


rr=np.zeros(8)
L=['EXP3S_data','DS_UCB_data','SW_TS_data','TS_data','LBSDA_data','DS_TS_data']
ii=0
for i in L:
    print(i+":",eval(i)[0][-1])
    rr[ii]=eval(i)[0][-1]
    ii += 1


print("T : {}, arms : {}, sigma: {} ".format(T, K, sigma))
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

DS_UCB_data1=DS_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_UCB_data[0][xx], '-bd', markerfacecolor='none', label='DS-UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='b')

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
#plt.title("T : {}, arms : {}, sigma: {} ".format(T, K, sigma))
plt.xlabel('Round t')
plt.ylabel('Regret')
#plt.savefig('result.pdf')
plt.show()






