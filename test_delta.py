import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Bernoulli_delta
from utils import plot_mean_arms, traj_arms,save_data
from param import *

T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times


for i in range(1):
    
    dsts_data=[]
    dsucb_data=[]
    cusum_data=[]
    swts_data=[]
    seed=0 
    delta_list=np.arange(0.01,0.3,0.01)
    #delta_list=[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.25,0.3,0.35]
    for delta in delta_list:   
        print("delta = ",delta)                                      
        arm_start,param_start,chg_dist,Delta=generate_arm_Bernoulli_delta(T,K,m,seed,delta)
        mab = GMAB(arm_start, param_start, chg_dist)
        DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=1)
        dsts_data.append(DS_TS_data[0][-1])
        DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=1)
        dsucb_data.append(DS_UCB_data[0][-1])
        CUSUM_data = mab.MC_regret('CUSUM', N, T, param_cumsum,store_step=1)
        cusum_data.append(CUSUM_data[0][-1])
        SW_TS_data = mab.MC_regret('SW_TS', N, T, {'tau':int(tau_theorique)},store_step=1)
        swts_data.append(SW_TS_data[0][-1])
        #seed += 1
        
        
    print("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.plot(delta_list,np.array(dsts_data),'-ro',markerfacecolor='none', label='DS-TS')
    plt.plot(delta_list,np.array(dsucb_data),'-bd',markerfacecolor='none', label='DS-UCB')
    plt.plot(delta_list,np.array(cusum_data),'-k^',markerfacecolor='none', label='CUSUM')
    plt.plot(delta_list,np.array(swts_data),'-y^',markerfacecolor='none', label='SW-TS')
plt.legend()
plt.xlabel(r'$\Delta_T$')
plt.ylabel('Regret')
plt.show()
