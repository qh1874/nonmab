import numpy as np

param={
    'T':10000, # round 
    'K':20,  # arm
    'm':10000, # length of stationary phase, breakpoints=T/m
    'sigma':0.001,
    'N':10 # repeat times
}

T=param['T']
K=param['K']
m=param['m']
nb_change=int(T/m)
Gamma_T_garivier = int(T/m)
reward_u_p = 1
expected_reward_u_p=1
sigma_max = 1
gamma_EXP3 = min(1, 0.1*np.sqrt(K*(nb_change*np.log(K*T)+np.exp(1))/((np.exp(1)-1)*T)))
gamma_D_UCB = 1 - 1/(4*reward_u_p)*np.sqrt(Gamma_T_garivier/T)
gamma_D_UCB_unb = 1 - 1/(4*(reward_u_p+ 2*sigma_max))*np.sqrt(Gamma_T_garivier/T)
tau_theorique = 2*reward_u_p*np.sqrt(T*np.log(T)/Gamma_T_garivier)
tau_theorique_unb = 2*(reward_u_p + 2*sigma_max)*np.sqrt(T*np.log(T)/Gamma_T_garivier)
tau_no_log = 2*reward_u_p*np.sqrt(T/Gamma_T_garivier)

#CUMSUM
h_CUSUM = np.log(T/Gamma_T_garivier)
alpha_CUSUM = np.sqrt(Gamma_T_garivier/T*h_CUSUM)
M_CUSUM = 50
eps_CUSUM = 0.05

#M-UCB
w_BRANO = 800
b_BRANO = np.sqrt(w_BRANO/2*np.log(2*K*T**2))
gamma_MUCB = np.sqrt(Gamma_T_garivier*np.log(T)*K/T)
delta_min = 0.2

param_exp3s={'alpha':1/T, 'gamma': gamma_EXP3}
param_cusum={'alpha':alpha_CUSUM , 'h': h_CUSUM, 'M':M_CUSUM, 'eps':eps_CUSUM, 'ksi':1/2}
param_dsucb={'B':1,'ksi':2/3, 'gamma': gamma_D_UCB_unb}
param_swucb={'C': sigma_max*np.sqrt(2), 'tau': int(tau_theorique_unb)}
param_swts={'mu_0':0, 'sigma_0':sigma_max, 'sigma':sigma_max, 'tau': int(tau_theorique)}
param_dsts={'kexi':1, 'tao_max':expected_reward_u_p/5, 'gamma': 1-10*np.sqrt(nb_change/T)}
param_dsts_b={'gamma': 1-np.sqrt(nb_change/T),'alpha_0':1,'beta_0':1}
param_lbsda={'tau': int(tau_theorique)}
param_cumsum={'alpha':alpha_CUSUM , 'h': h_CUSUM, 'M':M_CUSUM, 'eps':eps_CUSUM, 'ksi':1/2}
param_mucb={'w':w_BRANO, 'b':b_BRANO, 'gamma':gamma_MUCB}
