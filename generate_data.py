import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

def get_reward_distribution_Bernoulli(T,K,m,seed):
    np.random.seed(seed)
    test_bernoulli=np.zeros((T,K))
    for ii in range(0,T,m):
        x=np.random.uniform(0.,0.6,K)
        a=np.random.randint(0,K)
        x[a]=0.7#np.random.uniform(0.50,0.55)
        test_bernoulli[ii:ii+m]=np.random.uniform(0,1,K)
    ####################
    Delta=np.zeros(K)
    xmin=np.min(test_bernoulli,1)
    for i in range(K):
        x=test_bernoulli[:,i]
        xx=x-xmin
        xx[xx==0]=1
        Delta[i]=min(xx)
    ####################
    return test_bernoulli,Delta

def get_reward_distribution_Bernoulli_delta(T,K,m,seed,delta):
    np.random.seed(seed)
    test_bernoulli=np.zeros((T,K))
    for ii in range(0,T,m):
        #delta=np.random.uniform(0.0001,0.01)
        tmp=np.random.uniform(0.5,1)*np.ones(K)
        index=np.random.randint(0,K)
        tmp -= delta
        tmp[index] += delta
        test_bernoulli[ii:ii+m]=tmp
    #################### 
    Delta=np.zeros(K)
    xmin=np.min(test_bernoulli,1)
    for i in range(K):
        x=test_bernoulli[:,i]
        xx=x-xmin
        xx[xx==0]=1
        Delta[i]=min(xx)
    ####################
    return test_bernoulli,Delta

def get_reward_distribution_Bernoulli_smoothly(T,K,sigma,seed):
    np.random.seed(seed)
    test_bernoulli=np.zeros((T,K))
    
    for t in range(T):
        w=1+(K-1)*(1+np.sin(t*sigma))/2
        for i in range(K):
            test_bernoulli[t,i] = ((K-1)/K-abs(w-i-1)/K)#*K/2/(K-1)
    
    return test_bernoulli


def get_reward_distribution_Finite(T,K,m,seed):
    np.random.seed(seed)
    dim=5
    test_finite=np.zeros((T,K,2,dim))
    xmean=np.zeros((T,K))
    for ii in range(0,T,m):
        xp=np.random.uniform(0,1,(K,2,dim))
        p=xp[:,1,:]
        #ptemp=p/p.sum(1).reshape(-1,1)
        xx=np.random.uniform(1,10,5)
        ptemp=xx/np.sum(xx)
        xp[:,1,:]=ptemp
        test_finite[ii:ii+m]=xp
        xmean[ii:ii+m]=(xp[:,0,:]*xp[:,1,:]).sum(1) ##
    ####################
    Delta=np.zeros(K)
    xmin=np.min(xmean,1)
    for i in range(K):
        x=xmean[:,i]
        xx=x-xmin
        xx[xx==0]=1
        Delta[i]=min(xx)
    ###############
    return test_finite,Delta

def generate_arm_Bernoulli(T,K,m,seed):
    test_bernoulli,Delta=get_reward_distribution_Bernoulli(T,K,m,seed)
    KB=['B' for _ in range(K)]
    arm_start=KB
    param_start=test_bernoulli[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KB,test_bernoulli[i].tolist()]
    
    plt.figure(1)
    for i in range(K):
        plt.plot(test_bernoulli[:,i],label='Arm '+str(i+1))
    #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    plt.xlabel('Round t')
    plt.ylabel(r'$\mu_t(i)$')
    return arm_start,param_start,chg_dist

def generate_arm_Bernoulli_smoothly(T,K,sigma,seed):
    test_bernoulli=get_reward_distribution_Bernoulli_smoothly(T,K,sigma,seed)
    KB=['B' for _ in range(K)]
    arm_start=KB
    param_start=test_bernoulli[0].tolist()
    chg_dist={}
    for i in range(1,T):
        chg_dist[str(i)]=[KB,test_bernoulli[i].tolist()]
    
    plt.figure(1)
    for i in range(K):
        plt.plot(test_bernoulli[:,i],label='Arm '+str(i+1))
    #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    plt.xlabel('Round t')
    plt.ylabel(r'$\mu_t(i)$')
    #plt.show()
    return arm_start,param_start,chg_dist

def generate_arm_Bernoulli_delta(T,K,m,seed,delta):
    test_bernoulli,Delta=get_reward_distribution_Bernoulli_delta(T,K,m,seed,delta)
    KB=['B' for _ in range(K)]
    arm_start=KB
    param_start=test_bernoulli[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KB,test_bernoulli[i].tolist()]
    
    return arm_start,param_start,chg_dist,Delta

def generate_arm_Finite(T,K,m,seed):
    test_finite,Delta=get_reward_distribution_Finite(T,K,m,seed)
    KF=['F' for _ in range(K)]
    arm_start=KF
    param_start=[list(i) for i in list(test_finite[0])]
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KF,[list(ii) for ii in list(test_finite[i])] ]
        
    plt.figure(1)
    for i in range(K):
        plt.plot(np.sum(test_finite[:,i,1,:]*test_finite[:,i,0,:],1),label='Arm '+str(i+1))
    plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    return arm_start,param_start,chg_dist

