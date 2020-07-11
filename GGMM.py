"""Generalized Gaussian Mixture Model."""

# Author: Zihao Wang, HKUST <zihaophys@gmail.com> 
# Date: July 11 2020

import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import logsumexp

def discrete_rng(pdf, vals):
    cdf = np.cumsum(pdf)
    u = np.random.random()
    index = sum([u > p for p in cdf])
    return vals[index]

def print_vec(v):
    print('\t'.join(['{}'.format(round(item,3)) for item in v]));

def print_mat(m):
    print('\n'.join(['\t'.join(['{}'.format(round(item,3)) for item in row]) 
                     for row in m]))

def print_information(nu, r):
    r = r.tolist()
    r_print = [x for _,x in sorted(zip(nu,r),reverse=True)]
    nu_print = sorted(nu,reverse=True)
    print("nu: ")
    print_vec(nu_print)
    print("r: ")
    print_mat(r_print)

class Platform:
    def __init__(self, I, J, H, n_j):
        self.I = I
        self.J = J
        tmp = np.random.random(I) 
        self.nu_true = tmp/sum(tmp)
        self.labels = range(1, I+1)
        self.reward = np.random.random([I, J]) 
        self.customerType = np.array([discrete_rng(self.nu_true, self.labels) for h in range(H)])
        print("==================Initialization==================")
        print_information(self.nu_true, self.reward)
        self.H = H
        self.n_j = n_j
        self.n_served = n_j * np.ones((H, J))
        n_reward = []
        # np.random.seed(314)
        for cust in range(H):
            n_reward.append([np.random.binomial(self.n_served[cust, j], self.reward[self.customerType[cust]-1, j]) \
                            for j in range(J)])
        self.n_reward = np.array(n_reward)

class EMALG:
    @staticmethod
    def e(data, ntypes, nu, r):
        # prepare the data
        n_served, n_reward = data
        n_served = np.array(n_served)
        n_reward = np.array(n_reward)
        H, J  = n_served.shape
        I = ntypes

        # calculate the posterior distribution for each customer
        log_weights = np.log(nu)
        log_prob = np.empty((H, I))
        for h in range(H):
            for i in range(I):
                variance = n_served[h] * r[i] * (1 - r[i]) + np.finfo(r.dtype).eps
                log_prob[h, i] = np.sum((n_reward[h] - n_served[h]*r[i])**2 / (2*variance) + 0.5*np.log(variance))
        log_prob = -log_prob - 0.5 * J * np.log(2*np.pi)
        
        weighted_log_prob = log_prob + log_weights
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        
        log_post = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return np.exp(log_post), np.mean(log_prob_norm)


    @staticmethod
    def m(data, ntypes, q_h):
        #prepare the data
        n_served, n_reward = data
        n_served = np.array(n_served)
        n_reward = np.array(n_reward)
        H, J  = n_served.shape
        I = ntypes

        # update nu and r
        r = np.empty((I, J))
        for i in range(I):
            for j in range(J):
                numerator = q_h[:, i].dot(n_reward[:, j])
                denominator = q_h[:, i].dot(n_served[:, j])
                r[i, j] = numerator/(denominator + np.finfo(denominator.dtype).eps)
        
        return np.mean(q_h, axis=0), r


if __name__ == "__main__":
    I = 3
    J = 4
    H = 100
    n_j = 30
    num_em = 20

    platform = Platform(I, J, H, n_j)
    ll = []
    
    r = np.random.random([I, J])
    nu = np.random.random(I) 
    data = [platform.n_served, platform.n_reward]
    ntypes = platform.I
    print("===============Perform EM algorithm===============")
    from time import time
    start = time()

    for num in range(num_em):
        q_h, l = EMALG.e(data, ntypes, nu, r)
        nu, r = EMALG.m(data, ntypes, q_h)
        ll.append(l)
    stop = time()
    print_information(nu, r)
    print("EM time: " + str(stop-start) + "s")
    plt.plot(ll)
    plt.show()



    
    
