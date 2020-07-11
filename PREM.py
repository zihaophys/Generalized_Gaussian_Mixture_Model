"""EM with Pruning Initialization"""

# Author: Zihao Wang, HKUST <zihaophys@gmail.com> 
# Date: July 11 2020

import numpy as np
import matplotlib.pyplot as plt 
import random
from GGMM import *

class TREM:
    @staticmethod
    def firstRound(data, ntypes, nu_min):
        # prepare data
        n_served, n_reward = data
        n_served = np.array(n_served)
        n_reward = np.array(n_reward)
        H, J  = n_served.shape
        I = ntypes

        # initialization
        ell = int(1.0/nu_min*np.log10(I))
        nu = np.ones(ell)/ell
        tmp = random.sample(range(0, H), ell)
        r = np.array([n_reward[i]/n_served[i] for i in tmp])

        # first round EM
        # 10-step is enough
        for i in range(10):
            q_h, l = EMALG.e(data, ell, nu, r)
            nu, r = EMALG.m(data, ell, q_h)
        # print_vec(nu)

        # pruning
        # first pruning
        nu_T = 1/2/ell + 2/H 
        # print(nu_T)
        # cluster starvation
        r_pruned_1 = np.array([r[i] for i in range(ell) if nu[i] >= nu_T])
        # second pruning
        # clustering, generating distance matrix
        X2 = np.c_[np.sum(r_pruned_1**2,1)]
        D = X2 + X2.T - 2*r_pruned_1.dot(r_pruned_1.T)
        # perform farthest-first-traversal
        visited=[]
        i=np.int32(np.random.uniform(len(r_pruned_1)))
        visited.append(i)
        while len(visited) < I:
            dist=np.min([D[i] for i in visited],0)
            for i in np.argsort(dist)[::-1]:
                if i not in visited:
                    visited.append(i)
                    break

        r_pruned_2 = np.array([r_pruned_1[i] for i in visited])
        return r_pruned_2

    @staticmethod
    def secondRound(data, ntypes, r, tol=1e-8, num=200):
        nu = np.ones(ntypes)/ntypes

        q_h, ltmp = EMALG.e(data, ntypes, nu, r)
        nu, r = EMALG.m(data, ntypes, q_h)
        for i in range(num):
            q_h, l = EMALG.e(data, ntypes, nu, r)
            nu, r = EMALG.m(data, ntypes, q_h)
            if (l - ltmp < tol):
                return nu, r
            ltmp = l
        return nu, r


    
    
if __name__ == "__main__":
    I = 3
    J = 4
    H = 100
    n_j = 50
    num_em = 20

    platform = Platform(I, J, H, n_j)
    data = [platform.n_served, platform.n_reward]
    ntypes = platform.I
    print("===============Perform EM algorithm===============")
    from time import time
    start = time()

    r = TREM.firstRound(data, ntypes, 0.02)
    nu, r = TREM.secondRound(data, ntypes, r)
        
    stop = time()
    print_information(nu, r)
    print("EM time: " + str(stop-start) + "s")
