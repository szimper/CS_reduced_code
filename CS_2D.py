# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:38:33 2024

@author: sebzi
"""

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


#randomly generated initial conditions
@njit(fastmath=True)
def ABM_ICs(N,XL,XU,VL,VU):
 
    #generating the ICs of the individuals
    X_pos = np.random.uniform(low=XL, high=XU,size = (N,2) )

    V_pos = np.random.uniform(low=VL, high=VU,size = (N,2) )
         
    return X_pos, V_pos


#interaction potential
@njit(fastmath=True)
def phi(u):
    r = 0.5
    return 1.0/(1.0+u)**r 


#@njit(fastmath=True)
def F(t,y):

    y2 = np.reshape(y,(2,1000,2))
    X, V = y2
    
    N = len(X) 
    
    FX = np.zeros((N,2),dtype=np.float64)
    FV = np.zeros((N,2),dtype=np.float64)
    
    FX = V
    
    gamma = 50.0 #4.0
    
    for i in range(N):
               
        diff_X = X-X[i,:]
        

        
        diff_V = V-V[i,:]
        
        u = np.linalg.norm(diff_X,axis=1)**2 #axis arguament is not supported by numba
        
        
        FV[i,:] = gamma*np.sum( phi(u)[:,None]*diff_V,axis=0 )/N      
    
    resh = np.array([ FX, FV])
    
    return  np.reshape(resh,-1)



N = 1000

Xi, Vi = ABM_ICs(N,-20.0,20.0,-5.0,5.0)
t = np.arange(0,2.0,0.01)

y0 = np.array([Xi,Vi])

sol = solve_ivp(F,(0.0,2.0),np.reshape(y0,-1),t_eval=t)

solf = np.reshape(sol.y[:,-1],(2,1000,2))

solX = solf[0]
solV = solf[1]

