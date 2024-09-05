# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:26:59 2024

@author: sebzi

Code for the CS model in one dimension. By making sigma positive
one can also compute the stochastic CS model.
"""

import numpy as np
from numba import njit


#randomly generated initial conditions
def ABM_ICs(N,XL,XU,VL,VU):
    
    X_pos = np.zeros(N,dtype=np.float64)
    
    V_pos =  np.zeros(N,dtype=np.float64)
    
    for i in range(N):
        X_pos[i] = np.random.uniform(XL,XU) 
        V_pos[i] =  np.random.uniform(VL,VU)
    return X_pos, V_pos


#interaction potential
@njit(fastmath=True)
def phi(u):
    r = 0.5
    return 1.0/(1.0+u**2)**r


@njit(fastmath=True)
def F(X,V):
    N = len(X) 
    
    FX = np.zeros(N,dtype=np.float64)
    FV = np.zeros(N,dtype=np.float64)
    
    FX = V
    
    for i in range(N):
               
        diff_X = X[:]-X[i]
        
        diff_V = V[:]-V[i]
        
        u = np.minimum( np.abs(diff_X), 1-np.abs(diff_X))
        
        FV[i] = np.sum( phi(u)*diff_V )/N      
    
    return FX, FV

#make sigma positive for the stochastic CS model
sigma = 0.0

@njit(fastmath=True)
def updateABM(Xi,Vi,dt,v_m):
    N = len(Xi) 
    FX, FV = F(Xi,Vi)
    
    Xf = np.zeros(N,dtype=np.float64)
    Vf = np.zeros(N,dtype=np.float64)
    #Euler Maruyama scheme for individuals  
    Vf = Vi + FV*dt + sigma*(Vi-v_m)*np.sqrt(dt)*np.random.standard_normal()
    
    Xf = (Xi + FX*dt) % 1.0
     
    return Xf, Vf

@njit(fastmath=True)
def run(N,xL,xU,oL,oU,t,dt):
    
    Xi, Vi = ABM_ICs(N, xL,xU,oL,oU)

    time = np.arange(0,t,dt)
    
    X_t = np.zeros((len(time)+1, N),dtype=np.float64) #arrays for storing information at each time step
    V_t = np.zeros((len(time)+1, N),dtype=np.float64)
    
    #print(np.min(Xi))
    #print(np.max(Xi))
    
    X_t[0]= Xi
    V_t[0] = Vi

    for count in range(len(time)):
        
        Xi, Vi = updateABM(Xi,Vi,dt)      
           
        X_t[count+1] = Xi
        V_t[count+1] = Vi

    return X_t, V_t


N = 100 
T = 5.0
dt = 0.01

solX, solO = run(N,0.0,1.0,0.0,1.0,5.0,0.01)



