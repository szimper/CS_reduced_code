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
    gamma = 50.0
    return gamma/(1.0+u**2)**r

L = 100 #length of the periodic domain

@njit(fastmath=True)
def F(X,V):
    N = len(X) 
    
    FX = np.zeros(N,dtype=np.float64)
    FV = np.zeros(N,dtype=np.float64)
    
    FX = V
    
    for i in range(N):
               
        diff_X = X[:]-X[i]
        
        diff_V = V[:]-V[i]
        
        u = np.minimum( np.abs(diff_X), L-np.abs(diff_X))
        
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
    
    Xf = (Xi + FX*dt) % L
     
    return Xf, Vf

@njit(fastmath=True)
def run(N,xL,xU,vL,vU,t,dt):
    
    Xi, Vi = ABM_ICs(N, xL,xU,vL,vU)

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


#for a single realisation of the ABM
N = 100 
T = 2.0
dt = 0.01
xL,xU,vL,vU = -50,50,-20,20

solX, solO = run(N,xL,xU,vL,vU,T,dt)


#parallisation for numerous realisations over the random realisations for a variety of parameter settings

MC = 1e2 #number of realisations
M_t = np.linspace(0, 1, int(MC))

@njit(fastmath=True)
def M_eps(M,N): #function to run over M realisations and different parameter settings in
                #this case the number of agents N. For Fig 5.a) of the manuscript change 
                #this to oU and set oL = - oU (thereby increasing or decreasing the initial
                #velocity spread)
                
    distXI = np.zeros((len(M),N),dtype=np.float64) 
    distVI = np.zeros((len(M),N),dtype=np.float64)    
    distXF = np.zeros((len(M),N),dtype=np.float64)    
    distVF = np.zeros((len(M),N),dtype=np.float64)
    
    for j in range(len(M)):
            solX, solV = run(N,xL,xU,vL,vU,T,dt)
            
            distXI[j,:] = solX[0] #distXI[j,i,:] = solX[0]
            distVI[j,:] = solV[0]
            distXF[j,:] = solX[-1]
            distVF[j,:] = solV[-1]     
            
    
    return distXI, distVI, distXF, distVF 


import multiprocessing as mp #package to implement the paralisation
from functools import partial

def main_V():
    Ns = np.array([2**4,2**5,2**6,2**7, 2**8,2**9, 2**10,2**11,2**12,2**13,2**14,2**15 ])  

    pool = mp.Pool(mp.cpu_count())
    x_split= np.array_split(M_t,  mp.cpu_count())    
    for k in Ns:
        
        fix = partial(M_eps,N=k) 
        
        result = pool.map(fix, x_split)
        
        
        Xi = result[0][0]
        Vi = result[0][1]
        Xf = result[0][2]
        Vf = result[0][3]    
    
        for i in range(len(result)-1):
            #print(i)
            Xi = np.concatenate((Xi,np.array(result[i+1][0])))  
            Vi = np.concatenate((Vi,np.array(result[i+1][1])))  
            Xf = np.concatenate((Xf,np.array(result[i+1][2])))  
            Vf = np.concatenate((Vf,np.array(result[i+1][3])))    
        
        #saving initial and final data in an appropriately named .npy
        #file to analyse later
        np.save('ABM_t0_M1e2_N' + str(k) + 'X.npy',Xi)
        np.save('ABM_t0_M1e2_N' + str(k) + 'V.npy',Vi)
        np.save('ABM_t2_M1e2_N' + str(k) + 'X.npy',Xf)
        np.save('ABM_t2_M1e2_N' + str(k) + 'V.npy',Vf)
    

    
if __name__ == "__main__":
  main_V()



