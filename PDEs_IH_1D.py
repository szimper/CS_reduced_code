# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:51:41 2024

@author: sebzi

Code for the reduced inertial and hydrodynamic descriptions of the Cucker-Smale 
model in one dimension. A finite difference scheme is used to discretise the 
spatial component and the odeint package to solve the resulting system of ODEs 
to compute the PDEs.
"""

import numpy as np
from numba import njit #this decorator speeds up the computation
from scipy.integrate import odeint


#spatial interval over which to compute the solution
xL, xR , dx = -50.0 , 50.0 , 0.5
x = np.arange(xL,xR, dx)
L = 100 #length of the periodic domain

@njit(fastmath=True)
def first_deriv(xL,xR,yB,yT,dx):
    
    Nx= int((xR-xL)/dx )
    
    B = np.diag(-1*np.ones(Nx-1,dtype=np.float64),-1) + np.diag(np.ones(Nx-1,dtype=np.float64),1)
       
    B = B/(2.0*dx)

    return B

#interaction potential of the system
@njit(fastmath=True)
def phi(ud):
    r = 0.5
    gamma = 1.0 
    u_periodic = np.minimum( ud, L-ud)
    return gamma*1.0/((1.0+u_periodic**2)**r) 


@njit(fastmath=True)
def convolution(x1,h):
    y = np.arange(xL,xR,dx)

    return dx*np.sum( phi(np.abs(x1-y))*h )


@njit(fastmath=True)
def int_tot(x1,rho,u,index):
    y = np.arange(xL,xR,dx)
    return dx*np.sum( (phi(np.abs(x1-y))*rho*(u - u[index])) )    


#function for the weight
@njit(fastmath=True)
def weight(w0,tn):
    C = 25.0
    return 1.0 - np.exp(-C*tn) + np.exp(-C*tn)*w0


@njit(fastmath=True)
def fI(z0,t,A,v_mean,w0):  #function for the reduced inertial PDE
    rho = z0[:int(len(z0)/2)]
    j = z0[int(len(z0)/2):]

    drho = - (A @ j)
    
    #BC
    drho[0] = - (j[1] - j[-1])/(2.0*dx)
    drho[-1] = - (j[0] - j[-2])/(2.0*dx)    
    
    conv_aj = np.zeros(len(rho))
    conv_arho = np.zeros(len(rho))
    
    for i in range(len(x)):
        conv_aj[i] = convolution(x[i],j)
        conv_arho[i] = convolution(x[i],rho)
    
    dj = rho*conv_aj - j*conv_arho - v_mean**2 * A @ rho 

    ###uncomment the following if you want to use the modified PDE with weight term
    #w = weight(w0,t)
    #dj = rho*conv_aj - j*conv_arho - v_mean**2 * A @ (w*rho)  
    ###

    #BC
    dj[0] += v_mean**2 *rho[-1]/(2.0*dx)
    dj[-1] += -v_mean**2 *rho[0]/(2.0*dx)  
    
    return np.concatenate((drho, dj))


@njit(fastmath=True)
def fH(z0,t,A,a_c): #function for the hydrodynamic PDE

    rho = z0[:int(len(z0)/2)]
    u = z0[int(len(z0)/2):]
    
    rho_u = rho*u
    
    drho = - (A @ (rho_u))
    
    #BC
    drho[0] = -(rho_u[1] - rho_u[-1])/(2.0*dx)  
    drho[-1] = -(rho_u[0] - rho_u[-2])/(2.0*dx)
    ######

    whole_int = np.zeros(len(rho))
    for i in range(len(x)):

        whole_int[i] = int_tot(x[i],rho,u,i)

    du = whole_int - u * (A @ u) 
    #BC
    du[0] +=  u[0]*(u[-1])/(2.0*dx)
    du[-1] += - u[-1]*(u[0])/(2.0*dx)
    #
    return np.concatenate((drho, du))


######
#The functions below are for computing initial data rho, j and u from particle data
def vonmises(x,mu,kappa):
    x2 = 2*np.pi*x/L
    y = np.exp(kappa*np.cos(x2-2*np.pi*mu))
    
    return y/(np.sum(y)*dx )


def rho(x_pos,x):
    N = len(x_pos)

    epsilon = 1.0 
    
    rho_d = 0
    for i in x_pos:
        rho_d += vonmises(x,i,epsilon)/N
    return rho_d


def j(x_pos,v,x):
    N = len(x_pos)
    epsilon = 1.0 
    
    j_d = 0
    for i in range(len(x_pos)):
        j_d += v[i]*vonmises(x,x_pos[i],epsilon)/N
        
    
    return j_d


#function for calculating the initial weight in case the modified inertial PDE is used
def w_I(rho_I,x_pos,v):
    N = len(x_pos)
    epsilon = 1.0 
    v_m = np.mean(v)
    
    w_d = np.zeros(len(x),dtype=np.float64) 
    for i in range(len(x_pos)):
        w_d += v[i]**2.0 * vonmises(x,x_pos[i],epsilon)
        
    
    return w_d/(N*v_m**2.0 * rho_I)

######

#initial particle data. In this case N=2
X0 = np.array([0.4,0.6])
V0 = np.array([0.05,0.1])


#constructing rho, j and u from the particle data
rhoI = rho(X0,x) 
jI = j(X0,V0,x)
uI = jI/rhoI

vmean = np.sum(jI)*dx #the mean velocity used for the inertial PDE

w0 = w_I(rhoI,X0,V0) #initial weight for the modified inertial PDE


#time interval in which to compute a solution to the PDE
T = 1.0
dt = 0.0001
t = np.arange(0, T+dt/2, dt)

A = first_deriv(xL, xR, xL, xR, dx)

a_c = np.zeros(np.shape(rhoI))
 
for i in range(len(a_c)):
    a_c[i] = phi(x[i])


#Solution of the inertial PDE
        
I0 = np.concatenate((rhoI,jI))

sol = odeint(fI, I0, t,args=(A,vmean,w0))

rho = sol[:,:int(len(I0)/2)]
j = sol[:,int(len(I0)/2):]


#Solution of the hydrodynamic PDE
I0H = np.concatenate((rhoI,uI))

solH = odeint(fH, I0H, t,args=(A,a_c))

rhoH = solH[:,:int(len(I0)/2)]
u = solH[:,int(len(I0)/2):]

'''
for Figs. 4 and 5 where a large number of realisations (M) of the CS model and
inertial reduced PDE are compared, the initial CS data is loaded to initialise
the PDE which is then simulated for each realisation of the CS model. This is done
for different parameter values as well: in Fig. 4 for different N; Fig. 5a) 
different initial velocity spread; while in Fig 5b) the epsilon when initilasing 
the PDE is changed in the PDE simulations.
'''

@njit(fastmath=True) #function for constructing the PDE initial data
def mean(XM,VM,M_c):

    rho_F = np.zeros( (len(x),M_c) , dtype=np.float64)
    j_F = np.zeros( (len(x),M_c) , dtype=np.float64)    
    
    for i in range(M_c):
        
        Xf = XM[i,:]
        Vf = VM[i,:]
        
        rho_s = rho(Xf,x)
        j_s = j(Xf,Vf,x)
        
        rho_F[:,i] = rho_s
        j_F[:,i] = j_s
    
    return rho_F, j_F 


Ns = np.array([2**4,2**5,2**6,2**7, 2**8,2**9, 2**10,2**11,2**12,2**13,2**14,2**15 ])  

def PDE_multi(M):
    
    for k in Ns:
        #loading the initial data of the CS model for different values of N
        XiM = np.load('ABM_t0_m1e2_N' + str(k) + 'X.npy')
        ViM = np.load('ABM_t0_m1e2_N' + str(k) + 'V.npy')
        
        
        rho_PDE_M = np.zeros( (len(x),M) , dtype=np.float64)
        j_PDE_M = np.zeros( (len(x),M) , dtype=np.float64)    
        
    
        rhoI , jI = mean(XiM,ViM,M)
        
        for i in range(M):
            vmean = np.mean(ViM[i])
            
            I0 = np.concatenate((rhoI[:,i],jI[:,i]))
            
            w0 = w_I(rhoI[:,i],XiM[i,:],ViM[i,:])  
            
            sol = odeint(fI, I0, t,args=(A,vmean,w0))
            
            
            rho_PDE_M[:,i] = sol[-1,:int(len(I0)/2)]
            j_PDE_M[:,i] = sol[-1,int(len(I0)/2):]
        
        #saving the PDE data at the final time
        np.save("CS_PDE_N"+ str(k) + "_rho_t2_M1e2.npy",rho_PDE_M)
        np.save("CS_PDE_N"+ str(k) + "_j_t2_M1e2.npy",j_PDE_M)
        
    return rho_PDE_M, j_PDE_M


a, b = PDE_multi(int(1e2))