"""
Created on Mon Dec 16 22:01:17 2024

@author: sebzi
"""

"""
Created on Thu Sep  5 12:51:41 2024

@author: sebzi

Code for the stochastic reduced inertial PDE descriptions of the Cucker-Smale 
model in one dimension. A finite difference scheme is used to discretise the 
SPDE.
"""

import numpy as np
from numba import njit #this decorator speeds up the computation


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
    
    #noise related term
    sigma_n = 1.0
    noise = sigma_n*(j - v_mean*rho)
    #
    #BC
    dj[0] += v_mean**2 *rho[-1]/(2.0*dx)
    dj[-1] += -v_mean**2 *rho[0]/(2.0*dx)  
    
    return drho, dj , noise


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

#solving the PDE at each time step
@njit(fastmath=True)
def solve(rho_I,j_I,T,dt):
    
    t = np.arange(0.0, T, dt)
    
    v_mean = np.sum(j_I)*dx #the mean velocity used to close the reduced PDE
    
    A = first_deriv(xL, xR, xL, xR, dx)
    
    for i in range(len(t)-1):


        frho , fj , noise = fI(rho_I,j_I,A,x,v_mean)

        rho_F = rho_I + frho*dt #euler scheme for time evolution of rho
        
        j_F = j_I + fj*dt + noise*np.sqrt(dt)*np.random.standard_normal(1) #euler maruyame scheme for j
        
        rho_I = rho_F
        j_I = j_F
    
    return rho_F, j_F

######

#initial particle data. In this case N=2
X0 = np.array([0.4,0.6])
V0 = np.array([0.05,0.1])


#constructing rho, j and u from the particle data
rhoI = rho(X0,x) 
jI = j(X0,V0,x)

vmean = np.sum(jI)*dx #the mean velocity used for the inertial PDE

#time interval in which to compute a solution to the PDE
T = 0.5
dt = 0.0001

#Solution of the inertial SPDE
rho, j = solve(rhoI,jI,T,dt)
