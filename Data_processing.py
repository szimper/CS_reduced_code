# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:00:09 2024

@author: sebzi
Code for processing the data generated from the codes solving the CS model,
reduced inertial model and hydrodynamic PDE.
"""

import numpy as np
import matplotlib.pyplot as plt

#spatial grid on which the PDE was solved
xL, xR , dx = -50.0 , 50.0 , 0.5
x = np.arange(xL,xR, dx)
L = 100 #length of the periodic domain

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

def L2(g):
    return np.sum(g**2)*dx

'''
Comparison of the CS model and the inertial PDE as presented in Fig. 4. for 
a vairety of different particle numbers N each for a total of M=1e2 total
realisations of the randomly generated initial conditions. The errors are 
computed in the L2 square norm.
'''

Ns = np.array([2**4,2**5,2**6,2**7, 2**8,2**9, 2**10,2**11,2**12,2**13,2**14,2**15])   

M = int(10**2)

errs = np.zeros(len(Ns),dtype=np.float64)

for k in range(len(Ns)):
    XfM = np.load('ABM_t0_m1e2_N' + str(k) + 'X.npy')
    VfM = np.load('ABM_t0_m1e2_N' + str(k) + 'V.npy')

    rhoFA , jFA = mean(XfM,VfM,M)

    rho_PDE_M = np.load("CS_PDE_N"+ str(k) + "_rho_t2_M1e2.npy")
    j_PDE_M = np.load("CS_PDE_N"+ str(k) + "_j_t2_M1e2.npy")
    
    err_d = []
    
    for i in range(M):

        v_m = np.mean(VfM[i])
        err_d.append(v_m**2*L2(rho_PDE_M[:,i] - rhoFA[:,i] ) + L2(j_PDE_M[:,i] - jFA[:,i]))
    
    errs[k] = np.mean(np.array(err_d))
    

plt.figure()
plt.rcParams.update({'text.usetex': True,'text.latex.preamble': r'\usepackage{amsfonts}'})
plt.xlabel(r"$N$")
plt.ylabel(r"$\mathbb{E} [  ||X_{PDE}(t) - X_{CS}(t)||^2_{L^2} ]$")
plt.xscale("log")
plt.yscale("log")
plt.plot(Ns,errs,linestyle='--',marker='o')
plt.plot(Ns,0.05*((Ns).astype(np.float64))**(-1),c='k', label = r"$ \propto N^{-1} $")
plt.legend()
plt.tight_layout()





'''
comparison of the inertial PDE and hydrodynamic PDE with the particles system
at some final time t. We consider P total different parameter settings (e.g. 
the velocity spread) and M total different realisations of randomly generated
initial conditions of the CS model. The PDEs are solved using initial data corresponding
to the initial conditions of the particle system.
'''

P=10 
M = 1000

XF = np.load("the x position of the CS model at some final time t for N particles")
VF = np.load("the v position of the CS model at some final time t for N particles")

rhoI = np.load("rho(x,t) for the reduced inertial PDE")
jI = np.load("j(x,t) for the reduced inertial PDE")

rhoH = np.load("rho(x,t) for the hydrodynamic PDE")
uH = np.load("u(x,t) for the hydrodynamic PDE")
jH = rhoH*uH


AI = np.zeros((P,M))
AH = np.zeros((P,M))

I_err = np.zeros((P))
H_err = np.zeros((P))

for i in range(P):
    for k in range(M):
        
        rhoF_CS = rho(XF[i][k],x)
        jF_CS = j(XF[i][k], VF[i][k], x)
        
        vms = np.mean(VF[i][k])
        
        AI[i][k] = ( L2( rhoF_CS - rhoI[i][k] )*vms**2 + L2( jF_CS - jI[i][k] )   )
        AH[i][k] = ( L2( rhoF_CS - rhoH[i][k] )*vms**2 + L2( jF_CS - jH[i][k] )     )
        

    I_err[i] = np.mean(AI)
    H_err[i] = np.mean(AH)

###########################
'''
Plotting the difference between the reduced inertial PDE and the hydrodynamic
PDE as a function of time for P different parameter settings.
'''


T = 1.0
dt = 0.0001
t1 =np.arange(0,T,dt) #time interval on which the PDEs have been computed


rho_diff_t = []
j_diff_t = []
L2_diffs = []

I_rhos = np.load("rho of the inertial PDE over the time interval t1 and spatial interval x")
I_js = np.load("j corresponding to each entry of I_rhos")

H_rhos = np.load("rho of the hydrodynamic PDE corresponding to the same initial data as I_rhos")
H_us = np.load("u corresponding to each entry of U_rhos")

H_js = H_rhos*H_us


cmap = plt.colormaps['plasma']
colors = cmap(np.linspace(0, 1, P))

fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [20,1]})

for i in range(P):
    
    for k in range(len(t1)):
        rho_diff_t.append(L2(I_rhos[i,k] - H_rhos[i,k]  ))
        j_diff_t.append(L2(I_js[i,k] - H_js[i,k]) )  
    
    vmean = np.sum(I_js[i])*dx
    
    ax[0].plot(t1[1:], np.array(rho_diff_t[1:])*vmean**2 + j_diff_t[1:],  c = colors[i])
    L2_diffs.append(np.array(rho_diff_t)*vmean**2 + j_diff_t)

fraction =  4
import matplotlib as mpl

norm = mpl.colors.Normalize(vmin=-3, vmax=99)
cbar = ax[1].figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap='plasma'),
            ax=ax[1], pad=-1.0, aspect=50, 
            label=r" Parameter",ticks=[], fraction=fraction)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].axis('off')