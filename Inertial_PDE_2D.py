# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:43:48 2024

@author: sebzi
Code for solving the reduced inertial PDE in two dimensions
"""

from scipy.stats import multivariate_normal
import numpy as np
from numba import njit
from scipy import signal
from scipy.integrate import odeint


xL, xR, yB, yT, dx = -50.0, 50.0, -50.0, 50.0, 0.5  # 0.5
x_g, y_g = np.mgrid[xL:xR+dx/2:dx, yB:yT+dx/2:dx]


@njit(fastmath=True)
def first_deriv(xL, xR, yB, yT, dx):  

    Nx = int((xR-xL)/dx+1)

    B = np.diag(-1*np.ones(Nx-1, dtype=np.float64), -1) + \
        np.diag(np.ones(Nx-1, dtype=np.float64), 1)

    B = B/(2.0*dx)

    return B


gamma = 50.0  # 4.0

@njit(fastmath=True)
def phi(u1, u2):

    r = 0.5

    return gamma*1.0/(1.0+u1**2 + u2**2)**r  # C_array



def f(y0,t, v_m, B, a):

    rho_v ,j1_v,j2_v = y0

    rho = np.reshape(rho_v, (int( np.sqrt(len(rho_v)) ), int( np.sqrt(len(rho_v)) ) ))
    j1 = np.reshape(j1_v, (int( np.sqrt(len(j1_v)) ), int( np.sqrt(len(j1_v)) ) ))
    j2 = np.reshape(j2_v, (int( np.sqrt(len(j2_v)) ), int( np.sqrt(len(j2_v)) ) ))

    drho = - B @ (j1) - (j2) @ np.transpose(B)

    drho[0, :] += (j1[-1, :])/(2.0*dx)  # =0.0
    drho[-1, :] += -(j1[0, :])/(2.0*dx)
    drho[:, 0] += (j2[:, -1])/(2.0*dx)
    drho[:, -1] += -(j2[:, 0])/(2.0*dx)


    a_conv_j1 = signal.convolve(a,j1,mode='same')*dx**2.0
    a_conv_j2 = signal.convolve(a,j2,mode='same')*dx**2.0

    a_conv_rho = signal.convolve(a,rho,mode='same')*dx**2.0


    dj1 = rho * a_conv_j1 - j1 * a_conv_rho - \
        v_m[0]*(B @ (j1) + (j2) @ np.transpose(B))
    dj2 = rho * a_conv_j2 - j2 * a_conv_rho - \
        v_m[1]*(B @ (j1) + (j2) @ np.transpose(B))

    dj1[0, :] += -v_m[0]*(-j1[-1, :])/(2.0*dx)  
    dj1[-1, :] += -v_m[0]*(j1[0, :])/(2.0*dx)
    dj1[:, 0] += -v_m[0]*(-j2[:, -1])/(2.0*dx)
    dj1[:, -1] += -v_m[0]*(j2[:, 0])/(2.0*dx)

    dj2[0, :] += -v_m[1]*(-j1[-1, :])/(2.0*dx)
    dj2[-1, :] += -v_m[1]*(j1[0, :])/(2.0*dx)
    dj2[:, 0] += -v_m[1]*(-j2[:, -1])/(2.0*dx)
    dj2[:, -1] += -v_m[1]*(j2[:, 0])/(2.0*dx)

    drho_v = np.reshape(drho,len(rho_v))
    dj1_v = np.reshape(dj1,len(rho_v))
    dj2_v = np.reshape(dj2,len(rho_v))

    return  drho_v , dj1_v, dj2_v




def gaussian_2D(x, y, mean, epsilon):
   # x, y = np.mgrid[xL:xR:dx, yB:yT:dx]
    pos = np.dstack((x, y))

    rv = multivariate_normal(mean, [[epsilon, 0.0], [0.0, epsilon]])

    z = rv.pdf(pos)

    return z


def rho(X, x, y):
    N = len(X)

    epsilon = 1.0

    rho_d = np.zeros(np.shape(x), dtype=np.float64)

    for i in X:
        rho_d += gaussian_2D(x, y, i, epsilon)/N
    return rho_d


def j(X, V, x, y):
    N = len(X)

    epsilon = 1.0

    j_d1 = np.zeros(np.shape(x), dtype=np.float64)
    j_d2 = np.zeros(np.shape(x), dtype=np.float64)

    for i in range(len(X)):

        density = gaussian_2D(x, y, X[i], epsilon)/N
        j_d1 += V[i, 0]*density

        j_d2 += V[i, 1]*density

    return j_d1, j_d2





dt = 0.001  # 0.001
T = 2.0
t = np.arange(0, T+dt/2, dt)  # 0.01


Xi = np.array([[0.0,0.0],[0.5,0.5]])
Vi = np.array([[0.1,0.0],[0.0,1.0]])

a0_rho = rho(Xi, x_g, y_g)
a0_j1, a0_j2 = j(Xi, Vi, x_g, y_g)

v_mean = np.mean(Vi, axis=0)

I0 = np.array([a0_rho, a0_j1, a0_j2])

B = first_deriv(xL, xR, yB, yT, dx)

x = np.arange(xL, xR+dx/2, dx)
y = np.arange(yB, yT+dx/2, dx)

a = np.zeros(np.shape(a0_rho))

for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] = phi(x[i], y[j])

sol = odeint(f, I0, t,args = (v_mean,B,a))




