import numpy as np
import time as time
import scipy
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import scipy.io as sio
from numpy import genfromtxt
import math

NX = 1024
NY = 128

nu = 2e-2
n = int(NX/NY)
s = 20
gamma = 2

dt = 1e-2*s

reg = 18

Lx    = 100
dy    = Lx/NY
dx    = Lx/NX

x_bar     = np.linspace(0, Lx-dy, num=NY+1)
x_bar = (x_bar[1:]+x_bar[:NY])/2

x_prime = np.linspace(0, Lx-dx, num=NX)

kx_gamma = (2*math.pi/Lx)*np.concatenate((np.arange(0,(NY/gamma)/2+1,dtype=float),np.arange((-(NY/gamma)/2+1),0,dtype=float)))
kx_bar    = (2*math.pi/Lx)*np.concatenate((np.arange(0,NY/2+1,dtype=float),np.arange((-NY/2+1),0,dtype=float)))
kx_prime    = (2*math.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=float),np.arange((-NX/2+1),0,dtype=float)))


D1_gamma = 1j*kx_gamma.reshape([int(NY/gamma),1])

D1_bar = 1j*kx_bar
D2_bar = kx_bar*kx_bar
D1_bar = D1_bar.reshape([NY,1])
D2_bar = D2_bar.reshape([NY,1])

I = np.eye(NY)
D2x = 1 + 0.5*dt*nu*D2_bar

D1_prime = 1j*kx_prime
D1_prime = D1_prime.reshape([NX,1])


def filter_bar(u,n):
  u_bar = np.zeros((int(u.shape[0]/n),u.shape[1]))

  for i in range(int(u.shape[0]/n)):
    u_bar[i] = np.mean(u[n*i:(n*i+n)])

  return u_bar

def calc_cp(u, gamma):

  delta = dy
  delta_sub = dy*gamma

  uu_sub = filter_bar(u*u, gamma)
  u_sub = filter_bar(u, gamma)

  L = .5*(uu_sub - u_sub*u_sub).squeeze()

  der_u = np.real(ifft(D1_bar*fft(u,axis = 0),axis=0))

  der_u_sub = np.real(ifft(D1_gamma*fft(u_sub,axis = 0),axis=0))

  M = ((delta**2)*filter_bar(np.linalg.norm(der_u,2)*der_u,gamma)-(delta_sub**2)*np.linalg.norm(der_u_sub,2)*der_u_sub).squeeze()

  c_p = np.dot(L,M)/np.dot(M,M)

  if c_p < 0:
    c_p = abs(c_p)

  return c_p


u_bar_dict = sio.loadmat('./dealiasing/u_bar_region_' + str(reg) + '.mat')
u_bar_store=u_bar_dict['u_bar'].transpose()

force_dict = sio.loadmat('./dealiasing/f_bar_all_regions.mat')
force_bar=force_dict['f_bar'][:,int((reg-1)*12500)+int(1000000/s):]


num_pred = 200000

u_old = u_bar_store[1000000-s,:].reshape([NY,1])
u = u_bar_store[1000000,:].reshape([NY,1])


u_fft = fft(u,axis=0)
u_old_fft = fft(u_old,axis=0)

u_store = np.zeros((NY,num_pred))
sub_store = np.zeros((NY,num_pred))

# uses a fixed c_p to start and then calculates at each time step

c_p = .038
S_bar = .5*D1_bar*u_old_fft
S_bar_norm = np.sqrt(2*np.dot(S_bar.transpose(),S_bar))
tau_approx_old = -D1_bar*c_p*2*(dy**2)*S_bar_norm*S_bar

for i in range(num_pred):

  force=force_bar[:,i].reshape((NY,1))

  F = D1_bar*fft(.5*(u**2),axis=0)
  F0 = D1_bar*fft(.5*(u_old**2),axis=0)

  c_p = calc_cp(u,gamma)

  S_bar = .5*D1_bar*u_fft
  S_bar_norm = np.sqrt(2*np.dot(S_bar.transpose(),S_bar))
  tau_approx = -D1_bar*c_p*2*(dy**2)*S_bar_norm*S_bar

  uRHS = -0.5*dt*(3*F- F0) - 0.5*dt*nu*(D2_bar*u_fft)  + u_fft + dt*fft(force,axis=0)\
            -1/2*dt*(3*tau_approx-tau_approx_old)

  tau_approx_old = tau_approx

  u_old = u

  u_fft = uRHS/D2x.reshape([NY,1])

  u = np.real(ifft(u_fft,axis=0))
  u_store[:,i] = u.squeeze()
  sub_store[:,i] = np.real(ifft(tau_approx,axis=0)).squeeze()


sio.savemat('./DSMAG_region_.mat',
           {'u_pred':u_store, 'sub_pred':sub_store})
