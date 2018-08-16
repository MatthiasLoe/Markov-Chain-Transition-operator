#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:10:17 2018

@author: Matthias1


This program generates observations from the CIR or the radial Ornstein-Uhlenbeck using the Euler-Maruyama scheme
"""

#diffusiongen

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scipy.stats import norm
import numpy as np



def coxingersoll(x0,M,dt,theta,sigma2,mu, out=None):
    #generates samples from the Cox-Ingersoll-Ross (CIR) process using the Euler-Maruyama scheme
    #Inputs: real valued x0, dt, theta, sigma2, mu, integer valued M
    #x0: starting value, dt=time-steps in the approximation, theta, sigma2, mu are parameters of the CIR process
    #M denotes the total number of simulated steps 
    ev=np.zeros(M+1)
    r=norm.rvs(size=M, scale=1) #vector of standard normals
    ev[0]=x0 #starting value
    for i in range(1,len(ev)):
        b=-theta*(ev[i-1]-mu)*dt #drift
        sd=np.sqrt(abs(sigma2*ev[i-1]*dt)) #volatility
        ev[i]=ev[i-1]+b+sd*r[i-1] #add increment
    return ev
    

    
    
def radialorn(x0,M,dt,theta,sigma2,out=None):
    #generates samples from the Radial Ornstein-Uhlenbeck process using the Euler-Maruyama scheme
    #Inputs: real valued x0, dt, theta, sigma2, mu, integer valued M
    #x0: starting value, dt=time-steps in the approximation, theta, sigma2 are parameters of the Radial Ornstein-Uhlenbeck process
    #M denotes the total number of simulated steps 
    ev=np.zeros(M+1)
    r=norm.rvs(size=M, scale=1)
    ev[0]=x0 #start value
    for i in range(1,len(ev)):
        b=(theta*(1/ev[i-1])-ev[i-1])*dt #drift
        sd=np.sqrt(sigma2*dt) #volatility
        ev[i]=ev[i-1]+b+sd*r[i-1] #add increment
    return ev


def diffgen(Delta,n,theta,sigma2,mu):
     # This program generates a .txt file containing a column of observations of the CIR process
    #Inputs: real-valued Delta, theta, sigma2, mu, integer n
    #Delta is the desired step size of the chain to be generated, n its length
    #theta, sigma2, mu are parameters of the CIR process
    dt = 0.005 #step size for the Euler-Maruyama scheme, should be much smaller than Delta
    M=int(n*Delta/dt) #total number of samples to be generated, much larger than n
 

    x0 = 1 #starting value
    v1=coxingersoll(x0,M,dt,theta, sigma2,mu) #generate samples via the Euler-Maruyama scheme
    markers=[int(i*Delta/dt) for i in range(0,n+1)] #keep only every Delta/dt'th-sample to have samples at time distance Delta
#    markers_on = np.linspace(x0,M,T/Delta)
#    markers_on=[int(markers_on[i-1]) for i in range(1,len(markers_on))]
#    markers_on=markers_on.tolist()

    vamend=[v1[i] for i in markers]
    with open("Diffusiondata.txt", "w") as f: #writes data into .txt file as a column
        for s in vamend:
            f.write(str(s) +"\n")