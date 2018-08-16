#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:21:26 2018

@author: Matthias1
"""

import numpy as np
import scipy.special as spec



def tbp(i,x):
    #computes trigonometric basis functions on [-b,b]
    b=2
    c=2
    if i==0:
        tb=1/np.sqrt(2*b)
    else:
        if i%2==0:
            tb=np.sqrt(1/b)*np.cos(np.pi*(x-c)*i/(2*b))   
        else:
            tb=np.sqrt(1/b)*np.sin(np.pi*(x-c)*(i+1)/(2*b))
    return tb

def pinfty(d,J):
    # computes the matrix R, i.e. the action of the transition operator on the chosen basis
    #inputs: integer J, the resolution level, vector d containing the real valued observations of the processs
    # integer valued J denotes the number of basis coefficients minus 1
    # d is a datavector containing n+1 real valued entries
    n=len(d)-1 #number of increments
    pinfty=np.zeros((J+1,J+1)) #J+1 times J+1 matrix of zeroes
    for i in range(0,J+1):
        for j in range(0,J+1):
            if pinfty[i,j] ==0:
                for k in range(0,n):
                    pinfty[i,j]+=tbp(i,d[k])*tbp(j,d[k+1])+tbp(j,d[k])*tbp(i,d[k+1]) #compute the action of P in the (i,j)-th cel
                
                pinfty[i,j]=pinfty[i,j]/(2*n) #normalize
                pinfty[j,i]=pinfty[i,j]
    return(pinfty)



def pitreh(p,l):
    #hard thresholds eigenvalues of p at level l
    #inputs: matrix p, real valued l
    u,s,v=np.linalg.svd(p,full_matrices=True) #computes the svd
    for i in range(0,len(s)): #hard threshold at level l
        if s[i]-l <= 0: 
            s[i]=0
    a=np.dot(u*s,v) #return thresholded matrix
    return(a)
    

def ghati(d,J):
    # computes the inverse Gram matrix hat G^-1 at resolution level J
    #inputs: data vector d, resolution level J
    n=len(d)-1 
    ghat=np.zeros((J+1,J+1)) #ghat is the gram matrix given by the symmetrized 1/n sum e_j(x_i)e_i(x_i) with boundary correction
    for i in range(0,J+1):
        for j in range(0,J+1):
            if ghat[i,j] == 0:
                for k in range(0,n):
                    ghat[i,j]+=tbp(i,d[k])*tbp(j,d[k])+tbp(i,d[k+1])*tbp(j,d[k+1])
                ghat[i,j]=ghat[i,j]/(2*n)
                ghat[j,i]=ghat[i,j]
    ghati=np.linalg.inv(ghat)
    return(ghati)
    

    
def estimatoroph(d,J,l):
    #computes the matrix of coefficients (\hat G^1 \tilde R_J) for a basis of resolution level J for a given data vector with hard threshold level l 
    #inputs: data vector d, integer valued resolution level J, real valued hard threshold level l
    p=pinfty(d,J)
    pt=pitreh(p,l)
    gi=ghati(d,J)
    f=np.matmul(gi,pt)
    return(f)
    
def estimatortd(p,x,y):
    #given a matrix of basis coefficients (i.e. \textbf{ \tilde P})_J in the paper) computes the corresponding estimator \tilde p(x,y))
    #Inputs: Matrix P of trigoometric basis coefficients of the transition density, real valued x and y values
    J=len(p) #checks the resolution level
    g=np.zeros((J,J)) #for each basis coefficient e_i*e_j compute the corresponding (x,y) value
    for i in range(0,J):
        for j in range(0,J):
            g[i,j]+=tbp(i,x)*tbp(j,y)
    fe=np.sum(np.multiply(p,g)) #weight by the coefficients of p
    return(fe)
    
  
def ornstein(x,y,Delta,theta,sigma2):
    #transition density p(x,y) of an Ornstein-Uhlenbeck Process
    #inputs real-valued x,y, Delta, theta and sigma2, sigma2 denotes the variance of the brownian motion, theta the drift, Delta the step size
    mu=np.exp(-theta*Delta)*x #mean
    var=(sigma2/(2*theta))*(1-np.exp(-2*theta*Delta)) #variance 
    ohrn=(np.exp((-(y-mu)**2)/(2*var)))/(np.sqrt(2*np.pi*var))  #final density
    return ohrn

def cir(x,y,Delta,theta,sigma2,mu):
    #transition density p(x,y) of a Cox-Ingersoll-Ross (CIR) process
    #Inputs: Real valued x,y,Delta,theta,sigma2,mu
    #x,y denote the desired coordinates, Delta the step size of the chain, mu the drift, sigma2 the variance of the brownian motion and theta the scaling parameter
    beta=2*theta/sigma2
    nu=beta*mu-1
    p=1
    p=p*beta*((y/x)**(nu/2))*np.exp(theta*nu*Delta/2-beta*y)/(1-np.exp(-theta*Delta))
    p=p*np.exp(-beta*(x+y)/(np.exp(theta*Delta)-1))
    p=p*spec.iv(nu,beta*np.sqrt(x*y)/(np.sinh(theta*Delta/2)))
    p=p/spec.gamma(beta*mu)
    return p


        
    
