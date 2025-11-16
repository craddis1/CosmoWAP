import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils

from cosmo_wap.lib.kernels import K1
from scipy.interpolate import CubicSpline

def Ir1(cosmo_funcs,zz,kernel,nfft=2048,buffer=8):
    """Do integral first intgeral"""
    # shape (mu,p) # p is some combination of r2,k and mu
    func = getattr(K1,kernel)

    d = cosmo_funcs.comoving_dist(zz)
    
    if False:
        r1 = np.linspace(0,d,nfft) # get array in r1

        arr_dict = func(r1,cosmo_funcs,zz=zz)
        p_arr = np.fft.fftshift(2*np.pi*np.fft.fftfreq(buffer*nfft,d/(nfft+1))) # p frequencies   

        for i in arr_dict.keys():
            for j in arr_dict[i].keys():
                tmp_arr = arr_dict[i][j] # (n//2,n//2)) # pad it as it is not periodic
                I_arr = np.fft.fftshift(np.fft.fft(tmp_arr,n=buffer*nfft)) # fourier transform from r to p space
                # is complex so interpolate real and imaginary bits seperately
                arr_dict[i][j] = [CubicSpline(p_arr,I_arr.real),CubicSpline(p_arr,I_arr.imag)]

    else:
        d = cosmo_funcs.comoving_dist(zz)

        nodes, weights = np.polynomial.legendre.leggauss(128)#legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)
        r1 = (d)*(nodes+1)/2.0 # sample r range [0,d]

        arr_dict = K1.TD1(r1,cosmo_funcs,zz=zz)
        
        p_arr = np.concatenate((-np.logspace(-4,0,200)[::-1],np.logspace(-4,0,200)))[:,np.newaxis]
        
        for i in arr_dict.keys():
            for j in arr_dict[i].keys():
                tmp_int = arr_dict[i][j]*np.exp(-1j*r1*p_arr)
                I_arr = (d/2)*np.sum(tmp_int*weights,axis=-1)
                arr_dict[i][j] = [CubicSpline(p_arr[:,0],I_arr.real),CubicSpline(p_arr[:,0],I_arr.imag)]
        
    return p_arr,arr_dict

def Ir2(kernel1,kernel2,mu,kk,cosmo_funcs,zz,n=128,nfft=2048,buffer=8):
    # return 2D arr in mu*kk
    p_arr,arr_dict = Ir1(cosmo_funcs,zz,kernel1,nfft=nfft,buffer=buffer)
    func = getattr(K1,kernel2)
    d = cosmo_funcs.comoving_dist(zz)
    
    nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    r2 = (d)*(nodes+1)/2.0 # sample r range [0,d]
    
    G = r2/d
    qq = kk/G
    
    baseint = BaseInt(cosmo_funcs)

    r2_arr = func(r2,mu,qq,cosmo_funcs,zz)
    
    tot_arr = np.zeros((kk*mu).shape[:-1],dtype=np.complex128)#shape (mu,kk)
    for i in arr_dict.keys():
        for j in arr_dict[i].keys():
            coef = qq**j * mu**i
            r1_arr  = (arr_dict[i][j][0](qq*mu) + 1j*arr_dict[i][j][1](qq*mu)) # get complex array
            tot_arr += ((d) / 2.0) *np.sum(weights* coef* G**(-3) *baseint.pk(qq,zz)*r2_arr*r1_arr,axis=-1)
    
    return tot_arr

def get_mu(mu,kernel1,kernel2,cosmo_funcs,kk,zz,n=256,nfft=2048,buffer=8):
    """Get mu dependent expression - could save this (k,mu)"""
    d = cosmo_funcs.comoving_dist(zz)
    
    kk = kk[:,np.newaxis]
    
    arr = Ir2(kernel1,kernel2,mu[:,np.newaxis],kk[...,np.newaxis],cosmo_funcs,zz,n=n,nfft=nfft,buffer=buffer)

    # so will sample mu differently
    
    return arr*(np.cos(kk*mu*d)+ 1j*np.sin(kk*mu*d))#np.exp(1j *kk*mu*d)#np.sum(weights*arr*np.exp(1j *kk*mu*d),axis=-1)

def get_multipole(kernel1,kernel2,l,cosmo_funcs,kk,zz,n=256,n_mu=64,nfft=2048,buffer=8):
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    mu = (2)*(nodes+1)/2.0 - 1 # sample mu range [-1,1] - so this is the natural gauss legendre range!
    return np.sum(weights*get_mu(mu,kernel1,kernel2,cosmo_funcs,kk,zz,n=n,nfft=nfft,buffer=buffer),axis=-1)