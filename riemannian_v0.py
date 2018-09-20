#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy  as np
import scipy
#import pandas as pd
#import re
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%config InlineBackend.figure_format = 'retina'
#import seaborn #plotting lib, but just adding makes the matplotlob plots better
#import glob # use this for file IO later
#plt.style.use('ggplot')
#import utility_functions as uf
# %matplotlib notebook
# mpld3. check this out for interactive plot options
#import custom_functions as cf
#matplotlib.rcParams['agg.path.chunksize'] = 10000


def ReSampleCurve(c,N):
    '''
    by Alice Le Brigant
    c [dxn] : n points d'une courbe
    N [1x1] : nombre de points - 1 pour la reparamétrisation par longueur d'arc
    C [dx(N+1)] : N+1 points répartis uniformément par rapport à la longueur d'arc'''
    d = len(c[:,0])
    n = len(c[0,:])
    if n == 1:
        C = np.ones([d,N+1])*c
    else:
        delta = np.zeros([n])
        for i in range(1,n):
            delta[i] = np.linalg.norm(c[:,i] - c[:,i-1], axis=0)
        
        cumdel = np.cumsum(delta)/sum(delta)
        newdel = np.linspace(0,1,N+1)

        C = np.zeros([d,N+1])
        for k in range(d):
            cs = scipy.interpolate.CubicSpline(cumdel,c[k,0:n])
            C[k,:] = cs(newdel)
    return C

def redisc(c,p):
    ''' 
    by Alice Le BRIGANT
    Input : 
     - c [3x(n+1)] : n+1 points of a curve
     - p [1x1] : number of points added on each "segment"
       Output :
     - C [3x(p*n+1)] : rediscretized curve'''
    n=len(c[0,:])-1
    l=np.zeros([n+1])
    l[0]=0.
    for i in range(n):
        l[i+1] = np.linalg.norm(c[:,i+1]-c[:,i])
    t = np.cumsum(l)/sum(l)
    T = np.zeros([n*p+1])
    #print(T.shape)
    for i in range(n):
        T[i*p:(i+1)*p+1] = np.linspace(t[i],t[i+1],p+1)
    cs = scipy.interpolate.CubicSpline(t,c, axis=1)
    C = np.zeros([len(c),len(T)])
    for i in range(len(T)):
        C[:,i] = cs(T[i])
    return C

def hvsplit_R3(c):
    '''
    by Alice Le BRIGANT
    Computes the splitting of the speed vector field s|->cs(s) of a path of
    curves s|->c(s) in horizontal and vertical parts : cs_ver(s) = M(s)v(s)
    and cs_hor(s) = cs(s) - M(s)v(s).

    Input :
    - c [2x(n+1)x(m+1)] : path of curves in R^2.
    - verif [1 or 0]    : computes how orthogonal the horizontal and vertical
                           parts actually are and checks that M is sol of ODE

    Output :
    - M [(n+1)xm]   : defines splitting of the speed vector of the path c
                       cs_ver(s)=M(s)v(s), cs_hor(s)=cs(s)-M(s)v(s).
    - K [2x(n+1)xm] : tau(k,j) = log_H(c(k,j),c(k+1,j))
    ... if verif ==1 :
    - L [1x1]       : length of c
    - SP_cs [mx1]   : norm of speed vector SP_cs = G(cs,cs)
    - SP_hor [mx1]  : norm of horizont part SP_hor = G(cs-Mv,cs-Mv)
    - SP_ver [mx1]  : norm of vertical part SP_ver = G(Mv,Mv)
    - SP_hv [mx1]   : inner product SP_hv = G(cs-Mv,Mv)

    + plus rapide grace a vectorisation
    + plus rapide car sans verif'''
    
    n = len(c[0,:,0])-1 # 2nd dimension length
    m = len(c[0,0,:])-1 # 3rd dimension length
    
    # compute tau [3x(n+1)xm] : tau(k,j) = log(c(k,j),c(k+1,j)) 
    tau          = np.zeros([3,n+1,m])                                   
    tau[:,0:n,:] = c[:,1:n+1,0:m] - c[:,0:n,0:m]
    tau[:,n,:]   = tau[:,n-1,:]

    # compute K [(n+1)xm] : K(k,j) = |tau(k,j)|
    K = np.linalg.norm(tau, axis=0)

    # compute v [3x(n+1)xm] : v(k,j) = tau(k,j)/K(k,j)
    v = np.zeros([3,n+1,m])
    v = tau/K

    # compute lambda [nxm] : lambda = <v(k+1,j),v(k,j)>
    v_prod = v[:,1:,:] * v[:,0:n,:]
    lambd = np.sum(v_prod, axis=0)

    # compute cs [3x(n+1)xm] 
    cs = m *(c[:,:,1:m+1]-c[:,:,0:m])

    # compute Nstau [3xnxm]
    Nstau = cs[:,1:n+1,:]-cs[:,0:n,:]
    
    # compute A,B,C,D [(n-1)xm]
    A = K[1:n,0:m]/K[0:n-1,0:m]*lambd[0:n-1,:]
    B = -(1+4*K[1:n,:]/K[0:n-1,:] - 3*K[1:n,:]/K[0:n-1,:]*lambd[0:n-1,:]**2)
    C = lambd[1:n,:]

    d = v[:,1:n,:] - 3./4.*lambd[0:n-1,:]*v[:,0:n-1,:]
  
    Nstau_vk = np.sum(Nstau[:,1:n,:]*v[:,1:n,:], axis=0)
    Nstau_d  = np.sum(Nstau[:,0:n-1,:]*d[:,:,:], axis=0)

    D = Nstau_vk -4.*K[1:n,0:m]/K[0:n-1,0:m] * Nstau_d

    # compute M [(n+1)xm] : cs(k,j)^{ver} = M(k,j)v(k,j)
    M = np.zeros([n+1,m])
    for j in range(m):
        LL = np.diag(A[1:n-1,j],-1) + np.diag(B[:,j]) + np.diag(C[0:n-2,j],1)
        M[1:n,j] = np.linalg.solve(LL,D[:,j])
    return M, K

def reparhor_R3(c0, phi0, gamma_f):
    '''
    by Alice Le BRIGANT
    Calcul de la partie horizontale d'un chemin de courbes c0 à partir du
    chemin de reparamétrisations phi0 correspondant :
                      ch(s,t) = c0(s,phi0(s)^{-1}(t))
    
    Ce code rediscrétise le chemin de courbes c0 en t : à chaque temps s,
    on calcule n+1 points pour c(s,.) par interpolation géodésique entre les
    n0+1 points de c0(s,.), sauf pour s=m+1, où on donne les "vraies valeurs"
    c(m+1,t)=gamma_f(t). On fait pareil pour phi0 : on obtient (n+1)x(m+1)
    valeurs pour phi en interpolant linéairement entre les valeurs de phi0.
    On cherche parmi ces valeurs de phi(s,.) celles qui sont les plus proches
    de (k/n0, k=0...n0) et on range les indices l des phi(s,l) trouvés dans
    le tableau (n0+1)x(m+1) k_new. La 

    + plus rapide car ne fait pas appel a hvsplit_R3.m '''
    n0= len(c0[0,:])-1
    m = len(c0[0,0,:])-1
    n = len(gamma_f[0,:])-1
    p = int(n/n0)
    c    = np.zeros([3,n+1,m+1])
    phi  = np.zeros([n+1,m+1])
    ch   = np.zeros([3,n0+1,m+1])
    k_new= np.zeros([n0+1,m+1], dtype=int)
    
    # Calcul de c
    T = np.linspace(0,1,p+1)
    for j in range(m):
        for k in range(n0):
            start_i = (k)*p
            finish_i= (k+1)*p+1
            c[0,start_i:finish_i,j] = (1-T) * c0[0,k,j] + T * c0[0,k+1,j]
            c[1,start_i:finish_i,j] = (1-T) * c0[1,k,j] + T * c0[1,k+1,j]
            c[2,start_i:finish_i,j] = (1-T) * c0[2,k,j] + T * c0[2,k+1,j]
    c[:,:,m] = gamma_f
    
    # Calcul de phi --> I think this can be moved to the upper loop...
    phi[:,0] = np.arange(0,1+(1/n),1/n)
    for j in range(m):
        for k in range(n0):
            start_i = k*p # int((k-1)*p+1)
            finish_i= (k+1)*p+1 # int(k*p+2)
            phi[start_i:finish_i,j+1] = np.linspace(phi0[k,j+1],phi0[k+1,j+1],p+1)
    
    
    # Inversion de phi et construction de ch
    k_new[:,0] = np.arange(0,n+p,p)
    k_new[0,:] = 0.
    ch[:,:,0]  = c0[:,:,0]
    
    for j in range(m):
        k = 1
        for l in range(1,n):
            if k/n0 >= phi[l,j+1] and k/n0 < phi[l+1,j+1]:
                k_new[k,j+1] = l
                k += 1;

        k_new[n0,j+1] = n
        ch[:,:,j+1] = c[:,k_new[:,j+1],j+1]
       
    return ch, k_new


def geod_M_R3(ci, cf, m):
    '''
     Developed by Alice Le Brigant

     Calcule la géodésique SRV entre deux courbes dans R3, càd le chemin de
     courbes qui relie les origines par une droite et qui interpole
     linéairement entre les  SRVF (vitesses renormalisées par la racine carrée
     de leur norme).

     Inputs :
     - gamma_i [3x(n+1)] : courbe initiale
     - gamma_f [3x(n+1)] : courbe finale
     - m : discrétisation en temps de la géodésique

     Outputss :
     - c [3x(n+1)x(m+1)] : chemin de courbes géodésique de gamma_i à gamma_f
     - L : longueur de c = distance between gamma_i and gamma_f
    '''
    n = len(ci[0])-1
    T = np.linspace(0, 1, m+1)
    taui = ci[:, 1:n+1]-ci[:, 0:n]
    tauf = cf[:, 1:n+1]-cf[:, 0:n]
    norm_taui = np.linalg.norm(taui, axis=0)**(1/2)
    norm_tauf = np.linalg.norm(tauf, axis=0)**(1/2)
    norm_taui[norm_taui==0] = np.ones([len(norm_taui[norm_taui==0])])
    norm_tauf[norm_tauf==0] = np.ones([len(norm_tauf[norm_tauf==0])])
    qi = np.sqrt(n)*taui/norm_taui
    qf = np.sqrt(n)*tauf/norm_tauf
    A = np.transpose(np.tile(qi,[m+1,1,1]),(1,2,0))
    B = np.transpose(np.tile(qf,[m+1,1,1]),(1,2,0))
    q = (1-T)*A+T*B
    norm_q = np.linalg.norm(q, axis=0)
    tau = 1/n*norm_q*q
    c = np.zeros([3, n+1, m+1])
    c[:, 0, :] = (1-T)*np.tile(ci[:, 0], [m+1, 1]).T + T*np.tile(cf[:, 0], [m+1, 1]).T
    c[:, 1:n+1, :] = tau
    c = np.cumsum(c, 1)
    d1 = sum((cf[:, 0]-ci[:, 0])**2)
    d2 = 1/n * sum(sum((qf-qi)**2, 0))
    L = np.sqrt(d1+d2)
    return c, L

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%
#%                          OPTIMAL MATCHING
#%                                by
#%                          Alice Le-Brigant
#%                      Version : courbes dans R3 
#%                                avec vidéo 
#%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#% + plus rapide grace a vectorisation
#% + sans video, sans stockage
#% + alterne entre recalage rotation et reparametrisation
#% + plus rapide car sans calcul d'horizontalité

def geodshoothor_R3(ci, cf, m, p, Seuil):
    global tt, tt0, cc
    n = len(ci[0,:])-1
    N = 10
    Cf0 = redisc(cf,p)
    tvecf0 = np.linspace(0,1,(n*p+1))
    # initialisation
    Cf = Cf0.copy()
    tvecf = tvecf0.copy()
    Lh = []

    # Recalage par reparamétrisation
    a=0; ecart_courbes = 1; ecart_dist = 1;

    while ecart_courbes > Seuil :
        # Calcul de la géodésique avec discrétisation de taille n
        c, L = geod_M_R3(ci, cf, m)
        Lh.append(L)
        a += 1
        # Décomposition en parties horizontale et verticale de cs
        M, K = hvsplit_R3(c)
    
        # Calcul de phi
        phi      = np.zeros([n+1,m+1])
        phi_t    = np.zeros([n,m+1])
        phi_s    = np.zeros([n,m])
        test_phi = np.zeros([m,1])
        phi[:,0] = np.linspace(0,1,n+1) # * 1/n
        phi[n,:] = np.ones([1,m+1])
        for j in range(m):
            phi_t[0,j] = n* (phi[1,j] - phi[0,j])
            phi_s[0,j] = phi_t[0,j]* M[0,j]/(n*K[0,j])
            phi[0,j+1] = phi[0,j] + 1/m * phi_s[0,j]
            for k in range(1,n):# Matlab k = 2 : n
                if M[k,j]>0:
                    phi_t[k,j] = n*(phi[k+1,j]-phi[k,j])
                else:
                    phi_t[k,j] = n*(phi[k,j]-phi[k-1,j])

                phi_s[k,j] = phi_t[k,j]* M[k,j]/(n*K[k,j])
                phi[k,j+1] = phi[k,j] + 1/m * phi_s[k,j]

            phi[n,j+1] = 1
            test_phi[j] = sum( phi[2:n+1,j+1]-phi[1:n,j+1] < 0 ) == 0
        
        if sum(test_phi) < m : 
            print('*** warning ***')
            
        # Rediscrétisation et construction de ch
        ch, knew = reparhor_R3(c, phi, Cf)
        
        delta = (knew[:,m] - np.arange(1,n*p+2,p))/p
        
        # Reparamétrisation et rediscrétisation de la courbe cible
        cf_old = cf.copy()
        cf = ch[:,:,m]
        tvecf_old = tvecf

        for k in range(n):
            tvecf[(k+1)*p] = tvecf_old[knew[k+1,m]]
            tvecf[k*p:(k+1)*p+1] = np.linspace(tvecf[k*p],tvecf[(k+1)*p],p+1)

        Cf = np.zeros([3,len(tvecf0)])
        
        cs = scipy.interpolate.CubicSpline(tvecf0,Cf0, axis=1)
        Cf = cs(tvecf)

        _ , dist_L2 = geod_L2_R3( cf, cf_old, N); 
        
        ecart_courbes = dist_L2;
        if a > 1 :
            ecart_dist = (L_old-L)/L_old

        print('ecart_courbes est %.5f, ecart_dist est %.5f ' % (ecart_courbes, ecart_dist))
        L_old = L

    Ch = c
    curve1 = Ch[:,:,0]
    curve2 = Ch[:,:,m]
    del1 = np.zeros([len(Ch[0,:])])
    del2 = np.zeros([len(Ch[0,:])])
    for r in range(1,len(Ch[0,:])):
        del1[r] = np.linalg.norm(curve1[:,r] - curve1[:,r-1], axis=0)
        del2[r] = np.linalg.norm(curve2[:,r] - curve2[:,r-1], axis=0)
    chemin = np.array([np.cumsum(del1)/sum(del1) , np.cumsum(del2)/sum(del2)])
    chemin = ReSampleCurve(chemin,n)

    return Ch, Lh, chemin


def geod_L2_R3( gamma_i, gamma_f, m):
    ''' 
    by Alice Le Brigant
    Computes the L2-geodesic between the curves gamma_i and gamma_f of the
    hyperbolic half-plane.
    
    Input : 
    - gamma_i [2xn] : origin of the geodesic
    - gamma_f [2xn] : end of the geodesic
    
    Output :
    - c_L2 [2xnxm]  : geodesic between gamma_i and gamma_f
    - dist_L2 [1x1] : L2 distance between gamma_i and gamma_f '''

    dim = len(gamma_i)
    n = len(gamma_i[0,:])
    c = np.zeros([m,dim,n])
    I = np.ones([dim,n])
    sep = np.linspace(0,1,m)
    ecart = gamma_f - gamma_i
    dist_L2 = np.linalg.norm(ecart) * np.sqrt(1/n)
    
    for k in range(m):
        c[k,:,:] = gamma_i + ecart * (I*sep[k])
    
    c_L2 = np.transpose(c,(1,2,0))
    
    # Another way is to introduce new,axis with None...
    # c = a[..., numpy.newaxis]*b[numpy.newaxis, ...]
    return c_L2, dist_L2


data1 = np.loadtxt('femme1.txt')
data2 = np.loadtxt('femme2.txt')
start = 199
end = 220
c1 = data1[start:end,:].T
c2 = data2[start:end,:].T
c2 = np.flip(c2,1)
N =10
c1 = ReSampleCurve(c1,N)
c2 = ReSampleCurve(c2,N)
c1.shape

m=10; p=100; Seuil = 0.001
print(c1.shape)
Ch, Lh, chemin = geodshoothor_R3(c1, c2, m, p, Seuil)







