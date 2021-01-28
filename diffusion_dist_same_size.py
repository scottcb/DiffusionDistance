import autograd.numpy as np
import pymanopt

import sys

from pymanopt.manifolds import Stiefel, Oblique, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient, TrustRegions
import scipy.linalg as la
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from lapsolver import solve_dense

from scipy.sparse import coo_matrix

from time import time

import matplotlib.pyplot as plt

def match_given_alpha(diff):
    n2, n1 = diff.shape
    if n1 == n2:
        return np.eye(n1)
    P = np.zeros((n2, n1))
    am = np.argmin(diff, axis=0)
    if np.unique(am).shape[0] == n1 :
        msoln = (np.arange(n1), am)
    else:    
        msoln = solve_dense(diff.T)
    P[msoln[1], msoln[0]] = 1.0
    P = P[:,:n1]
    return P.copy() 
    
def opt_over_alpha(e1, e2, U1, U2, tt):    
    def obj(a):
        aa = np.exp(a)
        el1 = np.exp(tt*np.sqrt(1.0/aa)*e1)
        el2 = np.exp(tt*np.sqrt(aa)*e2)
         
        dd = np.sqrt(np.sum(
                np.power(el1-el2,2.0)
            )
        )    
        return dd
    soln=minimize_scalar(obj,method='bounded',bounds=[-3.5,3.5])
    return (soln.fun, soln.x)

def dist_calc(l1, l2):
    e1, V1 = la.eigh(l1)
    e1 = np.array(e1)
    V1 = np.array(V1)
    
    e2, V2 = la.eigh(l2)
    e2 = np.array(e2)
    V2 = np.array(V2)
    
    def obj(t):
        tt = np.exp(t)
        d = opt_over_alpha(e1, e2, V1, V2, tt)[0]
        return -1.0*d
        
    soln=minimize_scalar(obj,method='bounded',bounds=[-3.5,5.5])
    final_min=soln.fun*-1.0
    return final_min
    
def load_graph_laplacian(coo, wfunc=np.ones_like):
      
    if len(coo.shape) < 2:
        coo = np.reshape(coo,(-1,3))
    if coo.shape[0] <= 1:
        return np.array([[1.0]])    
    
    coo = coo[coo[:,0] != 0.0,:]
    
    row = coo[:,1].astype('int')
    col = coo[:,2].astype('int')
    
    data = wfunc(coo[:,0].astype('float'))

    aapr = coo_matrix((data,(row,col))).todense()
    
    aa = np.triu(aapr, k=1)
    aa += aa.T
    
    dd = np.diagflat(np.sum(np.abs(aa),axis=-1))
    ll =  aa - dd
    #print(data)

    return ll   

if __name__ == '__main__':
    fn1 = sys.argv[1]
    fn2 = sys.argv[2]
    
    l1 = load_graph_laplacian(fn1)
    l2 = load_graph_laplacian(fn2)

    
    print(fn1, fn2, dist_calc(l1,l2))
