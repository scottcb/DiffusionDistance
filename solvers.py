import numpy as np
import scipy as scp
import scipy.optimize as sciopt
import scipy.linalg as la
import sys
import os
from scipy.spatial.distance import cdist
from multiprocessing import Pool
from queue import Empty as EmptyQueueException
from multiprocessing import Process, Queue, Manager, get_context
from multiprocessing.managers import BaseManager, DictProxy, ListProxy
from collections import defaultdict
from time import time

from scipy.optimize import minimize_scalar

# This parameter controls the number of allowed subsidiary threads. 
# The higher this number, the faster the optimization over P matrices.
# If this code is run from the command line, this gets overwritten with a (possibly larger) number,
# By setting .
GLOBAL_THREAD_LIMIT = 4

class MyManager(BaseManager):
    pass

MyManager.register('defaultdict', defaultdict, DictProxy)
MyManager.register('list', list, ListProxy)

from lapsolver import solve_dense

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
    
# Given two lists of values, and an assignment,
# optimize over alpha to find the best scaling factor.
def get_soln_optimum(e1, e2, soln, mode='linear', bounds=[0.00001,10.0]):
    yyy = get_soln_func(soln, e1, e2, mode)
    sss = scp.optimize.minimize_scalar(
        lambda x: yyy(x)
        , bounds=bounds
        , tol=1e-12
    )
    return (sss['x'], sss['fun'])
    
# Given two lists of values, and an assignment,
# return a function to evaluate the distance at any alpha.
def get_soln_func(M, e1, e2, mode):
    if mode == 'linear':
        el1 = e1
        el2 = e2[np.nonzero(M[0])[0]]
        # add 1e-20 to avoid division by 0. 
        return lambda x: np.sqrt(np.sum(np.power(((1.0/(1e-20 + x))*el1) - (el2*x) ,2.0)))
    else:
        el1 = e1
        el2 = e2[np.nonzero(M[0])[0]]
        return lambda x: (None,np.sqrt(np.sum(np.power(np.power(el1,(1.0/(1e-20 + x))) - np.power(el2,x) ,2.0))))[1]

      
# Given two matchings M1 and M2, do the following:
# 1) Find all the places where they agree;
# 2) Create a new, smaller matching problem where they don't agree, and solve that matching problem;
# 3)         
def merged_solution(M1, M2, e1, e2):
    agreed_M = M1 * M2
    dscore = 0
    if np.sum(agreed_M) < e1.shape[0]:
        a_nz = np.nonzero(agreed_M)
        a_idx0 = sorted(np.unique(a_nz[0]))
        a_idx1 = sorted(np.unique(a_nz[1]))
        
        disagree_M = np.ones(agreed_M.shape)
        disagree_M[a_idx0,:] = 0.0
        disagree_M[:,a_idx1] = 0.0
        
        nz = np.nonzero(disagree_M)
        idx0 = sorted(np.unique(nz[0]))
        idx1 = sorted(np.unique(nz[1]))
        
        ddiff = cdist(np.expand_dims(e2,axis=1), np.expand_dims(e1,axis=1), 'sqeuclidean')
        
        ddiff_small = cdist(np.expand_dims(e2[idx0],axis=1), np.expand_dims(e1[idx1],axis=1), 'sqeuclidean')
        
        test = np.sum(M1*ddiff) - np.sum(agreed_M * ddiff)
        if test - np.sum(np.min(ddiff_small, axis=0)) > 1e-6:
            new_nz = match_given_alpha(ddiff_small)
            agreed_M[np.ix_(idx0,idx1)] = new_nz
        else:
            agreed_M = M1.copy()
        #print(agreed_M)
    return agreed_M
    
def get_soln_intersection(e1, e2, solns, mode='linear'):
    s1func = get_soln_func(solns[0], e1, e2, mode)
    s2func = get_soln_func(solns[1], e1, e2, mode)
    newa = sciopt.least_squares(
        lambda x: s1func(x) - s2func(x),
        (solns[0][1] + solns[1][1])/2,
        bounds=(0.000001, 10.0)
    )['x']
    if mode == 'linear':
        el1 = ((1.0/(1e-20 + newa))*e1)
        el2 = newa*e2
    else:
        el1 = np.power(e1,(1.0/(1e-20 + newa)))
        el1 += 1e-10*np.arange(len(el1))
        el2 = np.power(e2,newa)
        el2 += 1e-10*np.arange(len(el2))
    newS = merged_solution(solns[0][0], solns[1][0], el1, el2)
    return (newS, newa)
    
def expand_tuple(t1, t2, e1, e2, used, mode):
    if (np.sum(t1[0] * t2[0]) < e1.shape[0]):
            new_soln = get_soln_intersection(e1, e2, (t1, t2), mode=mode)
            if np.sum(t1[0] * new_soln[0]) == e1.shape[0] or np.sum(t2[0] * new_soln[0]) == e1.shape[0]:
                pass
            elif tuple(np.nonzero(new_soln[0])[0]) in used or len(tuple(np.nonzero(new_soln[0])[0])) < e1.shape[0]:
                pass
            else:
                used.append(tuple(np.nonzero(new_soln[0])[0]))
                return new_soln
    

    
def get_soln_at_alpha(e1, e2, alp):
    e1a = np.power(e1, 1.0/alp)
    e2a = np.power(e2, alp)
    ddiff = cdist(np.expand_dims(e2a,axis=1), np.expand_dims(e1a,axis=1), 'sqeuclidean')
    rez = match_given_alpha(ddiff)
    return (rez, alp, rez*ddiff)

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



class Solver():
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

class SameSize(Solver):
    def __init__(self,l1,l2):
        super().__init__(l1,l2)
        self.opt_alpha = -1.0
        self.opt_t = -1.0
        self.opt_dist = 1e20
        
    def opt_over_alpha(self, e1, e2, tt):    
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

    def dist_calc(self):
        
        e1, V1 = la.eigh(self.l1)
        e1 = np.array(e1)
        V1 = np.array(V1)
        
        e2, V2 = la.eigh(self.l2)
        e2 = np.array(e2)
        V2 = np.array(V2)
        
        def obj(t):
            tt = np.exp(t)
            d = self.opt_over_alpha(e1, e2, tt)[0]
            return -1.0*d
            
        soln=minimize_scalar(obj,method='bounded',bounds=[-3.5,5.5])
        final_min=soln.fun*-1.0
        self.opt_t = np.exp(soln.x)
        self.opt_alpha = np.exp(self.opt_over_alpha(e1, e2, np.exp(soln.x))[1])
        self.opt_dist = final_min
        return final_min


class ScipyBackend(Solver):
    def __init__(self,l1,l2,threadcount=4):
        super().__init__(l1,l2)
        global GLOBAL_THREAD_LIMIT
        GLOBAL_THREAD_LIMIT = threadcount
        self.opt_alpha = -1.0
        self.opt_t = -1.0
        self.opt_dist = 1e20  
        
    def opt_over_alpha(self, e1, e2, tt):    
        def obj(a):
            aa = np.exp(a)
            el1 = np.exp(tt*np.sqrt(1.0/aa)*e1)
            el2 = np.exp(tt*np.sqrt(aa)*e2)
            
            ddiff = cdist(np.expand_dims(el2,axis=1), np.expand_dims(el1,axis=1), 'sqeuclidean')
            rez = match_given_alpha(ddiff)
             
            dd = np.sqrt(np.sum(
                    np.power(ddiff*rez,2.0)
                )
            )
                
            return dd
        soln=minimize_scalar(obj,method='bounded',bounds=[-3.5,3.5])
        return (soln.fun, soln.x)

    def dist_calc(self):
        
        e1, V1 = la.eigh(self.l1)
        e1 = np.array(e1)
        V1 = np.array(V1)
        
        e2, V2 = la.eigh(self.l2)
        e2 = np.array(e2)
        V2 = np.array(V2)
        
        def obj(t):
            tt = np.exp(t)
            d = self.opt_over_alpha(e1, e2, tt)[0]
            return -1.0*d
            
        soln=minimize_scalar(obj,method='bounded',bounds=[-3.5,5.5])
        final_min=soln.fun*-1.0
        self.opt_t = np.exp(soln.x)
        self.opt_alpha = np.exp(self.opt_over_alpha(e1, e2, np.exp(soln.x))[1])
        self.opt_dist = final_min
        return final_min
    
class Complete(Solver):
    def __init__(self,l1,l2,threadcount=4,pbud=6e15,p_reopt=False):
        super().__init__(l1,l2)
        self.threadcount = threadcount
        self.opt_alpha = -1.0
        self.opt_t = -1.0
        self.opt_dist = 1e20  
        self.opt_P = np.zeros((l2.shape[0],l1.shape[0]))
        self.p_budget = pbud
        self.reoptimize_P = p_reopt

    def parallel_expand(self, e1, e2, frontier, mode='linear'):
        new_frontier = frontier[:]
        with Manager() as manager:
            used = manager.list(set([tuple(np.nonzero(item[0])[0].astype('int')) for item in frontier]))
            tups = [(frontier[i], frontier[i+1], e1, e2, used, mode) for i in range(len(frontier) - 1)]
            with get_context('fork').Pool(self.threadcount) as pool:
                #print(self.threadcount, len(frontier))
                new_frontier = frontier + [item for item in pool.starmap(expand_tuple, tups) if item is not None]
            return sorted(new_frontier, key= lambda x: tuple(np.nonzero(x[0])[0].astype('int')))
        
    def get_P_linear(self, e1, e2):
        n1 = e1.shape[0]
        n2 = e2.shape[0]
        initial_soln_set = [(np.eye(n2, n1),1e-6), (np.eye(n2, n1)[::-1,::-1],10.0)]
        psize = 0
        while len(initial_soln_set) > psize:
            psize = len(initial_soln_set)
            initial_soln_set = self.parallel_expand(e1, e2, initial_soln_set, mode='linear')
        optima = [get_soln_optimum(e1, e2, soln) for soln in initial_soln_set]
        return initial_soln_set
        
    def get_P_exponential(self, e1, e2, knownP, knownA=None):
        n1 = e1.shape[0]
        n2 = e2.shape[0]
        psize = 0
        while len(knownP) > psize and n1*n2*len(knownP) < self.p_budget:
            psize = len(knownP)
            #print(len(knownP))
            knownP = self.parallel_expand(e1, e2, knownP, mode='exp')
        if knownA is not None:
            optima = [get_soln_optimum(e1, e2, soln, mode='exp', bounds=[.33*aa, 3.0*aa]) for soln, aa in zip(knownP, knownA)] 
        else:
            optima = [get_soln_optimum(e1, e2, soln, mode='exp') for soln in knownP] 
        opt_sort = sorted(enumerate(optima), key = lambda x: x[1][1])
        return knownP, sorted(enumerate(optima), key = lambda x: x[1][1])[0], opt_sort

             
    def dist_calc(self):
        e1, V1 = la.eigh(self.l1)
        e2, V2 = la.eigh(self.l2)    
        P_set = self.get_P_linear(e1, e2)
        dmax = 0.0
        if self.reoptimize_P:
            t0 = .001
            P_set, minn, op = self.get_P_exponential(np.power(np.exp(e1),t0), np.power(np.exp(e2),t0), P_set)
            def obj(tt,P_set):
                P_set, minn, op = self.get_P_exponential(np.power(np.exp(e1),tt), np.power(np.exp(e2),tt), P_set)
                #print("{%f,%f}," % (tt, minn[1][1]))
                self.opt_alpha = minn[1][0]
                return -1.0*minn[1][1]
            soln=minimize_scalar(obj,method='bounded',bounds=[0.0005,3.5],args=(P_set,))
            dmax = -1.0*soln.fun
        else:
            t0 = .001
            P_set, minn, op = self.get_P_exponential(np.power(np.exp(e1),t0), np.power(np.exp(e2),t0), P_set)
            def obj(tt,P_set):
                global tempf
                def tempf(soln):
                    return get_soln_optimum(np.power(np.exp(e1),tt), np.power(np.exp(e2),tt), soln, mode='exp')
                with get_context('fork').Pool(self.threadcount) as pool:
                    optima = pool.map(tempf,P_set)
                bestval = sorted(enumerate(optima), key = lambda x: x[1][1])[0]
                self.opt_alpha = bestval[1][0]
                self.opt_P = P_set[bestval[0]]
                #print("{%f,%f}," % (tt, bestval))print(minn)
                return -1.0*bestval[1][1]
            soln=minimize_scalar(obj,method='bounded',bounds=[0.0005,3.5],args=(P_set,))
            dmax = -1.0*soln.fun
            
        self.opt_dist = -1.0*soln.fun
        self.opt_t = soln.x
        return dmax
        
