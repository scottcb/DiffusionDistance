import argparse as ap
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

import networkx as nx

from scipy.optimize import minimize_scalar

from smart_format import MyArgParser

from graph_loaders import load_graph

from solvers import SameSize, ScipyBackend, Complete
   
if __name__ == '__main__':
    
    parser = MyArgParser()
    args = parser.parse_args()

    file1, file2 = args.graph_files
    
    if args.csv_format.lower() == 'coords' and (args.connection_rule == None or args.connection_param == None):
        raise Exception('Please specify a neighboring rule and parameter.')
    
    l1 = load_graph(file1, gformat=args.csv_format, neigh_rule=(args.connection_rule,args.connection_param))
    l2 = load_graph(file2, gformat=args.csv_format, neigh_rule=(args.connection_rule,args.connection_param))

    
    if l1.shape[0] > l2.shape[0]:
        raise Exception("G1 must be smaller (fewer nodes) than G2") 
    
    elif l1.shape[0] == l2.shape[0]:
        solver = SameSize(l1,l2)#,threadcount=int(args.num_cores))
    else:
        if args.backend == 'complete':  
            solver = Complete(l1,l2,threadcount=int(args.num_cores))
        else:
            solver = ScipyBackend(l1,l2)    
    solver.dist_calc()
    if args.verbose:
        print(file1, file2, solver.opt_t, solver.opt_alpha, solver.opt_dist)
    else:
        print(file1, file2, solver.opt_dist)
    """
    try:
        options, rest = parser.parse_known_args()
    except BaseException as e:
        print(e)
        
    if len(rest) == 2:
        
    else:
        raise Exception("
    """    
