import networkx as nx
import numpy as np
import scipy as sp

from scipy.sparse import coo_matrix

from sklearn.neighbors import kneighbors_graph as knn_gr, radius_neighbors_graph as rad_gr

def load_coo_matrix(filename):
    coo = np.loadtxt(filename)
    if len(coo.shape) < 2:
        coo = np.reshape(coo,(-1,3))
    if coo.shape[0] <= 1:
        return np.array([[1.0]])    
    
    coo = coo[coo[:,0] != 0.0,:]
    
    row = coo[:,0].astype('int')
    col = coo[:,1].astype('int')
    
    data = coo[:,2].astype('float')

    aapr = coo_matrix((data,(row,col))).todense()    
    return aapr

def construct_coord_graph(filename, neighspec):
    print(neighspec)
    method, param = neighspec
    X = np.loadtxt(filename)
    if method == 'knn':
        return knn_gr(X,int(param),mode='connectivity',include_self=False).toarray()
    else: 
        return rad_gr(X,float(param),mode='connectivity',include_self=False).toarray()
    
def load_graph(filename, gformat='dense', neigh_rule=None):
    if 'gml' in filename.lower():
        gg = nx.read_gml(filename)
        aapr = nx.to_numpy_array(gg)
    elif gformat == 'dense':
        aapr = np.loadtxt(filename)
    elif gformat == 'scipy_sparse':
        aapr = load_coo_matrix(filename)
    elif gformat == 'coords':
        aapr = construct_coord_graph(filename,neigh_rule)
    else:
        pass    
        
    aa = np.triu(aapr, k=1)
    aa += aa.T
    
    dd = np.diagflat(np.sum(np.abs(aa),axis=-1))
    ll =  aa - dd       
    return ll
