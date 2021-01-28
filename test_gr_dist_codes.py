import networkx as nx
import numpy as np

#from diffusion_dist_same_size import dist_calc
from diffusion_dist_diff_size import dist_calc as dist_calc_diff


if __name__ == '__main__':
    test1 = -1.0*nx.laplacian_matrix(nx.cycle_graph(128)).todense()
    #test2 = -1.0*nx.laplacian_matrix(nx.path_graph(4*4)).todense()
    test3 = -1.0*nx.laplacian_matrix(nx.path_graph(128)).todense()
    
    #print(dist_calc(test1, test2))
    #print(dist_calc_diff(test1, test2))
    #print(dist_calc_diff(test2, test3))
    print(dist_calc_diff(test1, test3))
