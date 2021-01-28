import argparse as ap

class SmartFormatter(ap.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return ap.HelpFormatter._split_lines(self, text, width)

class MyArgParser(ap.ArgumentParser):

    def __init__(self):
        super().__init__(description='Compute the diffusion distance between two graphs.',formatter_class=SmartFormatter)
        self.add_argument('-c', '--csv-format', metavar='[FORMAT]', type=str, default='dense',
                        help="""R|The matrix format of the graphs to consider (case insensitive).\nOnly used to interpret the entries of a CSV file (i.e. not used if the input\nis a .GML file. One of:\n\n    'dense': the CSV file entries are the entries of the adjacency matrix.\n\n    'scipy_sparse': Scipy sparse matrix. An N x 3 CSV file where the first\n    two columns R and C are expected to be row and column indices, and the\n    third column DATA are the entries of the adjacency matrix.\n    So A[R[k],C[k]] = DATA[k]. Densified before matrix calculations.\n\n    'coords': Raw vertex coordinates. If this option is chosen, then \n    the CSV is interpreted as a N x K matrix of node embeddings.\n    'connection_rule' must also be specified in this case." 
                        """)
        self.add_argument('-b', '--backend', type=str, default='scipy_only',help='R|Backend for n1 < n2 case. One of:\n\n"scipy_only": Uses scipy bounded optimization over t and alpha. \n\n "complete": Uses the method developed in the paper.\n\nScipy may be faster for some cases, but\nhas no guarantee of getting the right answer\nfor the optimization over alpha.')
    
        self.add_argument('-r', '--connection-rule', metavar='[knn or rad]',help="Rule for connecting nodes in the 'coords' case.")
        self.add_argument('-p', '--connection-param', metavar='[number of neighbors or radius]',help='R|Parameter for node connection (number of neighbors for knn,\nradius for rad. If you need more complex graph construction methods, use\nthem separately and convert to CSV.')
        
        self.add_argument('-m', '--memory-budget', metavar='M',help="For 'complete' solver, the memory budget for how many P matrices to keep during optimiztion over t. The collection of P matrices is only added to as long as the total number of floats stored (n1 * n2 * len(Plist)) is less than M.")
    
        self.add_argument('-n', '--num-cores',default=1)
        
        self.add_argument('-v', '--verbose',action='store_true', help="If true, print the optimal alpha and t values as well as distance.")
    
        self.add_argument('graph_files', nargs=2, metavar="[GRAPH FILE]", help='R|Files storing the adjacency matrix of each graph.\nMust be either a GML file or a properly formatted CSV (see --graph-format).\nIf the edges of your graph are weighted, use a CSV. Otherwise edge weights are presumed to be 1.')


            
