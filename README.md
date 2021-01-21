### DiffusionDistance
Diffusion distance was first introduced by Hammond et al. in 2013. My work generalizes this distance measure between graphs to handle graphs of varying size. 

The core idea of diffusion distance is to compare two graphs by comparing their heat kernels with a Frobenius norm. The formal definition of distance is the...

### Installation
Requires the following packages:
 - numpy
 - scipy
 - scikit-learn (sklearn)
 - networkx 
 - Christoph Heindl's package lapsolver:https://github.com/cheind/py-lapsolver.

### How to use
If file1 and file2 are graphs, represented as .gml, .csv (dense), or .csv (sparse) files:

Examples:
 - diff_dist file1 file2
(more documentation forthcoming)

### Acknowledgements
Thanks to my advisor, Eric Mjolsness, for his help in working through much of the theory underlying this work.
Some of the development of this package was supported by National Science Foundation NRT Award 1613361, as well as the hospitality of the Center for Nonlinear Studies at Los Alamos National Laboratory. 

### Accompanying Paper
If you use this implementation of Graph Diffusion Distance, please cite: 

Scott, Cory B., and Eric Mjolsness. "Novel diffusion-derived distance measures for graphs." arXiv preprint arXiv:1909.04203 (2019).
