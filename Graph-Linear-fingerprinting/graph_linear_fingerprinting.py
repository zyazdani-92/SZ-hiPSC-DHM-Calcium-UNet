#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pdesrosiers
"""

import numpy as np
import numpy.linalg as la

def linear_fingerprint(weight_matrix):
    """
    Take the weight matrix defining a weighted graph, compute several auxiliary matrices (degree, Laplacian, Markov matrices),
    and use spectal analysis to compute many statitics that characterize this graph. If the graph is directed, the weight matrix
    is symmetrized to make sure that all measures are well defined. 
    
    Arguments
    ---------
    weight_matrix   2d numpy array,
                    entry (i,j) represents the weight of an edge going from vertex j to vertex i
                        
    Returns
    -------
    statistics      dictionary with keys corresponding to a series measures

    References
    ----------
    1. Esfahlani et al. Local structure-function relationships in human brain networks across the lifespan.
        Nat Commun 13, 2053 (2022). https://doi.org/10.1038/s41467-022-29770-y

    2. Thibeault, V., Allard, A. and Desrosiers, P., 2022. The low-rank hypothesis of complex systems.
        arXiv preprint arXiv:2208.04848.

    3. Fornito, A., Zalesky, A. and Bullmore, E., 2016. Fundamentals of brain network analysis. Academic Press.

    4. brain-connectivity-toolbox.net

    5. Rubinov, M. and Sporns, O., 2010. Complex network measures of brain connectivity: uses and interpretations.
        Neuroimage, 52(3), pp.1059-1069.
    
    6. Chung, F.R., 1997. Spectral graph theory (Vol. 92). American Mathematical Society.
       
    7. Van Mieghem, P., 2023. Graph Spectra for Complex Networks. Cambridge University Press.
    
    8. Estrada, E. and Hatano, N., 2008. Communicability in complex networks. Physical Review E, 77(3), p.036111.
        https://doi.org/10.1103/PhysRevE.77.036111
        
    9. Crofts, J.J. and Higham, D.J., 2009. A weighted communicability measure applied to complex brain networks. 
        Journal of the Royal Society Interface, 6(33), pp.411-414.https://doi.org/10.1098/rsif.2008.0484
    """
    


    #
    # Weight matrix redefinition
    W = (weight_matrix + weight_matrix.T)/2 # Symmetric weight matrix
    W = W - np.diag(np.diag(W)) # Values on the diagonal are set to 0, corresponding to a graph with no self-loops
    n = W.shape[0]
    #
    # Statistics deduces from weights using linear algebra
    #
    statistics, D, sqrtDminus1 = analyze_weights(W)
    
    
    ## Laplacian matrix analysis: 7 measures
    #
    # Matrix definition
    L = D - W # Laplacian matrix
    L_norm = sqrtDminus1@ L @ sqrtDminus1 # Normalized laplacian matrix (Chung, 1997)
                                          # pseudo inversion used to avoid problem when some degrees are zero
        
    statisticsL_norm = analyze_laplacian(L_norm)
    statistics = {**statistics, **statisticsL_norm}
    
    
    ## Markov matrix analysis: 8 measures
    #
    P = sqrtDminus1**2 @ W # Markov transition matrix for a random walk on the weighted graph
    statisticsP = analyze_markov(P)
    statistics = {**statistics, **statisticsP}
    
    
    
    return statistics
    
    ## Weight matrix analysis: 30 measures 
def add_regularization(W, epsilon=1e-10):
    return W + np.eye(W.shape[0]) * epsilon                                            
def analyze_weights(W):
    """
    Use matrix algebra and spectal analysis to compute many statitics that characterize a weighted undirected graph.
    If the graph is directed, the weight matrix is symmetrized to make sure that all measures are well defined. 
    
    Arguments
    ---------
    W               2d numpy array,
                    entry (i,j) represents the weight of an edge going from vertex j to vertex i
                    expected to be nonnegative and symmetric with diagonal 0
                        
    Returns
    -------
    measures       dictionary with keys corresponding to a series measures
    D              2d numpy array, diagonal matrix of weighted degrees
    sqrtDminus1    2d numpy array corresponding to D^{-1/2}, computed using pseudo-inverse

    """
    ## Dictionary of measures 
    #
    measures = dict()
    
    ## Weight matrix analysis: 30 measures 
    #
    n = W.shape[0]
    #
    # Statistics about independent weights 
    #
    independent_weights = W[np.triu_indices(n, k = 1)]
    measures['Weight: mean'] = independent_weights.mean()
    measures['Weight: std'] = independent_weights.std()
    measures['Weight: min'] = independent_weights.min()
    measures['Weight: max'] = independent_weights.max()
    #
    # Eigenvalues statistics
    #
    eigenval, eigenvec = np.linalg.eigh(W) # fast method for hermitian (e.g. real symmetric) matrices
                                           # eigenvalues are already sorted in increasing order
    measures['Weight: mean eigenval'] = eigenval.mean()
    measures['Weight: std eigenval'] = eigenval.std()
    measures['Weight: min eigenval'] = eigenval[0]
    measures['Weight: dominant eigenval'] = eigenval[-1]
    measures['Weight: spectral gap'] = eigenval[-1]- eigenval[-2]
    possible_outliers = 1.0*(eigenval>-eigenval[0]) # eigenval possibly to the right of the bulk
    largest_eigenval = 1.0* ((eigenval+eigenval[0])/ np.abs(eigenval[0])>0.5) # eigenval much larger than 
                                                                            # the smallest eigenval (in abs value) 
    outliers = possible_outliers*largest_eigenval
    measures['Weight: number outliers'] = outliers.sum() # essentially the number of communities
    #
    # Eigenvector centrality
    #
    dominant_eigenvec = np.abs(eigenvec[:,-1])
    dominant_eigenvec /= dominant_eigenvec.sum()
    measures['Weight: mean eigenvec centrality'] = dominant_eigenvec.mean()
    measures['Weight: std eigenvec centrality'] = dominant_eigenvec.std()
    measures['Weight: min eigenvec centrality'] = dominant_eigenvec.min()
    measures['Weight: max eigenvec centrality'] = dominant_eigenvec.max()
    #
    # Weighted degrees
    #
    d = np.sum(W, axis=1) # diagonal matrix of weighted degrees (total input to each vertex)
    measures['Weight: mean weighted degree'] = d.mean()
    measures['Weight: std weighted degree'] = d.std()
    measures['Weight: min weighted degree'] = d.min()
    measures['Weight: max weighted degree'] = d.max()
    #
    # Vertex similarity
    #
    Gram_matrix = W @ W.T
    norms = np.sqrt(np.diag(Gram_matrix))
    norms_minus_one = np.linalg.pinv(np.diag(norms))
    similarity = norms_minus_one @ Gram_matrix @ norms_minus_one
    independent_similarities = similarity[np.triu_indices(n, k = 1)]
    measures['Weight: mean similarity'] = independent_similarities.mean()
    measures['Weight: std similarity'] = independent_similarities.std()
    measures['Weight: min similarity'] = independent_similarities.min()
    measures['Weight: max similarity'] = independent_similarities.max()
    #
    # Network communicability
    #
    D = np.diagflat(d) # degree matrix
    sqrtDminus1 = np.linalg.pinv(np.sqrt(np.abs(D))) # (degree matrix)^(-1/2), normalization factor as in [2]
                                             # pseudo inversion used to avoid problem when some degrees are zero
    
    W_normalized = sqrtDminus1 @ W @sqrtDminus1 # normalized matrix as defined in [2]
    eigval, eigvec = np.linalg.eigh(W_normalized) # np.linalg.eigh works for hermitian matrices, ensuring real eigenvalues
    
    communicability = np.array(eigvec @ np.diagflat(np.exp(eigval)) @ eigvec.T ) # scipy version, expm(A_normalized), 
                                                                                 # doesn't always converge to the right answer 
    independent_communicabilities = communicability[np.triu_indices(n, k = 1)]
    measures['Weight: mean communicability'] = independent_communicabilities.mean()
    measures['Weight: std communicability'] = independent_communicabilities.std()
    measures['Weight: min communicability'] = independent_communicabilities.min()
    measures['Weight: max communicability'] = independent_communicabilities.max()
    #
    # SVD and effective ranks
#     try:
#         W = add_regularization(W)
#         s = np.linalg.svd(W, full_matrices=True, compute_uv=False, hermitian=True)
#         measures['Weight: stable rank'] = np.sum(s**2) / (s[0]**2)
#         measures['Weight: mean singular val'] = s.mean()
#         measures['Weight: std singular val'] = s.std()
#         measures['Weight: max singular val'] = s.max()
#     except la.LinAlgError as e:
#         print("SVD did not converge:", e)
#         measures['Weight: stable rank'] = np.nan
#         measures['Weight: mean singular val'] = np.nan
#         measures['Weight: std singular val'] = np.nan
#         measures['Weight: max singular val'] = np.nan
#     #
    s = np.linalg.svd(W, full_matrices=True, compute_uv=False, hermitian=True)
    measures['Weight: stable rank'] = np.sum(s**2)/(s[0]**2)
    measures['Weight: mean singular val'] = s.mean()
    measures['Weight: std singular val'] = s.std()
    measures['Weight: max singular val'] = s.max()


  
    return measures, D, sqrtDminus1
   
    
def analyze_laplacian(L):
    """Return 7 non trivial graph measures related to the (normalized) Laplacian matrix defining 
    a diffusion process on a weighted undirected graph"""
    measures = dict()
    eigenval, eigenvec = np.linalg.eigh(L) # fast method for hermitian (e.g. real symmetric) matrices
                                           # eigenvalues are already sorted in increasing order
    measures['Laplace: spectral gap'] = eigenval[1] # second smallest eigenval
    measures['Laplace: mean eigenval'] = eigenval.mean()
    measures['Laplace: max eigenval'] = eigenval.max()
    steady_state_dist = eigenvec[:,0]
    measures['Laplace: mean steady state'] = steady_state_dist.mean()
    measures['Laplace: std steady state'] = steady_state_dist.std()
    measures['Laplace: min steady state'] = steady_state_dist.min()
    measures['Laplace: max steady state'] = steady_state_dist.max()
    
    return measures
    
        
def analyze_markov(P):
    """Return 8 non trivial graph measures related to the (normalized) Markov chain matrix defining
    a random walk on a weighted undirected graph"""
    measures = dict()
    eigenval, eigenvec = np.linalg.eigh(P.T) # fast method for hermitian (e.g. real symmetric) matrices
                                            # eigenvalues are already sorted in increasing order
                                            # Transpose used to get nontrivial equilibrium distribution
    permutation = np.argsort(np.abs(eigenval)) # last eigenvalue should the largest one, i.e., equal to 1 
    eigenval = eigenval[permutation]
    measures['Markov: spectral gap'] = 1 - np.abs(eigenval[-2])
    measures['Markov: relaxation time'] = 1/(1 - np.abs(eigenval[-2]))
    measures['Markov: mean abs eigenval'] = np.abs(eigenval).mean()
    measures['Markov: min abs eigenval'] = np.abs(eigenval).min()
    eigenvec = eigenvec[:,permutation]
    steady_state = eigenvec[:,-1]
    steady_state /= steady_state.sum() # steady_state is now a probability vector
    measures['Markov: mean steady state'] = steady_state.mean()
    measures['Markov: std steady state'] = steady_state.std()
    measures['Markov: min steady state'] = steady_state.min()
    measures['Markov: max steady state'] = steady_state.max()
    
    return measures
    
def matrix_measures(weight_matrix):
    """
    From the weight matrix defining a undirected and unsigned weighted graph, compute several auxiliary matrices (Adjacency,
    Laplacian matrix, Markov transition matrix, Toplogical Overlap Matrix (TOM), Vertex Similarity, Communicability, etc.) 
    having the same size as the original weight matrix. If the graph is directed, the weight matrix is symmetrized to make 
    sure that all measures are well defined.
    
    Should be generalized to include directed and maybe signed graphs.
    
    Arguments
    ---------
    weight_matrix   2d numpy array,
                    entry (i,j) represents the weight of an edge going from vertex j to vertex i
                    expected to be nonnegative and symmetric
                        
    Returns
    -------
    measures        dictionary with keys corresponding to a series measures
                    keys: 
                    'Adjacency',
                    'Communicability',
                    'Diffusion matrix', 
                    'Laplacian', 
                    'Markov transition matrix', 
                    'MFPT' (Mean First Passage Time), 
                    'Normalized Laplacian', 
                    'Normalized weight', 
                    'Similarity', 
                    'TOM' (Toplogical Overlap Matrix ), 
                    'Shortest path lengths',
                    'Communication efficiency'
    """
    ## Dictionary of measures 
    #
    measures = dict()
    
    ## Weight matrix 
    #
    W = (weight_matrix + weight_matrix.T)/2 # Symmetric weight matrix
    W = W - np.diag(np.diag(W)) # Values on the diagonal are set to 0, corresponding to a graph with no self-loops
    n = W.shape[0]
    
    ## Vertex similarity (weighted)
    #
    Gram_matrix = W @ W.T
    norms = np.sqrt(np.diag(Gram_matrix))
    norms_minus_one = np.linalg.pinv(np.diag(norms))
    measures['Similarity'] = norms_minus_one @ Gram_matrix @ norms_minus_one
    
    ## Network communicability (weighted)
    #
    d = np.sum(W, axis=1) # vector of weighted degrees (total input to each vertex)
    D = np.diagflat(d)    # diagonal matrix of degrees
    sqrtDminus1 = np.linalg.pinv(np.sqrt(D)) # (degree matrix)^(-1/2), normalization factor as in [2]
                                             # pseudo inversion used to avoid problem when some degrees are zero
    W_normalized = sqrtDminus1 @ W @sqrtDminus1 # normalized weighted matrix
    measures['Normalized weight'] = W_normalized
    eigval, eigvec = np.linalg.eigh(W_normalized) # np.linalg.eigh works for hermitian matrices, ensuring real eigenvalues
    measures['Communicability'] = np.array(eigvec @ np.diagflat(np.exp(eigval)) @ eigvec.T ) # scipy version, expm(A_normalized), 
                                                                                 # doesn't always converge to the right answer
    
    ## Laplacian matrix (weighted)
    #
    L = D - W 
    measures['Laplacian'] = L
    L_norm = sqrtDminus1@ L @ sqrtDminus1 # Normalized laplacian matrix (Chung, 1997)
    measures['Normalized Laplacian'] = L_norm
    eigval, eigvec = np.linalg.eigh(L_norm) 
    measures['Diffusion matrix'] = np.array(eigvec @ np.diagflat(np.exp(-eigval)) @ eigvec.T ) 
    
    
    ## Markov transition matrix (weighted) and mean first passage time (MFPT)
    # Ref:
    #     Goñi, J., Avena-Koenigsberger, A., Velez de Mendizabal, N., van den Heuvel, M. P., Betzel, R. F., & Sporns, O. (2013)
    #     Exploring the morphospace of communication efficiency in complex networks. PLoS One, 8(3), e58070.
    P = sqrtDminus1**2 @ W # Markov transition matrix for a random walk on the weighted graph
    measures['Markov transition matrix'] = P
    eigenval, eigenvec = np.linalg.eigh(P.T) # fast method for hermitian (e.g. real symmetric) matrices
                                            # eigenvalues are already sorted in increasing order
                                            # Transpose used to get nontrivial equilibrium distribution
    permutation = np.argsort(np.abs(eigenval)) # last eigenvalue should the largest one, i.e., equal to 1 
    eigenvec = eigenvec[:,permutation]
    steady_state = eigenvec[:,-1]
    steady_state /= steady_state.sum() # steady_state is now a probability vector
    otherW = np.ones((n,1))@ steady_state.reshape((1,n))
    Z = np.linalg.pinv( np.eye(n)- P + otherW ) 
    MFPT =  (np.ones((n,1)) @ np.diag(Z).reshape((1,n))-Z)/otherW # MFPT[i,j] = (Z[j,j]-Z[i,j])/ steady_state[j]
    measures['MFPT'] = MFPT # MFPT[i,j] is the expected time for a random walkwer to go from i to j
    measures['Diffusion efficiency'] = 1/MFPT 
    
    ## Adjacency matrix 
    #
    A = 1.0*(W>0)
    measures['Adjacency'] = A
    k = np.sum(A, axis=1) # vector unweighted degrees 
 
    ## Toplogical overlap matrix (TOM) and Matching Index
    # Ref: 
    #     1. Ravasz, E., Somera, A. L., Mongru, D. A., Oltvai, Z. N., & Barabási, A. L. (2002). 
    #        Hierarchical organization of modularity in metabolic networks. Science, 297(5586), 1551-1555.
    #     2. Hilgetag, C. C., Kötter, R., Stephan, K. E., & Sporns, O. (2002). Computational methods for the analysis of brain 
    #        connectivity. In Computational neuroanatomy: Principles and methods (pp. 295-335). Totowa, NJ: Humana Press.
    K_row = k.reshape((n,1))@ np.ones((1,n)) # K_row[i,j]= k_i
    K_col = np.ones((n,1)) @ k.reshape((1,n))# K_col[i,j]= k_j
    K_max = np.maximum(K_row, K_col)         # K_max[i,j] = max{k_i,k_j}
    measures['TOM'] = (A @ A.T) / K_max 
    measures['Matching index'] = (A @ A.T) / (K_row+K_col)
    
    ## Shortest path lengths
    # Ref:
    max_iter = 200
    l=1                                        # path length
    Paths = np.copy(A)                         # matrix of paths og length l
    Distances = np.copy(A)                       # distance matrix
    for k in range(1,max_iter):
        l=l+1
        Paths = Paths @ A
        idx = np.where((Paths!=0)*(Distances==0))
        if len(idx[0])==0:
            break
        else:
            Distances[idx] = l 
    idx = np.where(Distances==0)
    if len(idx[0])>0:
        Distances[idx] = np.inf  
    measures['Shortest path lengths'] = Distances
    measures['Communication efficiency'] = 1/Distances
    
    
    return measures

    