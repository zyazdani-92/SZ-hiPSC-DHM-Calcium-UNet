#!/usr/bin/env python
# coding: utf-8
"""
@author: zyazdani-92

"""


import networkx as nx
import numpy as np
import disparity_filter_weighted_graphs as dfil
import matplotlib.pyplot as plt
import community as community_louvain


def fingerprint_measures(Adj):    
    """
    Compute many statitics that characterize the topology of the weighted-undirected graph
    corresponding to the adjacency matrix given as argument

    Refs:

    1. Esfahlani et al. Local structure-function relationships in human brain networks across the lifespan.
        Nat Commun 13, 2053 (2022). https://doi.org/10.1038/s41467-022-29770-y

    2. Thibeault, V., Allard, A. and Desrosiers, P., 2022. The low-rank hypothesis of complex systems.
        arXiv preprint arXiv:2208.04848.

    3. Fornito, A., Zalesky, A. and Bullmore, E., 2016. Fundamentals of brain network analysis. Academic Press.

    4. brain-connectivity-toolbox.net

    5. Rubinov, M. and Sporns, O., 2010. Complex network measures of brain connectivity: uses and interpretations.
        Neuroimage, 52(3), pp.1059-1069.
    """
    np.fill_diagonal(Adj, 0)
    F = nx.Graph(Adj)
    F.remove_edges_from(nx.selfloop_edges(F))
    print('F')
    print(F.number_of_nodes(),F.number_of_edges())

    components = sorted(nx.connected_components(F), key=len, reverse=True)
    # Analysis based on the largest connected component
    largest_component = components[0]
    F_c = F.subgraph(largest_component).copy() # digraph induced by largest connected  component

    
    # Compute the 'alpha' value for each edge.
    dfil.compute_alpha(F_c)

    # Find the optimal value for alpha. The dataframe used to find the optimal
    #   value for alpha is saved to `finding_optimal_alpha.csv.zip`.
    dfil.find_optimal_alpha(F_c, save_optimal_alpha_data=False, method='elbow')

    # Plot the position of the optimal value for alpha.
    dfil.plot_optimal_alpha(F_c)

    # Create a filtered version of the original graph using the optimal value for alpha.
    backbone = dfil.filter_graph(F_c)
    print('backbone')
    print(backbone.number_of_nodes(),backbone.number_of_edges())
    
    # Find the connected components
    components = sorted(nx.connected_components(backbone), key=len, reverse=True)
    # Analysis based on the largest connected component
    largest_component = components[0]
    backbone_c = backbone.subgraph(largest_component).copy() # digraph induced by largest connected  component
    print('backbone LCC')
    print(backbone_c.number_of_nodes(),backbone_c.number_of_edges())

    adj_backbone_c = nx.to_numpy_array(backbone_c)

    # Possitive weight
    backbone_pos_c = non_negative_weights(backbone_c)
    adj_backbone_pos_c = nx.to_numpy_array(backbone_pos_c)

    print('graph init')

    
        
    
    measures = {'Assortativity': nx.degree_assortativity_coefficient(backbone, weight='weight'),
                'Average clustering coefficient': nx.average_clustering(backbone, weight='weight'),
                'Average shortest path length': normalized_shortest_path_length_analysis(backbone_c),
                'Average square clustering': average_square_clustering(backbone),
                'Degree analysis': normalized_degree_analysis(backbone),
                'Density': nx.density(backbone),
                'Edge betweenness centrality analysis': edge_betweenness_centrality(backbone_c),
                'Eccentricity analysis': eccentricity_analysis(backbone_pos_c),
                'Fraction of nodes in the barycenter': fraction_barycenter(backbone_c),
                'Fraction of nodes in the center': fraction_center(backbone_c),
                'Fraction of nodes in the periphery': fraction_periphery(backbone_c),
                'Global efficiency': nx.global_efficiency(backbone), #possitve network
                'Local efficiency analysis': local_efficiency(backbone),
                'Modularity': Louvain_modularity(backbone_pos_c),
                'Node betweenness centrality analysis': node_betweenness_centrality(backbone_c),
                'Node strength analysis': Node_strength(backbone),
                'Rich club coefficient': rich_club_coefficient(backbone),
                'Transitivity': nx.transitivity(backbone)
               }
    return measures


def non_negative_weights(graph):
    """
    Ensure that all edge weights in the graph are non-negative.

    Parameters:
    - graph: NetworkX graph

    Returns:
    - graph: NetworkX graph with non-negative weights
    """
    for u, v, d in graph.edges(data=True):
        if d.get('weight', 0) < 0:
            d['weight'] = abs(d['weight'])
    return graph

def normalize_adjacency(Adj, new_min=-1, new_max=1):
    """
    Normalize an adjacency matrix so that its values fall between new_min and new_max.

    Parameters:
    - Adj: np.ndarray, the adjacency matrix to normalize
    - new_min: float, the new minimum value for normalization (default is -1)
    - new_max: float, the new maximum value for normalization (default is 1)

    Returns:
    - normalized_Adj: np.ndarray, the normalized adjacency matrix
    """
    old_min = Adj.min()
    old_max = Adj.max()
    print(old_min,old_max)
    
    # Step 1: Subtract the old minimum from each element
    normalized_Adj = Adj - old_min
    
    # Step 2: Divide by the range (old_max - old_min)
    normalized_Adj = normalized_Adj / (old_max - old_min)
    
    # Step 3: Multiply by the new range (new_max - new_min)
    normalized_Adj = normalized_Adj * (new_max - new_min)
    
    # Step 4: Add the new minimum value
    normalized_Adj = normalized_Adj + new_min
    
    return normalized_Adj

def dict_to_vec(dictionary):
    """
    Convert a dictionary, whose values are numbers or arrays of different size, into
    a row vector
    """
    vector = np.array([])
    for key in dictionary.keys():
        vector = np.hstack([vector,np.array(dictionary[key])])
    return vector

def basic_stats(data, with_entropy = False):
    """
    Compute 3 basic statistics that describe the distribution of the input data,
    collected in a n-dimenional array and converted into a one-dimensional array.
    The output contains: 
        - two measures of centrality, the median(med) and the mean or average (avg)
        - one measures of dispersion, the stadard deviation (std) 
        
    Arguments
    ---------
    data            numpy array, each element considered a a sample value from a random variable
    with_entropy    boolean, default False, if True the normalized entropy
    
    Returns
    -------
    statistics      dictionary with keys 'min', 'max', 'med', 'avd', 'std'
                    corresponding to the 7 measures explained above, and values being floats 
    """
    data_vector = (np.array(data)).flatten() # np.array to avoid problem with scipy matrix type
    minval = data_vector.min()
    maxval = data_vector.max()
    med = np.median(data_vector)
    avg = data_vector.mean()
    std = data_vector.std()
    statistics = {'min':minval, 'max':maxval,'avg': avg,'std': std,'med': med}
    return statistics


def fraction_center(G): 
    """ Fraction of vertices contained in the center of a graph
    The center is the set of nodes with eccentricity equal to radius."""
    return len(nx.center(G, weight='weight'))/G.number_of_nodes()

def fraction_periphery(G):
    """ Fraction of vertices that belong to the periphery of a graph
    The periphery is the set of nodes with eccentricity equal to the diameter."""
    return len(nx.periphery(G, weight='weight'))/G.number_of_nodes()

def fraction_barycenter(G):
    """ Return the proportion of vertices in a connected graph that belong to the barycenter """
    return len(nx.barycenter(G , weight='weight'))/G.number_of_nodes()

def normalized_degree_analysis(G):
    """
    Calculate the normalized degree of each node and return basic statistics for these normalized degrees.

    Parameters:
    - G: NetworkX graph (undirected)

    Returns:
    - Vector containing basic statistics (min, max, avg, std, med) of the normalized degrees
    """
    degrees = np.array(list(dict(G.degree()).values()))/G.number_of_nodes()
    return dict_to_vec(basic_stats(degrees))

def Node_strength(G):
    """
    Calculate the strength of each node and return statistics for a weighted undirected graph.
    Node strength in this context is defined as the sum of the weights of edges connected to that node.

    Parameters:
    - G: NetworkX graph (undirected)

    Returns:
    - statistics: Vector containing the statistics of the node strengths
    """
    # Calculate the strength of each node
    strength = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes}
    
    return dict_to_vec(basic_stats(list(strength.values())))

def average_square_clustering(G):
    """
    Compute the average over all square clustering coefficients of a NetworkX undirected graph
    using the function nx.square_clustering. The latter function doen't work 
    properly for directed graphs. Each square clustering coefficient is a number between 
    0 and 1 giving the fraction of possible squares that are observed in the graph.
    """
    dict_square_clustering= nx.square_clustering(G)
    square_clustering = np.array(list(dict_square_clustering.values()))
    avg = np.mean(square_clustering)
    if np.mean(square_clustering)>1:
        print('  Problem: average square clustering equal to '+ str(avg))
        avg = 1.0
    return avg


def local_efficiency(G):
    """
    Calculate the local efficiency of each node for a weighted undirected graph. Find all-pairs shortest path lengths using         Floyd’s algorithm.
    Floyd’s algorithm is appropriate for finding shortest paths in dense graphs or graphs with negative weights when Dijkstra’s     algorithm fails.

    Parameters:
    - G: NetworkX graph (undirected)

    Returns:
    - local_eff: Dictionary containing the local efficiency of each node
    - avg_local_eff: Average local efficiency of the graph
    """
    local_eff = {}

    for node in G.nodes:
        # Get the subgraph induced by the neighbors of the node
        neighbors = list(G.neighbors(node))
        subgraph = G.subgraph(neighbors)

        if len(subgraph) == 0 or len(subgraph) == 1:
            local_eff[node] = 0
        else:
            # Calculate the efficiency of the subgraph using Floyd-Warshall
            path_lengths = dict(nx.floyd_warshall(subgraph, weight='weight'))
            inv_distances = 0
            node_pairs = 0

            for u in subgraph.nodes:
                for v in subgraph.nodes:
                    if u != v:
                        try:
                            distance = path_lengths[u][v]
                            if distance > 0:
                                inv_distances += 1 / distance
                                node_pairs += 1
                        except KeyError:
                            continue

            if node_pairs > 0:
                local_eff[node] = inv_distances / node_pairs
            else:
                local_eff[node] = 0

    avg_local_eff = sum(local_eff.values()) / len(local_eff) if local_eff else 0

    return dict_to_vec(basic_stats(list(local_eff.values())))

def eccentricity_analysis(g):
    """
    Compute the normalized eccentricities for all nodes of a connected graph
    and return basic statitistics describing the distribution of excenticities.
    """

    eccentricities = np.array(list(dict(nx.eccentricity(g, weight='weight')).values()))/(g.number_of_nodes()-1)
    
    
    return dict_to_vec(basic_stats(eccentricities))

def Louvain_modularity(G):
    """
    Detect communities using the Louvain method and calculate modularity for a weighted undirected graph.

    Parameters:
    - G: NetworkX graph (undirected, weighted)

    Returns:
    - communities: Dictionary mapping each node to its community
    - modularity: Modularity value of the partition
    """
    # Ensure all edge weights are non-negative
    for u, v, data in G.edges(data=True):
        if data['weight'] < 0:
            data['weight'] = 0

    # Perform community detection using Louvain method
    partition = community_louvain.best_partition(G, weight='weight')

    # Calculate the modularity of the partition
    modularity = community_louvain.modularity(partition, G, weight='weight')

    return modularity


def node_betweenness_centrality(G):
    """
    Calculate the node betweenness centrality for each node in the graph and return basic statistics for these values.

    Parameters:
    - G: NetworkX graph (undirected, weighted)

    Returns:
    - Vector containing basic statistics (min, max, avg, std, med) of the node betweenness centrality values
    """
    node_betweenness_centrality = nx.betweenness_centrality(G,weight='weight', normalized=True)
    return dict_to_vec(basic_stats(list(node_betweenness_centrality.values())))

def edge_betweenness_centrality(G):
    """
    Calculate the edge betweenness centrality for each edge in the graph and return basic statistics for these values.

    Parameters:
    - G: NetworkX graph (undirected, weighted)

    Returns:
    - Vector containing basic statistics (min, max, avg, std, med) of the edge betweenness centrality values
    """
    edge_betweenness_centrality = nx.edge_betweenness_centrality(G , weight='weight',normalized=True)
    return dict_to_vec(basic_stats(list(edge_betweenness_centrality.values())))


def rich_club_coefficient(G):
    """
    Calculate the rich club coefficient and its average for an undirected weighted graph.

    Parameters:
    - G: NetworkX graph (undirected and weighted)

    Returns:
    - rich_club: Dictionary containing the rich club coefficient for each degree k
    - average_rich_club_coefficient: Average of the rich club coefficients
    """
    # Calculate the rich club coefficient
    rich_club = nx.rich_club_coefficient(G, normalized=False, Q='weight')

    # Extract the values from the dictionary
    rich_club_values = list(rich_club.values())
    return dict_to_vec(basic_stats(rich_club_values))


def normalized_shortest_path_length_analysis(G):
    """
    Calculate the normalized average shortest path length for a weighted undirected graph.

    Parameters:
    - G: NetworkX graph (undirected, weighted)

    Returns:
    - normalized_shortest_path: The normalized average shortest path length
    """
    num_nodes = G.number_of_nodes()  # Get the number of nodes in the subgraph
    shortest_path_length = nx.average_shortest_path_length(G, weight='weight')
    # Normalize the shortest path length
    normalized_shortest_path = shortest_path_length / (num_nodes - 1)
    return normalized_shortest_path

