from networkx import DiGraph as _DiGraph
from networkx import Graph as _Graph
from networkx import is_directed as _is_directed
from networkx import isolates as _isolates
from numpy import absolute as _absolute
from numpy import nan as _nan
from numpy import power as _power
from numpy import sqrt as _sqrt
from pandas import DataFrame as _DataFrame
from pandas import concat as _concat



def compute_alpha(G, weight='weight'):
    ''' Computes significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009.

        Args
        ----
            G: Weighted NetworkX graph

            weight: string (default: `weight`)
                Edge property corresponding to the weight.

        References
        ----------
        .. M. Á. Serrano, M. Boguñá and A. Vespignani. Extracting the multiscale backbone of complex weighted networks. Proc. Natl. Acad. Sci. U.S.A. 106, 6483-6488 (2009)
    '''
    if _is_directed(G):
        _compute_alpha_directed(G, weight)
    else:
        _compute_alpha_undirected(G, weight)



def _compute_alpha_directed(G, weight):

    for u in G:

        k_out = G.out_degree(u)
        k_in = G.in_degree(u)

        if k_out > 1:
            sum_w_out = sum(_absolute(G[u][v][weight]) for v in G.successors(u))
            for v in G.successors(u):
                w = G[u][v][weight]
                p_ij_out = float(_absolute(w))/sum_w_out
                alpha_ij_out = _power(1 - p_ij_out, k_out - 1)
                G.add_edge(u, v, alpha_out=float('{:.6f}'.format(alpha_ij_out)))

        elif k_out == 1 and G.in_degree(list(G.successors(u))[0]) == 1:
            # we need to keep the connection as it is the only way to maintain the connectivity of the network
            # there is no need to do the same for the k_in, since the link is built already from the tail
            v = list(G.successors(u))[0]
            G.add_edge(u, v, alpha_out=float('{:.6f}'.format(0)), alpha_in=float('{:.6f}'.format(0)))

        elif k_out == 1:
            for v in G.successors(u):
                G.add_edge(u, v, alpha_out=1.)

        if k_in > 1:
            sum_w_in = sum(_absolute(G[v][u][weight]) for v in G.predecessors(u))
            for v in G.predecessors(u):
                w = G[v][u][weight]
                p_ij_in = float(_absolute(w))/sum_w_in
                alpha_ij_in = _power(1 - p_ij_in, k_in - 1)
                G.add_edge(v, u, alpha_in=float('{:.6f}'.format(alpha_ij_in)))

        elif k_in == 1:
            for v in G.predecessors(u):
                G.add_edge(v, u, alpha_in=float('{:.6f}'.format(1)))



def _compute_alpha_undirected(G, weight):

    for u in G:

        k = len(G[u])

        if k > 1:
            sum_w = sum(_absolute(G[u][v][weight]) for v in G[u])
            for v in G[u]:
                w = G[u][v][weight]
                p_ij = float(_absolute(w))/sum_w
                alpha_ij = _power(1 - p_ij, k - 1)
                try:
                    alpha = G[v][u]['alpha']
                except KeyError:
                    alpha = 1
                G.add_edge(u, v, alpha=float('{:.6f}'.format(min([alpha, alpha_ij]))))

        elif k == 1 and G.degree(list(G[u])[0]) == 1:
            # we need to keep the connection as it is the only way to maintain the connectivity of the network
            v = list(G[u])[0]
            G.add_edge(u, v, alpha=float('{:.6f}'.format(0)))

        elif k == 1:
            for v in G[u]:
                try:
                    alpha = G[v][u]['alpha']
                except KeyError:
                    alpha = 1
                G.add_edge(u, v, alpha=float('{:.6f}'.format(min([alpha, 1]))))



def filter_graph(G, alpha_t=None, cut_mode='or'):
    ''' Performs a cut of the graph previously filtered through the disparity_filter function.

        Args
        ----
        G: Weighted NetworkX graph

        alpha_t: double (default uses the value obtained previously using `find_optimal_alpha`)
            The threshold for the alpha parameter that is used to select the surviving edges.
            It has to be a number between 0 and 1.

        cut_mode: string (default='or')
            Possible strings: 'or', 'and'.
            It applies only to directed graphs. It represents the logic operation to filter out edges
            that do not pass the threshold value, combining the alpha_in and alpha_out attributes
            resulting from the disparity_filter function.

        Returns
        -------
        B: Weighted NetworkX graph
            The resulting graph contains only edges that survived from the filtering with the alpha_t threshold

        Reference
        ---------
        .. M. Á. Serrano, M. Boguñá and A. Vespignani. Extracting the multiscale backbone of complex weighted networks. Proc. Natl. Acad. Sci. U.S.A. 106, 6483-6488 (2009)
    '''
    if _is_directed(G):
        return _filter_graph_directed(G, alpha_t, cut_mode)
    else:
        return _filter_graph_undirected(G, alpha_t)



def _filter_graph_directed(G, alpha_t, cut_mode='or'):

    if alpha_t == None:
        alpha_t = G.graph['optimal_alpha']

    B = _DiGraph()
    B.add_nodes_from(G.nodes(data=True))

    for u, v, w in G.edges(data=True):

        alpha_in = w['alpha_in']
        alpha_out = w['alpha_out']

        if cut_mode == 'or':
            if alpha_in < alpha_t or alpha_out < alpha_t:
                B.add_edge(u, v)
                B[u][v].update(G[u][v])

        elif cut_mode == 'and':
            if alpha_in < alpha_t and alpha_out < alpha_t:
                B.add_edge(u, v)
                B[u][v].update(G[u][v])

    return B



def _filter_graph_undirected(G, alpha_t):

    if alpha_t == None:
        alpha_t = G.graph['optimal_alpha']

    B = _Graph()
    B.add_nodes_from(G.nodes(data=True))

    for u, v, w in G.edges(data=True):

        alpha = w['alpha']

        if alpha < alpha_t:
            B.add_edge(u, v)
            B[u][v].update(G[u][v])

    return B



def find_optimal_alpha(G, save_optimal_alpha_data=False, method='elbow', weight='weight'):
    ''' Finds the optimal value for alpha.

        Args
        ----
        G: Weighted NetworkX graph

        asave_optimal_alpha_data: bool (default is False)
            If `True`, the dataframe used to find the optimal value for alpha
            is saved to `finding_optimal_alpha.csv.zip`.

        method: string (default='elbow')
            Possible strings: 'elbow'.
            The method `elbow` computes the optimal value for alpha as the one that minimizes
            the number of edges while maximizing the number of vertices. The function
            `plot_optimal_alpha()` produces an illustration of the method.

        Reference
        ---------
        .. M. Á. Serrano, M. Boguñá and A. Vespignani. Extracting the multiscale backbone of complex weighted networks. Proc. Natl. Acad. Sci. U.S.A. 106, 6483-6488 (2009)
    '''

    def _remove_edge_and_count_vertices(x, g):
        g.remove_edge(x.v_source, x.v_target)
        g.remove_nodes_from(list(_isolates(g)))
        return g.number_of_nodes()

    G_copy = G.copy()

    if G_copy.is_directed():
        for u, v in G_copy.edges():
            G_copy[u][v]["alpha"] = min(G[u][v]["alpha_in"], G[u][v]["alpha_out"])

    nb_vertices = G.number_of_nodes()
    nb_edges = G.number_of_edges()

    df = _DataFrame([[u, v, d[weight], d['alpha']] for u, v, d in G_copy.edges(data=True)],
                      columns=['v_source', 'v_target', weight, 'alpha'])

    df.sort_values(by=['alpha', weight], ascending=False, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df['Remaining number of edges'] = G_copy.number_of_edges() - df.index.values - 1
    df['Remaining number of vertices'] = df.apply(_remove_edge_and_count_vertices, g=G_copy, axis=1)
    df['Remaining fraction of edges'] = df['Remaining number of edges'] / nb_edges
    df['Remaining fraction of vertices'] = df['Remaining number of vertices'] / nb_vertices

    #df.drop_duplicates(subset=['alpha'], keep='last', inplace=True)

    last_row = {'v_source': _nan, 'v_target': _nan,
                weight: _nan, 'alpha': _nan,
                'Remaining number of edges': nb_edges, 'Remaining number of vertices': nb_vertices,
                'Remaining fraction of edges': 1, 'Remaining fraction of vertices' :1}
    df = _concat([_DataFrame([last_row]), df])


    if method == 'elbow':
        G.graph['optimal_alpha_data_method'] = method
        df['dist_to_diagonal'] = _absolute(df['Remaining fraction of vertices'] - df['Remaining fraction of edges']) / _sqrt(2)
    else:
        raise KeyError('This method for finding optimal alpha has not been implemented.')

    df.reset_index(drop=True, inplace=True)

    idx = df['dist_to_diagonal'].idxmax()
    G.graph['optimal_alpha_data_index'] = idx
    alpha_t = df.iloc[idx]['alpha']
    G.graph['optimal_alpha'] = alpha_t

    G.graph['optimal_alpha_data'] = df

    if save_optimal_alpha_data:
        df.to_csv('finding_optimal_alpha.csv.zip', compression='zip')



def plot_optimal_alpha(G):
    ''' Plots an illustration of the method used to find the optimal alpha value.

        Args
        ----
        G: Weighted NetworkX graph

        Reference
        ---------
        .. M. Á. Serrano, M. Boguñá and A. Vespignani. Extracting the multiscale backbone of complex weighted networks. Proc. Natl. Acad. Sci. U.S.A. 106, 6483-6488 (2009)
    '''

    try:
        method = G.graph['optimal_alpha_data_method']
    except KeyError:
      print('No filtering method has been run. Please run `find_optimal_alpha()`.')
    except:
      print("Something else went wrong")

    if method == 'elbow':
        import seaborn as sns
        sns.set_theme(style="ticks", palette="deep")

        df = G.graph['optimal_alpha_data']
        idx = G.graph['optimal_alpha_data_index']

        x_elbow = df.iloc[idx]['Remaining fraction of edges']
        y_elbow = df.iloc[idx]['Remaining fraction of vertices']
        x_inter = (x_elbow + y_elbow) / 2
        y_inter = (x_elbow + y_elbow) / 2

        ax = sns.lineplot(x='Remaining fraction of edges', y='Remaining fraction of vertices', data=df)

        ax.plot([0, 1, None, x_inter, x_elbow],
                [0, 1, None, y_inter, y_elbow],
                linestyle='--', linewidth=0.5)
        ax.plot(x_elbow, y_elbow, ls='None', marker='o', markersize=3)

        ax.set_aspect('equal')
        sns.despine(ax=ax)
        #ax.get_figure().savefig('finding_optimal_alpha.pdf')

    else:
        raise KeyError('No illustration has been implemented for this filtering method yet.')
