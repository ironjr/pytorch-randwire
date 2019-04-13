import networkx as nx


class GraphGenerator(object):
    def __init__(self, model, params, directed=True):
        '''Graph generation for RandWire neural net

        TODO add more variaty than the paper has presented
        especially convertion logic to directed graph

        Args:
            model (str): either 'ER', 'BA' or 'WS'
            params (dict): parameters for random graph generators with capital keys
            directed (bool): whether output to be a directed graph
        '''
        self.model = model
        self.params = params
        self.directed = directed

        if model == 'ER':
            assert 'P' in params, 'Erdos-Renyi model requires param P'
            assert params['P'] <= 1 and params['P'] >= 0, 'Param P of ER model should be a real number between 0 and 1'
        elif model == 'BA':
            assert 'M' in params, 'Barabasi-Albert model requires param M'
        elif model == 'WS':
            assert 'K' in params and 'P' in params, 'Watts-Strogatz model requires param K and P'
            assert self.params['K'] % 2 == 0, 'Param K of WS model should be an even number'
            assert params['P'] <= 1 and params['P'] >= 0, 'Param P of WS model should be a real number between 0 and 1'
        else:
            raise NotImplementedError

    def generate(self, nnode, seed=None):
        # Generate an undirected graph
        if self.model == 'ER':
            # Force a connected graph
            while True:
                G = nx.erdos_renyi_graph(nnode, self.params['P'], seed=seed)
                if nx.is_connected(G):
                    break
        elif self.model == 'BA':
            assert self.params['M'] < nnode, 'Param M of BA model should be an integer smaller than N'
            G = nx.barabasi_albert_graph(nnode, self.params['M'], seed=seed)
        elif self.model == 'WS':
            assert self.params['K'] < nnode, 'Param K of WS model should be an integer smaller than N'
            G = nx.watts_strogatz_graph(nnode, self.params['K'], self.params['P'], seed=seed)
        else:
            raise NotImplementedError

        # Make directed graph
        if self.directed:
            G = nx.DiGraph(G)
            ebunch = []
            for e in G.edges:
                if e[0] >= e[1]:
                    ebunch.append(e)
            G.remove_edges_from(ebunch)
        # Since the DAG is not sorted, topological sorting algorithm is needed
        return G


def get_graphs(model, params, ngraphs, nnodes, seeds=None):
    '''Get random graphs needed for initialization of the network

    Args:
        model (str): name of the random graph generation model
        params (dict): parameters for each model
        ngraphs (int): number of graphs to generate
        nnodes (int or list(int)): number of nodes for each graphs
        seeds (int or list(int), optional): seeds for each graph

    Returns:
        List of generated DAGs as `networkx.DiGraph`
    '''
    # Assume all the graphs have the same number of nodes
    if isinstance(nnodes, int):
        nnodes = [nnodes,] * ngraphs

    # If only one seed is given, list of consecutive integers are used
    if seeds is None:
        seeds = [None,] * ngraphs
    elif isinstance(seeds, int):
        seeds = [s for s in range(seeds, seeds + ngraphs - 1)]

    # Generate random graph
    gen = GraphGenerator(model, params, directed=True)

    Gs = []
    for i in range(ngraphs):
        Gs.append(gen.generate(nnodes[i], seed=seeds[i]))

    return Gs


def test():
    model = 'ER'
    params = {
            'P': 0.5,
            'M': 4,
            'K': 4,
            }
    gen = GraphGenerator(model, params, directed=True)
    G = gen.generate(nnode=32, seed=2)
    print('Nodes: ', G.nodes)
    print('Edges: ', G.edges)

    # Draw graph 
    import matplotlib.pyplot as plt
    pos = nx.layout.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='black')
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='white')
    nx.draw_networkx_edges(G, pos, node_size=80, arrowstyle='->', arrowsize=10,
            edge_color='black', width=2)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    test()
