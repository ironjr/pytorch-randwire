import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import networkx as nx

from layer import Node


class RandomNetwork(nn.Module):
    def __init__(self, in_planes, planes, G, Gopt=False, downsample=True, drop_edge=0):
        '''Random DAG network of nodes

        Args:
            planes (int): number of channels each nodes have
            G (DiGraph): DAG from random graph generator
            Gopt (bool): whether if given G is optimal
            downsample (bool): overrides downsample setting of the top layer
        '''
        super(RandomNetwork, self).__init__()
        self.drop_edge = drop_edge

        if Gopt:
            self.Gopt = G
            self.nxorder, self.live = self._get_livevars(G)

        else:
            # Since initialization of the graphs takes minor proportion on compu-
            # tation time, we try multiple random permutation and sort to find
            # optimal configuration for minimum memory consumption. Empirically,
            # the number of tries are set to 20 which provides suboptimal result on
            # network with nodes <= 32.
            num_reorder = 20
            self.Gopt = None
            min_lives = len(G.nodes)
            for i in range(num_reorder):
                nxorder, live = self._get_livevars(G)

                # Maximum #live-vars
                nlives = max([len(nodes) for nodes in live])
                if nlives < min_lives:
                    min_lives = nlives
                    self.Gopt = G
                    self.nxorder = nxorder
                    self.live = live

                # Reorder graph
                if i is not num_reorder - 1:
                    new_order = np.random.permutation(len(G.nodes))
                    mapping = {i: new_order[i] for i in range(len(G.nodes))}
                    G = nx.relabel_nodes(G, mapping)

        # Generate nodes based on the final graph
        self.pred = self.Gopt.pred
        self.in_degree = self.Gopt.in_degree
        out_degree = self.Gopt.out_degree
        self.bottom_layer = []
        self.nodes = nn.ModuleList()
        for nxnode in self.nxorder:
            # Top layer nodes
            if self.in_degree(nxnode) == 0:
                node = Node(in_planes, planes, self.in_degree(nxnode),
                        downsample=downsample)
            else:
                node = Node(planes, planes, self.in_degree(nxnode))

            # Bottom layer nodes
            if out_degree(nxnode) == 0:
                self.bottom_layer.append(nxnode)

            self.nodes.append(node)

    def forward(self, x):
        '''

        TODO do parallel processing of uncorrelated nodes (using compiler techniques)

        Args:
            x: A Tensor with size (N, C, H, W)

        Returns:
            A Tensor with size (N, C, H, W)
        '''
        # Traversal in the sorted graph
        outs = []
        for order, nxnode in enumerate(self.nxorder):
            node = self.nodes[order]
            to_delete = []
            # Top layer nodes receive data from the upper layer
            if self.in_degree(nxnode) == 0:
                out = node(x.unsqueeze(-1)) # (N,Cin,H,W,F=1)
            else:
                y = []

                # Apply random edge drop if drop_edge is nonzero
                # Randomly remove one edge with some probability
                # if input degree > 1 (following the paper)
                drop_idx = -1
                if self.training and self.drop_edge is not 0:
                    # Whether to drop an input edge or not
                    if torch.bernoulli(torch.Tensor((self.drop_edge,))) == 1:
                        # Get random index out of predecessors
                        drop_idx = torch.randint(len(self.pred[nxnode]), (1,))

                # Aggregate input values to each node
                for i, p in enumerate(self.pred[nxnode]):
                    ipred = self.nxorder.index(p)
                    iout = self.live[order].index(ipred)
                    if i == drop_idx:
                        # Drop simply by not passing the value of predecessor
                        y.append(torch.zeros_like(outs[iout]))
                    else:
                        # Normal data flow
                        y.append(outs[iout])
                    if order is not len(self.nxorder) - 1 and \
                            ipred not in self.live[order + 1]:
                        to_delete.append(iout)
                y = torch.stack(y) # (F,N,Cin,H,W)
                y = y.permute(1, 2, 3, 4, 0) # (N,Cin,H,W,F)
                out = node(y)

            # Make output layer compact by deleting values not to be used
            if len(to_delete) is not 0:
                # Delete element in backwards in order to maintain consistency
                to_delete.sort(reverse=True)
                for i in to_delete:
                    del outs[i]
            outs.append(out)
        #  outs = [outs[i] for i in self.bottom_layer]

        # Aggregation on the output node
        out = torch.stack(outs) # (F,N,Cin,H,W)
        out = torch.mean(out, 0) # (N,Cin,H,W)
        return out

    def _get_livevars(self, G):
        '''Get node traversal order for optimal memory usage

        Args:
            G (DiGraph): graph of topology of the random network

        Returns:
            nxorder (list): node traversal order
            live (list): list of list containing live intermediate outcome at each iteration
        '''
        # Nodes are sorted in topological order (edge start nodes fisrt)
        nxorder = [n for n in nx.lexicographical_topological_sort(G)]

        # Count live variable to reduce the memory usage
        ispans = [] # indices from ordered list stored in topological order
        succ = G.succ
        for nxnode in nxorder:
            nextnodes = [nxorder.index(n) for n in succ[nxnode]]
            span = max(nextnodes) if len(nextnodes) != 0 else G.number_of_nodes()
            ispans.append(span)

        live = [None for _ in nxorder] # list of nodeids in topological order stored in topological order
        for order, nxnode in enumerate(nxorder):
            live[order] = [inode for inode, ispan in enumerate(ispans) \
                    if ispan >= order and inode < order]

        return nxorder, live


def test():
    from graph import GraphGenerator
    graphgen = GraphGenerator('WS', { 'K': 6, 'P': 0.25, })
    G = graphgen.generate(nnode=32)
    randnet = RandomNetwork(in_planes=3, planes=32, G=G, downsample=False)
    #  return
    x = torch.randn(32, 3, 224, 224)
    out = randnet(x)
    print(out.size())


if __name__ == '__main__':
    test()
