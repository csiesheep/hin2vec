#!/usr/bin/python
# -*- encoding: utf8 -*-

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import optparse
import sys

from tools import exp_classification



__author__ = 'sheep'


def main(graph_fname, node_xy_fname, node_classes_fname, output_fname):
    '''\
    %prog [options]
    '''

    g = load_nx_graph(graph_fname)
    node2xy = load_node_xys(node_xy_fname)
    node2color = get_node2color(g, node_classes_fname)
    print node2color

    plt.figure(1, figsize=(10,10))
    nx.draw_networkx(g, pos=node2xy,
               node_color=node2color,
               with_labels=True,
               font_size=8,
               node_size=500,
               alpha=0.3,
               width=1)
    plt.savefig(output_fname)

    return 0

def load_nx_graph(graph_fname):
    g = nx.Graph()
    with open(graph_fname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            source_node, _, dest_node, _, _ = line.strip().split('\t')
            g.add_edge(source_node, dest_node)
    return g

def load_node_xys(node_xy_fname):
    node2xy = {}
    with open(node_xy_fname) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            node, x, y = line.strip().split(' ')
            node2xy[node] = [float(x), float(y)]
    return node2xy

def get_node2color(g, node_classes_fname):
    node2class = exp_classification.load_node2classes(node_classes_fname,
                                                      is_multiclass=False)
    node2color = [float(node2class[node])-1 for node in g.nodes()]
    return node2color


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 4:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

