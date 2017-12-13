#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
import os
import sys
import tempfile

from ds import loader


__author__ = 'sheep'


def main(graph_fname, node_vec_fname, path_vec_fname, options):
    '''\
    %prog [options] <graph_fname> <node_vec_fname> <path_vec_fname>

    graph_fname: the graph file
        It can be a file contained edges per line (e.g., res/karate_club_edges.txt)
        or a pickled graph file.
    node_vec_fname: the output file for nodes' vectors
    path_vec_fname: the output file for meta-paths' vectors
    '''

    print 'Load a HIN...'
    g = loader.load_a_HIN(graph_fname)

    print 'Generate random walks...'
    _, tmp_walk_fname = tempfile.mkstemp()
    with open(tmp_walk_fname, 'w') as f:
        for walk in g.random_walks(options.walk_num, options.walk_length):
            f.write('%s\n' % ' '.join(map(str, walk)))

    _, tmp_node_vec_fname = tempfile.mkstemp()
    _, tmp_path_vec_fname = tempfile.mkstemp()
    print 'Learn representations...'
    statement = ("model_c/bin/hin2vec -size %d -train %s -alpha %f "
                 "-output %s -output_mp %s -window %d -negative %d "
                 "-threads %d -no_circle %d -sigmoid_reg %d "
                 "" % (options.dim,
                       tmp_walk_fname,
                       options.alpha,
                       tmp_node_vec_fname,
                       tmp_path_vec_fname,
                       options.window,
                       options.neg,
                       options.num_processes,
                       1-(options.allow_circle * 1),
                       options.sigmoid_reg * 1))
    print statement
    os.system(statement)

    print 'Dump vectors...'
    output_node2vec(g, tmp_node_vec_fname, node_vec_fname)
    output_path2vec(g, tmp_path_vec_fname, path_vec_fname)
    return 0

def output_node2vec(g, tmp_node_vec_fname, node_vec_fname):
    with open(tmp_node_vec_fname) as f:
        with open(node_vec_fname, 'w') as fo:
            id2node = dict([(v, k) for k, v in g.node2id.items()])
            first = True
            for line in f:
                if first:
                    first = False
                    fo.write(line)
                    continue

                id_, vectors = line.strip().split(' ', 1)
                line = '%s %s\n' % (id2node[int(id_)], vectors)
                fo.write(line)

#FIXME: to support more than 10 different meta-paths
def output_path2vec(g, tmp_path_vec_fname, path_vec_fname):
    with open(tmp_path_vec_fname) as f:
        with open(path_vec_fname, 'w') as fo:
            id2edge_class = dict([(v, k) for k, v
                                  in g.edge_class2id.items()])
            first = True
            for line in f:
                if first:
                    first = False
                    fo.write(line)
                    continue

                ids, vectors = line.strip().split(' ', 1)
                edge = ','.join([id2edge_class[int(id_)] for id_ in ids])
                line = '%s %s\n' % (edge, vectors)
                fo.write(line)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option('-l', '--walk-length', action='store',
                      dest='walk_length', default=100, type='int',
                      help=('The length of each random walk '
                            '(default: 100)'))
    parser.add_option('-k', '--walk-num', action='store',
                      dest='walk_num', default=10, type='int',
                      help=('The number of random walks starting from '
                            'each node (default: 10)'))
    parser.add_option('-n', '--negative', action='store', dest='neg',
                      default=5, type='int',
                      help=('Number of negative examples (>0) for '
                            'negative sampling, 0 for hierarchical '
                            'softmax (default: 5)'))
    parser.add_option('-d', '--dim', action='store', dest='dim',
                      default=100, type='int',
                      help=('Dimensionality of word embeddings '
                            '(default: 100)'))
    parser.add_option('-a', '--alpha', action='store', dest='alpha',
                      default=0.025, type='float',
                      help='Starting learning rate (default: 0.025)')
    parser.add_option('-w', '--window', action='store', dest='window',
                      default=3, type='int',
                      help='Max window length (default: 3)')
    parser.add_option('-p', '--num_processes', action='store',
                      dest='num_processes', default=1, type='int',
                      help='Number of processes (default: 1)')
#TODO
#   parser.add_option('-i', '--iter', action='store', dest='iter',
#                     default=1, type='int',
#                     help='Training iterations (default: 1)')
#TODO
#   parser.add_option('-s', '--same-matrix', action='store_true',
#                     dest='same_w', default=False,
#                     help=('Same matrix for nodes and context nodes '
#                           '(Default: False)'))
    parser.add_option('-c', '--allow-circle', action='store_true',
                      dest='allow_circle', default=False,
                      help=('Set to all circles in relationships between '
                            'nodes (Default: not allow)'))
    parser.add_option('-r', '--sigmoid_regularization',
                      action='store_true', dest='sigmoid_reg',
                      default=False,
                      help=('Use sigmoid function for regularization '
                            'for meta-path vectors '
                            '(Default: binary-step function)'))
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
        sys.exit()

    sys.exit(main(args[0], args[1], args[2], options))

