#!/usr/bin/python
# -*- encoding: utf8 -*-

import copy
import sys


__author__ = 'sheep'


class Node(object):
    '''
        Node for Nodes
    '''
    def __init__(self, node_id, count=0):
        self.node_id = node_id
        self.count = count

    def __str__(self):
        return '%s(%d)' % (self.node_id, self.count)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        if self.node_id != other.node_id:
            return False
        if self.count != other.count:
            return False
        return True


class NodeVocab(object):

    def __init__(self):
        self.nodes = []
        self.node2index = {}
        self.node_count = 0

    def add_node(self, node_id):
        self.node2index[node_id] = len(self.nodes)
        self.nodes.append(Node(node_id))
        self.node_count += 1

    @staticmethod
    def load_from_network(g):
        nodes = []
        node2index = {}
        node_count = len(g.graph)

        for id_ in g.graph:
            degree = 0
            for to_ids in g.graph[id_].values():
                degree += len(to_ids)

            id_string = str(id_)
            node2index[id_string] = len(nodes)
            node = Node(id_string)
            node.count = degree
            nodes.append(node)

        node_vocab = NodeVocab()
        node_vocab.nodes = nodes
        node_vocab.node2index = node2index
        node_vocab.node_count = node_count
        node_vocab._sort()
        return node_vocab

    @staticmethod
    def load_from_file(fname, available_ids=None):
        '''
            input:
                training_fname:
                    each line: <node_id> <edge_id> ...
                available_ids: set([<node_id>])
                    node_id should be a string
        '''
        with open(fname, 'r') as f:
            nodes = []
            node2index = {}
            node_count = 0

            for line in f:
                tokens = line.strip().split()
                for i, token in enumerate(tokens):
                    if i % 2 == 1:
                        continue
                    if available_ids is not None and token not in available_ids:
                        continue

                    if token not in node2index:
                        node2index[token] = len(nodes)
                        nodes.append(Node(token))

                    nodes[node2index[token]].count += 1
                    node_count += 1

                    if node_count % 10000 == 0:
                        sys.stdout.write("\rReading nodes %d" % node_count)
                        sys.stdout.flush()
            print

            node_vocab = NodeVocab()
            node_vocab.nodes = nodes
            node_vocab.node2index = node2index
            node_vocab.node_count = node_count
            node_vocab._sort()
            return node_vocab

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, key):
        return key in self.node2index

    def __str__(self):
        return ('Node vocab size: %d\nTotal nodes: %d'
                '' % (len(self), self.node_count))

    def __eq__(self, other):
        if not isinstance(other, NodeVocab):
            return False
        if self.nodes != other.nodes:
            return False
        if self.node_count != other.node_count:
            return False
        if self.node2index != other.node2index:
            return False
        return True

    def _sort(self):
        self.nodes.sort(key=lambda x: x.count, reverse=True)
        node2index = {}
        for i, node in enumerate(self.nodes):
            node2index[node.node_id] = i
        self.node2index = node2index

    def count(self):
        return self.node_count

    def to_indices(self, node_ids):
        return [self.node2index[id_] for id_ in node_ids]


class Path(object):
    '''
        Path for PathVocab
    '''
    def __init__(self, path_id, count=0, is_inverse=False):
        self.path_id = path_id
        self.count = count
        self.is_inverse = is_inverse

    def __str__(self):
        return '%s(count:%d, inverse:%s)' % (self.path_id, self.count, self.is_inverse)

    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        if self.path_id != other.path_id:
            return False
        if self.count != other.count:
            return False
        if self.is_inverse != other.is_inverse:
            return False
        return True


class PathVocab(object):
    '''
        a path is a list of edges
    '''

    def __init__(self):
        self.paths = []
        self.path2index = {}
        self.path_count = 0

    def add_path(self, path_id):
        self.path2index[path_id] = len(self.paths)
        self.paths.append(Path(path_id))
        self.path_count += 1

    @staticmethod
    #TODO the order of edges of the path
    def load_from_file(fname, window_size, inverse_mapping=None):
        '''
            input:
                training_fname:
                    each line: <node_id> <edge_id> ...
                window_size: the maximal window size for paths
        '''
        def inverse_edges(edges, inverse_mapping):
            return [inverse_mapping.get(e, e) for e in edges[::-1]]

        with open(fname, 'r') as f:
            paths = []
            path2index = {}
            path_count = 0

            for line in f:
                tokens = line.strip().split()
                tokens = [t for i, t in enumerate(tokens) if i % 2 == 1]
                for w in range(window_size):
                    for i in range(len(tokens)-w):
                        edges = tokens[i:i+1+w]
                        path = ','.join(edges)
                        if path not in path2index:

                            if inverse_mapping is not None:
                                inverse_path = ','.join(inverse_edges(edges, inverse_mapping))
                                if inverse_path not in path2index:
                                    path2index[path] = len(paths)
                                    paths.append(Path(path))
                                else:
                                    path2index[path] = path2index[inverse_path]
                                    paths.append(Path(path, is_inverse=True))
                            else:
                                path2index[path] = len(paths)
                                paths.append(Path(path))

                        paths[path2index[path]].count += 1
                        path_count += 1

                        if path_count % 10000 == 0:
                            sys.stdout.write("\rReading paths %d"
                                             "" % path_count)
                            sys.stdout.flush()
            print

            path_vocab = PathVocab()
            path_vocab.paths = paths
            path_vocab.path2index = path2index
            path_vocab.path_count = path_count
            path_vocab._sort()
            return path_vocab

    def __getitem__(self, i):
        return self.paths[i]

#   def __len__(self):
#       return len(self.paths)

    def distinct_path_count(self):
        return max(self.path2index.values())+1

    def count(self):
        return self.path_count

    def __contains__(self, key):
        return key in self.path2index

    def __str__(self):
        return ('Path vocab size: %d\nTotal paths: %d'
                '' % (len(self), self.node_count))

    def __eq__(self, other):
        if not isinstance(other, PathVocab):
            return False
        if self.paths != other.paths:
            return False
        if self.path_count != other.path_count:
            return False
        if self.path2index != other.path2index:
            return False
        return True

    def _sort(self):
        old_paths = copy.deepcopy(self.paths)
        self.paths.sort(key=lambda x: x.count, reverse=True)
        path2index = {}
        for i, path in enumerate(self.paths):
            if path.is_inverse:
                index = path2index[old_paths[self.path2index[path.path_id]].path_id]
#               print path, self.path2index[path.path_id], index
            else:
                index = i
            path2index[path.path_id] = index
        self.path2index = path2index

    def to_indices(self, path_ids):
        return [self.path2index[id_] for id_ in path_ids]


class EdgeNodePathVocab(PathVocab):
    '''
        a path is: <edge> <node> <edge> ...
    '''
    @staticmethod
    #TODO the order of edges of the path
    def load_from_file(fname, window_size):
        '''
            input:
                training_fname:
                    each line: <node_id> <edge_id> ...
                window_size: the maximal window size for paths
        '''
        with open(fname, 'r') as f:
            paths = []
            path2index = {}
            path_count = 0

            for line in f:
                tokens = line.strip().split()
#               tokens = [t for i, t in enumerate(tokens) if i % 2 == 1]
                for w in range(window_size):
                    for i in range(1, len(tokens)-w*2, 2):
                        path = ','.join(tokens[i:i+1+w*2])
                        if path not in path2index:
                            path2index[path] = len(paths)
                            paths.append(Path(path))

                        paths[path2index[path]].count += 1
                        path_count += 1

                        if path_count % 10000 == 0:
                            sys.stdout.write("\rReading paths %d"
                                             "" % path_count)
                            sys.stdout.flush()
            print

            path_vocab = EdgeNodePathVocab()
            path_vocab.paths = paths
            path_vocab.path2index = path2index
            path_vocab.path_count = path_count
            path_vocab._sort()
            return path_vocab

    def __str__(self):
        return ('EdgeNodePath vocab size: %d\nTotal paths: %d'
                '' % (len(self), self.node_count))

    def __eq__(self, other):
        if not isinstance(other, EdgeNodePathVocab):
            return False
        if self.paths != other.paths:
            return False
        if self.path_count != other.path_count:
            return False
        if self.path2index != other.path2index:
            return False
        return True
