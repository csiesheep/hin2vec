#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
from sklearn import cross_validation
from sklearn.svm import LinearSVC
import sys


__author__ = 'sheep'


def main(node_vec_fname, groundtruth_fname):
    '''\
    %prog [options] <node_vec_fname> <groundtruth_fname>

    groundtruth_fname example: res/karate_club_groups.txt
    '''
    node2vec = load_node2vec(node_vec_fname)
    node2classes = load_node2classes(groundtruth_fname)
    exp_classification(node2classes, node2vec)
    return 0

def load_node2vec(fname):
    node2vec = {}
    with open(fname) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue

            line = line.strip()
            tokens = line.split(' ')
            node2vec[tokens[0]] = map(float, tokens[1:])
    return node2vec

def load_node2classes(fname, is_multiclass=True):
    node2classes = {}
    with open(fname) as f:
        for line in f:
            if line.startswith('#'):
                continue

            node, classes = line.strip().split('\t')
            classes = map(int, classes.split(','))
            if is_multiclass:
                node2classes[node] = classes
            else:
                node2classes[node] = classes[0]
    return node2classes

def exp_classification(node2classes, node2vec, seed=None):
    nodes = list(node2classes.keys())

    classes = set()
    for cs in node2classes.values():
        classes.update(cs)

    X = []
    for node_ in nodes:
        X.append(node2vec[node_])

    weights = []
    total_scores = []
    for class_ in sorted(classes):
        y = []
        for node in nodes:
            if class_ in node2classes[node]:
                y.append(1)
            else:
                y.append(0)

        model = LinearSVC()
        print class_, sum(y), len(y)
        scores = cross_validation.cross_val_score(model, X, y, cv=5,
                                                  scoring='f1',
                                                  n_jobs=5)
        print sum(scores)/5
        total_scores.append(sum(scores)/5)
        weights.append(sum(y))

    print total_scores
    print 'macro f1:', sum(total_scores)/len(total_scores)

    micro = 0.0
    for i, s in enumerate(total_scores):
        micro += float(s * weights[i])/sum(weights)
    print 'micro f1:', micro


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

