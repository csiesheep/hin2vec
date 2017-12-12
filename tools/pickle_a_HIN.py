#!/usr/bin/python
# -*- encoding: utf8 -*-

import cPickle
import optparse
import sys

from ds import loader


__author__ = 'sheep'


def main(fname, output_fname):
    '''\
    %prog [options]
    '''
    g = loader.load_a_HIN_from_edge_file(fname)
    cPickle.dump(g, open(output_fname, 'w'))
    return 0


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

