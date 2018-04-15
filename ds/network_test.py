#!/usr/bin/python
# -*- encoding: utf8 -*-

import copy
import unittest

import network


__author__ = "sheep"


class RandomWalkTest(unittest.TestCase):

    def setUp(self):
        g = network.HIN()

        g.add_edge('P1', 'P', 'P2', 'P', ('P-I', 'I-P'), weight=2)
        g.add_edge('P1', 'P', 'P3', 'P', ('P-I', 'I-P'), weight=1)
        g.add_edge('P1', 'P', 'P4', 'P', ('P-I', 'I-P'), weight=1)
        g.add_edge('P2', 'P', 'P1', 'P', ('P-I', 'I-P'), weight=2)
        g.add_edge('P2', 'P', 'P4', 'P', ('P-I', 'I-P'), weight=1)
        g.add_edge('P3', 'P', 'P1', 'P', ('P-I', 'I-P'), weight=1)
        g.add_edge('P4', 'P', 'P1', 'P', ('P-I', 'I-P'), weight=1)
        g.add_edge('P4', 'P', 'P2', 'P', ('P-I', 'I-P'), weight=1)

        g.add_edge('P1', 'P', 'P2', 'P', ('P-P', '-P-P'), weight=1)
        g.add_edge('P2', 'P', 'P1', 'P', ('P-P', '-P-P'), weight=1)
        g.add_edge('P2', 'P', 'P3', 'P', ('P-P', '-P-P'), weight=1)
        g.add_edge('P3', 'P', 'P2', 'P', ('P-P', '-P-P'), weight=1)

        g.add_edge('P1', 'P', 'P4', 'P', ('P-P', 'P-P'), weight=2)
        g.add_edge('P1', 'P', 'P3', 'P', ('P-P', 'P-P'), weight=1)
        g.add_edge('P2', 'P', 'P4', 'P', ('P-P', 'P-P'), weight=1)

        self.g = g

    def testRandomWalks(self):
        actual = list(self.g.random_walks(2, 3, seed=1))
        expected = [
            [0, 0, 1, 2, 3],
            [1, 0, 3, 0, 0],
            [2, 0, 0, 0, 2],
            [3, 0, 1, 0, 3],
            [0, 0, 1, 0, 0],
            [1, 2, 3, 0, 0],
            [2, 1, 1, 0, 0],
            [3, 0, 0, 0, 3]
        ]
        self.assertEquals(expected, actual)


class HINTest(unittest.TestCase):

    def testInit(self):
        expected = network.HIN()
        expected.node2id = {
            'P1': 0,
            'P2': 1,
            'P3': 2,
            'I1': 3,
            'I2': 4,
            'I3': 5,
        }
        expected.graph = {
            0: {
                1: {0:1},
                2: {0:1},
                3: {1:1},
                4: {1:1},
                5: {1:1},
            },
            1: {
                0: {2:1},
                2: {0:1},
                3: {1:1},
                4: {1:1},
            },
            2: {
                0: {2:1},
                1: {2:1},
                5: {1:1},
            },
            3 : {
                0: {3:1},
                1: {3:1},
            },
            4 : {
                0: {3:1},
                1: {3:1},
            },
            5 : {
                0: {3:1},
                2: {3:1},
            },
        }
        expected.edge_class2id = {
            'P-P': 0,
            'P-I': 1,
            '-P-P': 2,
            'I-P': 3,
        }
        expected.class_nodes = {
            'P': set([0, 1, 2]),
            'I': set([3, 4, 5]),
        }
        expected.edge_class_id_available_node_class = {
            0: ('P', 'P'),
            1: ('P', 'I'),
            2: ('P', 'P'),
            3: ('I', 'P'),
        }

        g = network.HIN()
        g.add_edge('P1', 'P', 'P2', 'P', 'P-P')
        g.add_edge('P1', 'P', 'P3', 'P', 'P-P')
        g.add_edge('P1', 'P', 'I1', 'I', 'P-I')
        g.add_edge('P1', 'P', 'I2', 'I', 'P-I')
        g.add_edge('P1', 'P', 'I3', 'I', 'P-I')

        g.add_edge('P2', 'P', 'P1', 'P', '-P-P')
        g.add_edge('P2', 'P', 'P3', 'P', 'P-P')
        g.add_edge('P2', 'P', 'I1', 'I', 'P-I')
        g.add_edge('P2', 'P', 'I2', 'I', 'P-I')

        g.add_edge('P3', 'P', 'P1', 'P', '-P-P')
        g.add_edge('P3', 'P', 'P2', 'P', '-P-P')
        g.add_edge('P3', 'P', 'I3', 'I', 'P-I')

        g.add_edge('I1', 'I', 'P1', 'P', 'I-P')
        g.add_edge('I1', 'I', 'P2', 'P', 'I-P')

        g.add_edge('I2', 'I', 'P1', 'P', 'I-P')
        g.add_edge('I2', 'I', 'P2', 'P', 'I-P')

        g.add_edge('I3', 'I', 'P1', 'P', 'I-P')
        g.add_edge('I3', 'I', 'P3', 'P', 'I-P')

        self.assertEquals(expected, g)


class ToHomogeneousNetwork(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('P1', 'P', 'P2', 'P', 'P->P', weight=2)
        g.add_edge('P2', 'P', 'P1', 'P', 'P<-P', weight=2)
        g.add_edge('P1', 'P', 'A1', 'A', 'P-A', weight=1)
        g.add_edge('A1', 'A', 'P1', 'P', 'P-A', weight=1)
        g.add_edge('P1', 'P', 'A1', 'A', 'P-A2', weight=2)
        g.add_edge('A1', 'A', 'P1', 'P', 'P-A2', weight=2)

        expected = network.HIN()
        expected.add_edge('P1', '', 'P2', '', '', weight=2)
        expected.add_edge('P2', '', 'P1', '', '', weight=2)
        expected.add_edge('P1', '', 'A1', '', '', weight=3)
        expected.add_edge('A1', '', 'P1', '', '', weight=3)
        expected.edge_class_id_available_node_class = {}

        g.to_homogeneous_network()
        self.assertEquals(expected, g)


class ToWeightedEdgeListTest(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('P1', 'P', 'P2', 'P', 'P->P', weight=2)
        g.add_edge('P2', 'P', 'P1', 'P', 'P<-P', weight=2)
        g.add_edge('P1', 'P', 'A1', 'A', 'P-A', weight=1)
        g.add_edge('A1', 'A', 'P1', 'P', 'P-A', weight=1)

        edges = g.to_weighted_edge_list()
        edges = sorted(edges)
        expected_edges = [
            ('A1', 'P1', 1),
            ('P1', 'A1', 1),
            ('P1', 'P2', 2),
            ('P2', 'P1', 2),
        ]
        self.assertEquals(expected_edges, edges)


class GetEdgeClassInverMapplingTest(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('P1', 'P', 'P2', 'P', 'P-P', weight=2)
        g.add_edge('P2', 'P', 'P1', 'P', 'P-P', weight=2)
        g.add_edge('P1', 'P', 'P3', 'P', 'P-P', weight=2)
        g.add_edge('P3', 'P', 'P1', 'P', 'P-P', weight=2)
        g.add_edge('P1', 'P', 'A1', 'A', 'P-A', weight=1)
        g.add_edge('A1', 'A', 'P1', 'P', 'A-P', weight=1)

        expected = {'1': '2', '2': '1'}
        actual = g.get_edge_class_inverse_mappling()
        self.assertEquals(expected, actual)

    def testDirected(self):
        g = network.HIN()
        g.add_edge('P1', 'P', 'P2', 'P', 'P->P', weight=2)
        g.add_edge('P2', 'P', 'P1', 'P', 'P<-P', weight=2)
        g.add_edge('P1', 'P', 'P3', 'P', 'P->P', weight=2)
        g.add_edge('P3', 'P', 'P1', 'P', 'P<-P', weight=2)
        g.add_edge('P1', 'P', 'A1', 'A', 'P-A', weight=1)
        g.add_edge('A1', 'A', 'P1', 'P', 'A-P', weight=1)

        expected = {'0': '1', '1': '0', '2': '3', '3': '2'}
        actual = g.get_edge_class_inverse_mappling()
        self.assertEquals(expected, actual)


class InKHopNeighborhood(unittest.TestCase):

    def setUp(self):
        g = network.HIN()
        g.add_edge('A', '', 'B', '', '')
        g.add_edge('B', '', 'A', '', '')

        g.add_edge('A', '', 'C', '', '')
        g.add_edge('C', '', 'A', '', '')

        g.add_edge('B', '', 'C', '', '')
        g.add_edge('C', '', 'B', '', '')

        g.add_edge('C', '', 'D', '', '')
        g.add_edge('D', '', 'C', '', '')

        g.add_edge('D', '', 'E', '', '')
        g.add_edge('E', '', 'D', '', '')

        g.add_edge('D', '', 'F', '', '')
        g.add_edge('F', '', 'D', '', '')

        self.g = g

    def testGetKHopNeighbors(self):
        expected = {
            self.g.node2id['B'],
            self.g.node2id['C'],
            self.g.node2id['D'],
        }
        actual = self.g._get_k_hop_neighborhood(0, 2)
        self.assertEquals(expected, actual)

    def testInKHop(self):
        self.assertEquals(True, self.g.in_k_hop_neighborhood(0, 1, 2))
        self.assertEquals(True, self.g.in_k_hop_neighborhood(0, 2, 2))
        self.assertEquals(True, self.g.in_k_hop_neighborhood(0, 3, 2))
        self.assertEquals(False,self.g.in_k_hop_neighborhood(0, 4, 2))
        self.assertEquals(False,self.g.in_k_hop_neighborhood(1, 4, 2))

        expected = {
            2: {
                0: set([1, 2, 3]),
                1: set([0, 2, 3]),
            }
        }
        self.assertEquals(expected, self.g.k_hop_neighbors)

    def testGetCandidates(self):
        g = network.HIN()
        g.add_edge('A', '', 'B', 'x', '')
        g.add_edge('B', 'x', 'A', '', '')

        g.add_edge('A', '', 'C', 'x', '')
        g.add_edge('C', 'x', 'A', '', '')

        g.add_edge('B', 'x', 'C', 'x', '')
        g.add_edge('C', 'x', 'B', 'x', '')

        g.add_edge('C', 'x', 'D', 'x', '')
        g.add_edge('D', 'x', 'C', 'x', '')

        g.add_edge('C', 'x', 'E', 'y', '')
        g.add_edge('E', 'y', 'C', 'x', '')

        g.add_edge('D', 'x', 'F', 'x', '')
        g.add_edge('F', 'x', 'D', 'x', '')

        expected = {
            g.node2id['D'],
        }
        actual = g.get_candidates(0, 2, 'x')
        self.assertEquals(expected, actual)


class GetShortestDistance(unittest.TestCase):

    def setUp(self):
        g = network.HIN()
        g.add_edge('A', '', 'B', '', '')
        g.add_edge('B', '', 'A', '', '')

        g.add_edge('A', '', 'C', '', '')
        g.add_edge('C', '', 'A', '', '')

        g.add_edge('B', '', 'C', '', '')
        g.add_edge('C', '', 'B', '', '')

        g.add_edge('C', '', 'D', '', '')
        g.add_edge('D', '', 'C', '', '')

        g.add_edge('D', '', 'E', '', '')
        g.add_edge('E', '', 'D', '', '')

        g.add_edge('D', '', 'F', '', '')
        g.add_edge('F', '', 'D', '', '')

        g.add_edge('X', '', 'Y', '', '')
        g.add_edge('Y', '', 'X', '', '')

        self.g = g

    def testSimple(self):
        actual = self.g.get_shortest_distance(0, 1) # A, B
        self.assertEquals(1, actual)

        actual = self.g.get_shortest_distance(1, 0) # B, A
        self.assertEquals(1, actual)

        actual = self.g.get_shortest_distance(0, 4) # A, E
        self.assertEquals(3, actual)

        actual = self.g.get_shortest_distance(4, 0) # E, A
        self.assertEquals(3, actual)

        actual = self.g.get_shortest_distance(0, 7) # A, X
        self.assertEquals(None, actual)

        actual = self.g.get_shortest_distance(7, 0) # X, A
        self.assertEquals(None, actual)

    def testWithMax(self):
        actual = self.g.get_shortest_distance(0, 6, max_=2) # A, F
        self.assertEquals(None, actual)

        actual = self.g.get_shortest_distance(6, 0, max_=2) # F, A
        self.assertEquals(None, actual)


class AvailabeNodeClassOfEdgeClassTest(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('A1', 'A', 'B1', 'B', 'A-B')
        g.add_edge('B1', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B2', 'B', 'A-B')
        g.add_edge('B2', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A2', 'A', 'C1', 'C', 'A-C')
        g.add_edge('C1', 'C', 'A2', 'A', 'C-A')

        expected = {
            0: ('A', 'B'), #A-B
            1: ('B', 'A'), #B-A
            2: ('A', 'C'), #A-C
            3: ('C', 'A'), #C-A
        }
        self.assertEquals(expected, g.edge_class_id_available_node_class)


class RandomRemoveEdge(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('A1', 'A', 'B1', 'B', 'A-B')
        g.add_edge('B1', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B2', 'B', 'A-B')
        g.add_edge('B2', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B3', 'B', 'A-B')
        g.add_edge('B3', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A2', 'A', 'C1', 'C', 'A-C')
        g.add_edge('C1', 'C', 'A2', 'A', 'C-A')

        expected_g = copy.deepcopy(g)
        expected_g.graph[0].pop(3)
        expected_g.graph[3].pop(0)
        expected_edges = [(0, 3)]
        actual = g.random_remove_edges('A-B', ratio=0.5, seed=1)
        self.assertEquals(expected_edges, actual)
        self.assertEquals(expected_g, g)


class RandomSelectNegEdges(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('A1', 'A', 'B1', 'B', 'A-B')
        g.add_edge('B1', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B2', 'B', 'A-B')
        g.add_edge('B2', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B3', 'B', 'A-B')
        g.add_edge('B3', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A2', 'A', 'C1', 'C', 'A-C')
        g.add_edge('C1', 'C', 'A2', 'A', 'C-A')

        expected = set([(4, 1), (4, 3)])
        actual = g.random_select_neg_edges('A-B', 2, seed=1)
        self.assertEquals(expected, actual)


class ToEdgeClassIdString(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('A1', 'A', 'B1', 'B', 'A-B')
        g.add_edge('B1', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B2', 'B', 'A-B')
        g.add_edge('B2', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A1', 'A', 'B3', 'B', 'A-B')
        g.add_edge('B3', 'B', 'A1', 'A', 'B-A')

        g.add_edge('A2', 'A', 'C1', 'C', 'A-C')
        g.add_edge('C1', 'C', 'A2', 'A', 'C-A')

        edge_classes = ['A-B', 'B-A', 'A-C']
        expected = '0,1,2'
        actual = g.to_edge_class_id_string(edge_classes)
        self.assertEquals(expected, actual)


class GenerateTestSet(unittest.TestCase):

    def testSimple(self):
        g = network.HIN()
        g.add_edge('U1', 'U', 'U2', 'U', 'U-U', weight=1)
        g.add_edge('U2', 'U', 'U1', 'U', 'U-U', weight=1)

        g.add_edge('U2', 'U', 'U3', 'U', 'U-U', weight=1)
        g.add_edge('U3', 'U', 'U2', 'U', 'U-U', weight=1)

        g.add_edge('U1', 'U', 'U4', 'U', 'U-U', weight=2)
        g.add_edge('U4', 'U', 'U1', 'U', 'U-U', weight=2)

        g.add_edge('U4', 'U', 'U3', 'U', 'U-U', weight=1)
        g.add_edge('U3', 'U', 'U4', 'U', 'U-U', weight=1)

        g.add_edge('U4', 'U', 'U5', 'U', 'U-U', weight=3)
        g.add_edge('U5', 'U', 'U4', 'U', 'U-U', weight=3)

        g.add_edge('U1', 'U', 'B1', 'B', 'U-B', weight=1)
        g.add_edge('B1', 'B', 'U1', 'U', 'B-U', weight=1)

        g.add_edge('U2', 'U', 'B1', 'B', 'U-B', weight=1)
        g.add_edge('B1', 'B', 'U2', 'U', 'B-U', weight=1)

        expected = [(0, 4, 0), (3, 1, 0), (4, 2, 0), (3, 0, 1)]
        actual = g.generate_test_set([0], 4, seed=1)
        self.assertEquals(expected, actual)

        expected = [(0, 5, 1), (3, 5, 0), (2, 5, 0), (3, 5, 0)]
        actual = g.generate_test_set([1], 4, seed=1)
        self.assertEquals(expected, actual)

        expected = [(0, 4, 1), (3, 1, 1), (4, 2, 1), (3, 0, 0)]
        actual = g.generate_test_set([0, 0], 4, seed=1)
        self.assertEquals(expected, actual)

        expected = [(4, 5, 0), (0, 5, 1), (4, 5, 0), (3, 5, 1)]
        actual = g.generate_test_set([0, 1], 4, seed=2)
        self.assertEquals(expected, actual)


if __name__ == '__main__':
    unittest.main()

