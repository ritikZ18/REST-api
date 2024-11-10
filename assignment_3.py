# -*- coding: utf-8 -*-
"""
SER501 Assignment 3 scaffolding code
created by: Xiangyu Guo
"""
import sys
# =============================================================================


class Graph(object):
    """docstring for Graph"""
    user_defined_vertices = []
    dfs_timer = 0

    def __init__(self, vertices, edges):
        super(Graph, self).__init__()
        n = len(vertices)
        self.matrix = [[0 for x in range(n)] for y in range(n)]
        self.vertices = vertices
        self.edges = edges
        for edge in edges:
            x = vertices.index(edge[0])
            y = vertices.index(edge[1])
            self.matrix[x][y] = edge[2]

    def display(self):
        print(self.vertices)
        for i, v in enumerate(self.vertices):
            print(v, self.matrix[i])

    def transpose(self):
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")

    def in_degree(self):
        print("In degree of the graph:")
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")

    def out_degree(self):
        print("Out degree of the graph:")
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")

    def dfs_on_graph(self):
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")

        # TODO: invoke print_discover_and_finish_time to print out the final
        # result.
        # self.print_discover_and_finish_time()

    def prim(self, root):
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")
        # TODO: invoke print_d_and_pi in each iteration to print out the value.
        # self.print_d_and_pi()

    def bellman_ford(self, source):
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")
        # TODO: invoke print_d_and_pi in each iteration to print out the value.
        # self.print_d_and_pi()

    def dijkstra(self, source):
        # TODO remove the following print message once method is implemented
        print("Not implemented yet!")
        # TODO: invoke print_d_and_pi in each iteration to print out the value.
        # self.print_d_and_pi()

    def print_d_and_pi(self, iteration, d, pi):
        assert((len(d) == len(self.vertices)) and
               (len(pi) == len(self.vertices)))

        print("Iteration: {0}".format(iteration))
        for i, v in enumerate(self.vertices):
            val = 'inf' if d[i] == sys.maxsize else d[i]
            print("Vertex: {0}\td: {1}\tpi: {2}".format(v, val, pi[i]))

    def print_discover_and_finish_time(self, discover, finish):
        assert((len(discover) == len(self.vertices)) and
               (len(finish) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDiscovered: {1}\tFinished: {2}".format(
                    v, discover[i], finish[i]))

    def print_degree(self, degree):
        assert((len(degree) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDegree: {1}".format(v, degree[i]))


def main():
    # Thoroughly test your program and produce useful output.
    # Q1 and Q2
    graph = Graph(['1', '2'], [('1', '2', 1)])
    graph.display()
    graph.transpose()
    graph.display()
    graph.transpose()
    graph.display()
    graph.in_degree()
    graph.out_degree()
    graph.print_d_and_pi(1, [1, sys.maxsize], [2, None])
    graph.print_degree([1, 0])
    graph.print_discover_and_finish_time([0, 2], [1, 3])

    # Q3
    graph = Graph(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
                  [('q', 's', 1),
                   ('s', 'v', 1),
                   ('v', 'w', 1),
                   ('w', 's', 1),
                   ('q', 'w', 1),
                   ('q', 't', 1),
                   ('t', 'x', 1),
                   ('x', 'z', 1),
                   ('z', 'x', 1),
                   ('t', 'y', 1),
                   ('y', 'q', 1),
                   ('r', 'y', 1),
                   ('r', 'u', 1),
                   ('u', 'y', 1)])
    graph.display()
    graph.dfs_on_graph()

    # Q4 - Prim
    graph = Graph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                  [('A', 'H', 6),
                   ('H', 'A', 6),
                   ('A', 'B', 4),
                   ('B', 'A', 4),
                   ('B', 'H', 5),
                   ('H', 'B', 5),
                   ('B', 'C', 9),
                   ('C', 'B', 9),
                   ('G', 'H', 14),
                   ('H', 'G', 14),
                   ('F', 'H', 10),
                   ('H', 'F', 10),
                   ('B', 'E', 2),
                   ('E', 'B', 2),
                   ('G', 'F', 3),
                   ('F', 'G', 3),
                   ('E', 'F', 8),
                   ('F', 'E', 8),
                   ('D', 'E', 15),
                   ('E', 'D', 15)])
    graph.prim('G')

    # Q5
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 7),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('z')

    # Q5 alternate
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 4),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('s')

    # Q6
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('s', 't', 3),
                   ('s', 'y', 5),
                   ('t', 'x', 6),
                   ('t', 'y', 2),
                   ('x', 'z', 2),
                   ('y', 't', 1),
                   ('y', 'x', 4),
                   ('y', 'z', 6),
                   ('z', 's', 3),
                   ('z', 'x', 7)])
    graph.dijkstra('s')


if __name__ == '__main__':
    main()
