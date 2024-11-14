# -*- coding: utf-8 -*-
"""
SER501 Assignment 3 scaffolding code
created by: Xiangyu Guo
edited by: Ritik Zambre --> 🤣
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

    # transpose of matrix
    def transpose(self):

        # [i][j] is swapped by [j][i]

        print("\n--- Q2. Transposed Matrix ---")
        transposed_matrix = [
            [0] * len(self.matrix) for _ in range(len(self.matrix))
        ]  # store new transpose
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                transposed_matrix[j][i] = self.matrix[i][j]  # swap indexes to transpose

        self.matrix = transposed_matrix

        # #transposed done with List compresion (swaping the indices)
        # print('Transposed Matrix M-2')
        # n = len(self.matrix)
        # transposed_matrix = [[self.matrix[j][i] for j in range(n)] for i in range(n)]
        # self.matrix = transposed_matrix

    # in_degree
    def in_degree(self):

        print("\n--- Q1. In-degree graph ---")
        in_degree_track = {
            i: 0 for i in range(len(self.matrix))
        }  # dictionary to track in-degree

        # traverse the matrix to countt the degrees
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[j][i] != 0:  # incoming edge
                    in_degree_track[i] += 1

        for vertex, degree in in_degree_track.items():
            print(
                f"Vertex: {self.vertices[vertex]} Degree: {degree}"
            )  # print in-degree for each vertex

    # out_degree
    def out_degree(self):
        print("\n--- Q1. Out-degree graph---")
        out_degree_track = {i: 0 for i in range(len(self.matrix))}

        # traver #traverse the matrix to countt the degrees
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i][j] != 0:  # outgoing edge
                    out_degree_track[i] += 1

        for vertex, degree in out_degree_track.items():
            print(
                f"Vertex: {self.vertices[vertex]} Degree: {degree}"
            )  # print out-degree for each vertex

    def dfs_on_graph(self):
        # TODO remove the following print message once method is implemented
        time = 0
        discovered = [-1] * len(self.matrix)  # time to discover vertex
        finished = [-1] * len(self.matrix)  # time to finish vertex
        visited = [False] * len(self.matrix)  # visit status

        def dfs_visited(curr_node):
            nonlocal time
            visited[curr_node] = True
            time += 1
            discovered[curr_node] = time
            for next_node in range(len(self.matrix[curr_node])):
                if self.matrix[curr_node][next_node] != 0 and not visited[next_node]:
                    dfs_visited(next_node)  # recursive dfs on adjacent/next_node

            time += 1
            finished[curr_node] = time

        for curr_node in range(len(self.matrix)):
            if not visited[curr_node]:
                dfs_visited(curr_node)

        self.print_discover_and_finish_time(discovered, finished)

    def prim(self, root):
        total_vertice = len(self.matrix)
        d = [float("inf")] * total_vertice  # shortest weight to add in mst
        pi = [None] * total_vertice  # parent vertex
        d[self.vertices.index(root)] = 0  # distance of root vertex to 0

        visited = [False] * total_vertice

        self.print_d_and_pi("Initial", d, pi)

        for iterations in range(total_vertice):

            min_distance = float("inf")
            curr_vertex = -1

            for vertex in range(total_vertice):
                if not visited[vertex] and d[vertex] < min_distance:
                    min_distance = d[vertex]
                    curr_vertex = vertex

            visited[curr_vertex] = True  # curr vertex visited
            self.print_d_and_pi(iterations, d, pi)

            # updating adj vertices distances
            for candidate_vertex in range(total_vertice):
                if (
                    self.matrix[curr_vertex][candidate_vertex] != 0
                    and not visited[candidate_vertex]
                ):
                    weight = self.matrix[curr_vertex][candidate_vertex]
                    if weight < d[candidate_vertex]:
                        d[candidate_vertex] = weight
                        pi[candidate_vertex] = self.vertices[curr_vertex]

        self.print_d_and_pi("Final", d, pi)

    def bellman_ford(self, source):

        total_vertices = len(self.matrix)
        d = [float("inf")] * total_vertices  # shortest distances
        pi = [None] * total_vertices  # parent vertice
        d[self.vertices.index(source)] = 0

        self.print_d_and_pi("Initial", d, pi)

        for i in range(total_vertices - 1):
            for u, v, weight in self.edges:
                u_index = self.vertices.index(u)
                v_index = self.vertices.index(v)

                if (
                    d[u_index] != float("inf") and d[u_index] + weight < d[v_index]
                ):  # checking shorter path
                    d[v_index] = d[u_index] + weight  # update weight
                    pi[v_index] = u  # update parent

            self.print_d_and_pi(i, d, pi)

        self.print_d_and_pi("final", d, pi)

    def dijkstra(self, source):
        total_vertices = len(self.matrix)
        d = [float("inf")] * total_vertices
        pi = [None] * total_vertices
        d[self.vertices.index(source)] = 0
        visited = [False] * total_vertices

        self.print_d_and_pi("Initial", d, pi)

        for iteration in range(total_vertices):
            min_distance = float("inf")
            curr_vertex = -1
            for i in range(total_vertices):
                if not visited[i] and d[i] < min_distance:
                    min_distance = d[i]
                    curr_vertex = i

            # no vertex confdition
            if curr_vertex == -1:
                break

            visited[curr_vertex] = True

            self.print_d_and_pi(iteration, d, pi)

            for candidate_vertex in range(total_vertices):
                if (
                    self.matrix[curr_vertex][candidate_vertex] != 0
                    and not visited[candidate_vertex]
                    and d[curr_vertex] + self.matrix[curr_vertex][candidate_vertex]
                    < d[candidate_vertex]
                ):
                    d[candidate_vertex] = (
                        d[curr_vertex] + self.matrix[curr_vertex][candidate_vertex]
                    )
                    pi[candidate_vertex] = curr_vertex

    # conditions  DO NOT MODIFY <--STARTS HERE-->

    def print_d_and_pi(self, iteration, d, pi):
        assert (len(d) == len(self.vertices)) and (len(pi) == len(self.vertices))

        print("Iteration: {0}".format(iteration))
        for i, v in enumerate(self.vertices):
            val = "inf" if d[i] == sys.maxsize else d[i]
            print("Vertex: {0}\td: {1}\tpi: {2}".format(v, val, pi[i]))

    def print_discover_and_finish_time(self, discover, finish):
        assert (len(discover) == len(self.vertices)) and (
            len(finish) == len(self.vertices)
        )
        for i, v in enumerate(self.vertices):
            print(
                "Vertex: {0}\tDiscovered: {1}\tFinished: {2}".format(
                    v, discover[i], finish[i]
                )
            )

    def print_degree(self, degree):
        assert len(degree) == len(self.vertices)
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDegree: {1}".format(v, degree[i]))


def main():
    # Thoroughly test your program and produce useful output.
    # Q1 and Q2

    graph = Graph(["1", "2"], [("1", "2", 1)])
    print("\n --- Example Matrix  ---\n")
    graph.display()
    graph.transpose()
    graph.display()
    graph.in_degree()
    graph.out_degree()

    graph.print_d_and_pi(1, [1, sys.maxsize], [2, None])

    graph.print_degree([1, 0])

    graph.print_discover_and_finish_time([0, 2], [1, 3])

    # Q3
    print("\n--- Q3. DFS Traversal ---")
    graph = Graph(
        ["q", "r", "s", "t", "u", "v", "w", "x", "y", "z"],
        [
            ("q", "s", 1),
            ("s", "v", 1),
            ("v", "w", 1),
            ("w", "s", 1),
            ("q", "w", 1),
            ("q", "t", 1),
            ("t", "x", 1),
            ("x", "z", 1),
            ("z", "x", 1),
            ("t", "y", 1),
            ("y", "q", 1),
            ("r", "y", 1),
            ("r", "u", 1),
            ("u", "y", 1),
        ],
    )
    # graph.display()
    graph.dfs_on_graph()

    # Q4 - Prim
    print("\n--- Q4. Prim's Algorithm ---")
    graph = Graph(
        ["A", "B", "C", "D", "E", "F", "G", "H"],
        [
            ("A", "H", 6),
            ("H", "A", 6),
            ("A", "B", 4),
            ("B", "A", 4),
            ("B", "H", 5),
            ("H", "B", 5),
            ("B", "C", 9),
            ("C", "B", 9),
            ("G", "H", 14),
            ("H", "G", 14),
            ("F", "H", 10),
            ("H", "F", 10),
            ("B", "E", 2),
            ("E", "B", 2),
            ("G", "F", 3),
            ("F", "G", 3),
            ("E", "F", 8),
            ("F", "E", 8),
            ("D", "E", 15),
            ("E", "D", 15),
        ],
    )
    graph.prim("G")

    # Q5
    print("\n--- Q5. Bellman Ford ---")
    graph = Graph(
        ["s", "t", "x", "y", "z"],
        [
            ("t", "x", 5),
            ("t", "y", 8),
            ("t", "z", -4),
            ("x", "t", -2),
            ("y", "x", -3),
            ("y", "z", 9),
            ("z", "x", 7),
            ("z", "s", 2),
            ("s", "t", 6),
            ("s", "y", 7),
        ],
    )
    graph.bellman_ford("z")

    # Q5 alternate
    print("\n--- Q5. Bellman Ford ALternate ---")
    graph = Graph(
        ["s", "t", "x", "y", "z"],
        [
            ("t", "x", 5),
            ("t", "y", 8),
            ("t", "z", -4),
            ("x", "t", -2),
            ("y", "x", -3),
            ("y", "z", 9),
            ("z", "x", 4),
            ("z", "s", 2),
            ("s", "t", 6),
            ("s", "y", 7),
        ],
    )
    graph.bellman_ford("s")

    # Q6
    print("\n--- Q6. Djikstra's Algorithm ---")
    graph = Graph(
        ["s", "t", "x", "y", "z"],
        [
            ("s", "t", 3),
            ("s", "y", 5),
            ("t", "x", 6),
            ("t", "y", 2),
            ("x", "z", 2),
            ("y", "t", 1),
            ("y", "x", 4),
            ("y", "z", 6),
            ("z", "s", 3),
            ("z", "x", 7),
        ],
    )
    graph.dijkstra("s")

    # conditions  DO NOT MODIFY <-- ENDS HERE -->


if __name__ == "__main__":
    main()
