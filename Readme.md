
# Assignment 3 : Graph Algorithm

### SER 501 : Advance Data Structure and Algorithm

Introduction : 
The Assignment implements graph Algorithm on Directed Graphs represented by Adjanceny Matrix.

File edited : assignment_3.py

Expected Objectives : 
1. Calculating in-degrees and out-degrees of vertices.
2. Finding Transpose of Directed Graph.
3. Performing DFS and classifying Edges.
4. Implementing Prim's Algorithm for Minimum Spanning Tree (MST).
5. Finding shortest path using Bellman Ford Algorithm. 
6. Using Dijkstra's Algorithm find shortest path.


#### Problems and  Solutions :  

###### Q1 : In-Degrees and Out-Degree Calculations 
-->  Computing in-degree and out-degree of each vertex in directed graph.
*  Functions :  in_degrees()  , out_degrees()
* Implementation : 
    * in_degrees() iterates over columns to count incoming edges for each vertex.
    * out_degrees() iterates over columns to count outgoing edges for each vertex.
    * print_degree() is used by both functions to give output.

* Complexiety :
    * Calculating in_degrees()    => O(V^2)   [iterating over verticews and edges]
    * Calculating out_degrees()   => O(V^2)   [traversing through Adjacency matrix]



###### Q2 : Graph Transpose
--> Reverse all edges in graph to get Transposed graph
* Function : transpose()
* Implementation : 
    * Funtions swaps adjacency matrix rows and columns to reverse edge direction
    * Updates original matrix with transposed version.

* Complexiety : 
    * Finding Transpose of Matrix  => O(V^2)  [iterating over all edges]


###### Q3 : Depth First Search (DFS)
--> Implemented DFS to search graph, and record discover and finsih time for each vertex.
* Function : dfs_on_graph()
* Implementation : 
    * Recusrsive Function dfs_visited() performs dfs and record respective time in specified order
* Edge classification : 
    * Tree Edges connects vertices in DFS tree.
    * Back Edges link vertex to ancestor.
    * Forward Edges connect vertex to descendant in dfs tree.
    * Cross Edges link two vertices without ancestrsl relationship.
* Complexiety : 
    * Every vertex(V) and edge(E) explored once => O(V+E)


###### Q4 :  Prim's Algorithm for MST
--> Computing MST from root verter using linear search..
* Function : prim(root)
* Implementation : 
    * Linear Search is used to find min weight edge to add to MST
    * print_d_and_pi() method gives outpput MST state after each iteration
* Complexiety :
    * Starting with single vertex(V) and adding each edge(E) at a time => O(ElogV)
    
###### Q5 : Bellman Ford Algorithm 
--> Bellman ford Algorithm implementation to find shortest path from soure and  stability using different edge weights 
* Function : bellman_ford(source)
* Implementation : 
    * Relax edges (V-1)times for each vertex.
    * Demonstrated the algorithm in normal and alternate edge weight scenario as well.
    * Output using print_d_and_pi() after each pass.
* Complexity : 
    * Each edge(E) is relaxed for each vertex(V)  => O(VxE)


###### Q6 : Dikstra's Algorithm 
--> Calculate shortest path from source vertex using Dijkstra's 
* Function : dijkstra(source)
* Implementation : 
    * Dijkstra updates distances and predecesors iteratively based on min edge weights.
    * print_d_and_pi() for output having path information after each iteration.
* Complexiety :
    *  Linear Search for smallest distance vertex => O(V^2)