import os
import numpy as np


class Graph:
    def __init__(self, *args):
        self.density = None
        self.edges = None
        self.path = None
        self.node_weights = None
        self.num_edges = None
        self.num_nodes = None

        if len(args) > 1:
            self.Generate(args[0], args[1])
        else:
            self.ReadFromFile(args[0])

    def Generate(self, numNodes: int, density: float):
        self.num_nodes = numNodes
        self.density = density
        path = "./Generated_Graphs/Generated_Graph_n" + str(self.num_nodes) + "_d" + str(int(density * 10)) + ".txt"
        if not os.path.exists("./Generated_Graphs"):
            os.mkdir("./Generated_Graphs")

        graph_weight_buffer = ""
        graph_edge_buffer = ""
        self.num_edges = 1
        self.node_weights = np.zeros(self.num_nodes)
        self.edges = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            node_weight = np.random.uniform(0.0, 1.0)
            self.node_weights[i] = node_weight
            graph_weight_buffer += str(i) + " " + str(node_weight) + "\n"
            for j in range(i + 1, self.num_nodes):
                uniform_val = np.random.uniform(0.0, 1.0)
                if uniform_val < density:
                    graph_edge_buffer += str(i) + " " + str(j) + "\n"
                    self.num_edges += 1
                    self.edges[i][j] = True

        graph_file = open(path, "w")
        graph_file.write(str(self.num_nodes) + "\n")
        graph_file.write(str(self.num_edges) + "\n")
        graph_file.write(str(graph_weight_buffer))
        graph_file.write(str(graph_edge_buffer))
        graph_file.close()
        self.path = path

    def ReadFromFile(self, path: str):
        graph_file = open(path, "r")
        self.density = int(path[len(path) - 5: len(path) - 4]) / 10
        read = graph_file.readline()[:-1]
        self.num_nodes = int(read)
        read = graph_file.readline()[:-1]
        self.num_edges = int(read)

        self.node_weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            read = graph_file.readline()[:-1]
            index = read.find(" ")
            read2 = read[index:len(read)]
            self.node_weights[i] = float(read2)

        self.edges = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_edges):
            read = graph_file.readline()[:-1]
            index = read.find(" ")
            read1 = read[0:index]
            read2 = read[index:len(read)]
            if read1 != '' and read2 != '':
                a = int(read1)
                b = int(read2)
                self.edges[a][b] = True
                self.edges[b][a] = True

        graph_file.close()
