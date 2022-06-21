import os
import time
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from Graph import Graph


class MWISPMIP:
    def __init__(self, graph: Graph):
        self.final_solution = ""
        self.time_limit = 3600.0
        self.graph = graph

    def Solve(self):
        model = gp.Model("MIP")
        timestamp = int(time.time())
        solutions_directory = "./MIP_Solutions"
        solution_directory = "./MIP_Solutions/MIP_Solution_" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_" + str(timestamp)

        if not os.path.exists(solutions_directory):
            os.mkdir(solutions_directory)

        if not os.path.exists(solution_directory):
            os.mkdir(solution_directory)

        model.setParam(GRB.Param.TimeLimit, self.time_limit)
        model.setParam("LogFile",
                       solution_directory + "/MWISP_n" + str(self.graph.num_nodes) + "_d" + str(
                           int(self.graph.density * 10)) + "_" + str(timestamp) + ".log")

        X = []
        for i in range(self.graph.num_nodes):
            X.append(model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "x_" + str(i)))

        model.setObjective(sum([X[i] * self.graph.node_weights[i] for i in range(len(X))]), GRB.MAXIMIZE)

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if self.graph.edges[i][j] == 1:
                    model.addConstr(X[i] + X[j] <= 1, "c" + str(i) + str(j))

        model.optimize()

        for i in X:
            self.final_solution += str(int(i.X))

        solution_file = open(solution_directory + "/MWISP_n" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_" + str(timestamp) + "_final_solution.txt", "w")
        solution_file.write(self.final_solution)
        solution_file.close()
