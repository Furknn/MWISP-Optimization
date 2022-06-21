import os
import time

import numpy


class MWISPGreedy:
    copy_of_node_weights = []
    solution_string = []

    def __init__(self, graph):
        self.graph = graph

    def Solve(self):
        print("Greedy solver started")

        timestamp = int(time.time())
        solutions_directory = "./Greedy_Solutions"
        solution_directory = "./Greedy_Solutions/Greedy_Solution_" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_" + str(timestamp)

        if not os.path.exists(solutions_directory):
            os.mkdir(solutions_directory)

        if not os.path.exists(solution_directory):
            os.mkdir(solution_directory)

        self.solution_string = [0] * self.graph.num_nodes
        self.copy_of_node_weights = self.graph.node_weights.copy()

        for i in range(self.graph.num_nodes):

            max_node_weight_index = numpy.argmax(self.copy_of_node_weights)
            sol_str = self.solution_string.copy()
            sol_str[max_node_weight_index] = 1
            if self.CheckFeasibility(sol_str):
                self.solution_string = sol_str.copy()
                self.copy_of_node_weights[max_node_weight_index] = 0
                print("iter " + str(i + 1) + " eval: " + str(self.Evaluate(self.solution_string)))
            else:
                self.copy_of_node_weights[max_node_weight_index] = 0

        final_solution_string = [str(i) for i in self.solution_string]
        final_solution_string = "".join(final_solution_string)
        solution_file = open(solution_directory + "/MWISP_n" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_" + str(timestamp) + "_final_solution.txt", "w")
        solution_file.write(final_solution_string + "\n\n")
        solution_file.write("Evaluation: " + str(self.Evaluate(self.solution_string)) + "\n")
        completed_time = int(time.time()) - timestamp
        solution_file.write("Completed in: " + str(completed_time) + "s")
        solution_file.close()

        print("Greedy solver done in" + str(completed_time))

    def CheckFeasibility(self, solution_string):
        for i in range(len(solution_string)):
            for j in range(i + 1, len(solution_string)):
                if self.graph.edges[i][j] == 1:
                    x_i = int(solution_string[i])
                    x_j = int(solution_string[j])
                    if not x_i + x_j <= 1:
                        return False
        return True

    def Evaluate(self, solution_string):
        solution_eval = 0.0
        for i in range(len(solution_string)):
            x = int(solution_string[i])
            solution_eval += x * self.graph.node_weights[i]
        return solution_eval
