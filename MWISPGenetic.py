import math
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy.random

from Graph import Graph


class Gene:
    def __init__(self, string: list[int], graph: Graph):
        self.string = string
        self.feasible = self.CheckFeasibility(graph)
        self.fitness = self.EvalFunc(graph) if self.feasible else 0.0

    def CheckFeasibility(self, graph: Graph):
        # Feasibility checking function
        # Only looks at indexes with value of 1 in the string
        indexes_of_nodes = [i for i in range(len(self.string)) if self.string[i] == 1]
        for i in range(len(indexes_of_nodes)):
            for j in range(i + 1, len(indexes_of_nodes)):
                if graph.edges[indexes_of_nodes[i]][indexes_of_nodes[j]] == 1:
                    x_i = int(self.string[indexes_of_nodes[i]])
                    x_j = int(self.string[indexes_of_nodes[j]])
                    if not x_i + x_j <= 1:
                        self.feasible = False
                        return self.feasible
        self.feasible = True
        return self.feasible

    def EvalFunc(self, graph: Graph):
        # Evaluation function (fitness)
        # Also checks for feasibility, if unfeasible sets fitness to 0.0
        gene_eval = 0.0
        if self.CheckFeasibility(graph):
            for i in range(len(self.string)):
                x = int(self.string[i])
                gene_eval += x * graph.node_weights[i]
            self.fitness = gene_eval
        else:
            self.fitness = 0.0

        return self.fitness

    def Repair(self, graph):
        # Repair function
        # Generates a list from indices of 1's
        # Picks a random index from this list
        # Flip's it to zero, increases the change counter
        # Repeats until Gene becomes feasible
        # Then Greedily picks change_counter number of nodes and make's them 1
        indexes_of_nodes = [i for i in range(len(self.string)) if self.string[i] == 1]
        change_counter = 0
        while len(indexes_of_nodes) > 0 and not self.CheckFeasibility(graph):
            rand_index = numpy.random.randint(0, len(indexes_of_nodes))
            self.string[indexes_of_nodes[rand_index]] = 0
            indexes_of_nodes.pop(rand_index)
            change_counter += 0

        sorted_node_weights = sorted(
            {i: v for i, v in enumerate(graph.node_weights) if not indexes_of_nodes.__contains__(i)}.items(),
            key=lambda x: x[1], reverse=True)
        while change_counter > 0:
            self.string[sorted_node_weights[0][0]] = 1
            if self.CheckFeasibility(graph):
                change_counter -= 0
                sorted_node_weights.pop(0)
            else:
                self.string[sorted_node_weights[0][0]] = 0
                sorted_node_weights.pop(0)

        self.EvalFunc(graph)


class MWISPGenetic:
    def __init__(self, graph: Graph, numOfGen: int, popSize: int, crossProb: float):
        # Initialized properties
        self.graph = graph
        self.number_of_generations = numOfGen
        self.population_size = popSize
        self.crossover_prob = crossProb

        # Generated properties
        self.current_population = []
        self.best_gene = None
        self.new_best_gene = None
        self.mutation_prob = 1 / graph.num_nodes

        # Log and report properties
        self.solution_start_time = None
        self.mutation_count = 0
        self.initial_pop_time = None
        self.generation_start_time = None
        self.crossover_count = 0
        self.log_output_buffer = ""
        self.generation_times = []
        self.generation_std = []
        self.generation_best_fitnesses = []
        self.generation_means = []
        self.best_fitnesses = []
        self.log_file = None

    def Solve(self):
        # Main method for running solution
        # Generates an initial population
        # Pick's parents using Binary Tournament Selection
        # Crossovers using Uniform Crossover with given crossover probability
        # Mutates and Repairs offsprings
        self.solution_start_time = time.time()
        self.current_population = []
        self.GenerateInitialPopulation()
        self.best_gene = self.BestFromPopulation()
        for generation in range(self.number_of_generations):
            self.generation_start_time = time.time()
            parents = self.BinaryTournamentSelection()
            offsprings = self.UniformCrossover(parents)
            self.current_population = self.Mutate(offsprings)
            self.new_best_gene = self.BestFromPopulation()
            if self.new_best_gene.fitness > self.best_gene.fitness:
                self.best_gene = self.new_best_gene
            self.LogGeneration(generation + 1)
        self.LogSolution()
        self.Report()

    def GenerateInitialPopulation(self):
        # Greedily picks 1 unique node for each member of the initial population
        self.initial_pop_time = time.time()
        sorted_node_weights = sorted({i: v for i, v in enumerate(self.graph.node_weights)}.items(), key=lambda x: x[1],
                                     reverse=True)
        while len(self.current_population) < self.population_size:
            gene_str = [0] * self.graph.num_nodes
            gene_str[sorted_node_weights[0][0]] = 1
            sorted_node_weights.pop(0)
            gene = Gene(gene_str, graph=self.graph)
            self.current_population.append(gene)

        self.InitialPopLog()

    def BestFromPopulation(self):
        # Picks the best gene from current population
        best = self.current_population[0]
        for gene in self.current_population:
            if gene.fitness > best.fitness:
                best = gene
        return best

    def BinaryTournamentSelection(self):
        # Generates a parent population by
        # picking the one with the best fitness from 2 random parents
        parents = []
        while len(parents) < self.population_size:
            contender1 = self.current_population[numpy.random.randint(0, self.population_size)]
            contender2 = self.current_population[numpy.random.randint(0, self.population_size)]
            parents.append(contender1 if contender1.fitness > contender2.fitness else contender2)
        return parents

    def UniformCrossover(self, parents):
        # Crossovers parents using uniform crossover
        # Picks 2 parent from parent population
        # Copies their strings to 2 children separately
        # For each node in the graph
        # Generate a random value from uniform distribution
        # if this value is smaller than crossover probability
        # swap values of nodes in children
        # add both children to population
        offsprings = [self.best_gene, self.best_gene]
        self.crossover_count = 0
        while len(offsprings) < self.population_size:
            parent1 = parents[numpy.random.randint(0, self.population_size)]
            parent2 = parents[numpy.random.randint(0, self.population_size)]
            child1 = parent1.string.copy()
            child2 = parent2.string.copy()
            for i in range(len(parent1.string)):
                dice = numpy.random.uniform(0, 1)
                if dice < self.crossover_prob:
                    self.crossover_count += 1
                    child1[i] = parent1.string[i]
                    child2[i] = parent2.string[i]

            offsprings.append(Gene(child1, self.graph))
            offsprings.append(Gene(child2, self.graph))

        return offsprings

    def Mutate(self, offsprings):
        # Iterates over each member's nodes
        # Generate a random value from uniform distribution
        # Flip node, if value is smaller than mutation probability
        # Check feasibility of each member
        # Call the repair function, if unfeasible
        mutated_offsprings = []
        self.mutation_count = 0
        for offspring in offsprings:
            mutated_offspring = offspring
            for i in range(len(mutated_offspring.string)):
                dice = numpy.random.uniform(0, 1)
                if dice < self.mutation_prob:
                    mutated_offspring.string[i] = 0 if mutated_offspring.string[i] == 1 else 1
                    self.mutation_count += 1

            mutated_offspring = Gene(mutated_offspring.string, self.graph)

            if not mutated_offspring.CheckFeasibility(self.graph):
                mutated_offspring.Repair(self.graph)

            mutated_offsprings.append(mutated_offspring)
        return mutated_offsprings

    # --------------- LOGGING AND REPORTING METHODS --------------- #

    def LogSolution(self):
        solution_time = str(time.time() - self.solution_start_time)
        best_fitness = str(self.best_gene.fitness)
        final_solution_string = [str(i) for i in self.best_gene.string]
        final_solution_string = "".join(final_solution_string)

        solution_time_text = "\nSolution Took: " + solution_time + "s"
        best_fitness_text = "Best Fitness: " + best_fitness
        final_solution_string_text = "Solution: " + final_solution_string

        print(solution_time_text)
        print(best_fitness_text)
        print(final_solution_string_text)

        self.log_output_buffer += solution_time_text + "\n"
        self.log_output_buffer += best_fitness_text + "\n"
        self.log_output_buffer += final_solution_string_text + "\n"

    def LogGeneration(self, generation):
        generation_time = time.time() - self.generation_start_time
        best_fitness = self.best_gene.fitness
        generation_best_fitness = self.new_best_gene.fitness
        mean = sum([pop.fitness for pop in self.current_population]) / self.population_size
        std = math.sqrt(sum([(pop.fitness - mean) ** 2 for pop in self.current_population]) / self.population_size)

        self.generation_times.append(generation_time)
        self.best_fitnesses.append(best_fitness)
        self.generation_best_fitnesses.append(generation_best_fitness)
        self.generation_means.append(mean)
        self.generation_std.append(std)

        header = "--------------------- Generation " + str(generation) + " ---------------------"
        generation_took_text = "Generation Took: " + str(generation_time) + "s"
        best_fitness_text = "Best Fitness: " + str(best_fitness)
        generation_best_fitness_text = "Generation Best Fitness: " + str(generation_best_fitness)
        mean_fitness_text = "Mean Fitness: " + str(mean)
        standard_deviation_text = "Standard Deviation: " + str(std)
        crossover_count_text = "Number of Crossovers: " + str(self.crossover_count)
        mutation_count_text = "Number of Mutations: " + str(self.mutation_count)
        footer = "-" * len(header)

        print(header)
        print(generation_took_text)
        print(best_fitness_text)
        print(generation_best_fitness_text)
        print(mean_fitness_text)
        print(standard_deviation_text)
        print(crossover_count_text)
        print(mutation_count_text)
        print(footer)

        self.log_output_buffer += header + "\n"
        self.log_output_buffer += generation_took_text + "\n"
        self.log_output_buffer += best_fitness_text + "\n"
        self.log_output_buffer += generation_best_fitness_text + "\n"
        self.log_output_buffer += mean_fitness_text + "\n"
        self.log_output_buffer += standard_deviation_text + "\n"
        self.log_output_buffer += crossover_count_text + "\n"
        self.log_output_buffer += mutation_count_text + "\n"
        self.log_output_buffer += footer + "\n"

    def InitialPopLog(self):
        number_of_nodes_text = "Number of nodes: " + str(self.graph.num_nodes)
        pop_size_text = "Population size: " + str(self.population_size)
        initial_pop_gen_text = "Generating initial population generation took: " + str(
            int(time.time() - self.initial_pop_time)) + "s "
        number_of_generations_text = "Generations: " + str(self.number_of_generations)

        print(number_of_nodes_text)
        print(pop_size_text)
        print(number_of_generations_text)
        print(initial_pop_gen_text)

        self.log_output_buffer += number_of_nodes_text + "\n"
        self.log_output_buffer += pop_size_text + "\n"
        self.log_output_buffer += number_of_generations_text + "\n"
        self.log_output_buffer += initial_pop_gen_text + "\n"

    def Report(self):
        data = pd.DataFrame({
            "Mean Fitness": self.generation_means,
            "Generation Best Fitness": self.generation_best_fitnesses,
            "Best Fitness": self.best_fitnesses
        })

        sd = pd.DataFrame({
            "Standard Deviation": self.generation_std
        })

        p = sns.lineplot(data=data)

        p.fill_between([i for i in range(self.number_of_generations)],
                       y1=data["Mean Fitness"] - sd["Standard Deviation"],
                       y2=data["Mean Fitness"] + sd["Standard Deviation"], alpha=.5)
        p.set_xlabel("Generation")
        p.set_ylabel("Fitness")

        solutions_directory = "./Genetic_Solutions"
        solution_directory = "./Genetic_Solutions/Genetic_Solution_" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_Gen_" + str(self.number_of_generations) + "_Pop_" + str(
            self.population_size) + "_" + str(int(self.solution_start_time))

        if not os.path.exists(solutions_directory):
            os.mkdir(solutions_directory)

        if not os.path.exists(solution_directory):
            os.mkdir(solution_directory)

        solution_file = open(solution_directory + "/MWISP_n" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_Gen_" + str(self.number_of_generations) + "_Pop_" + str(
            self.population_size) + "_" + str(int(self.solution_start_time)) + "_log.txt", "w")

        solution_file.write(self.log_output_buffer)
        solution_file.close()

        plt.savefig(solution_directory + "/MWISP_n" + str(self.graph.num_nodes) + "_d" + str(
            int(self.graph.density * 10)) + "_Gen_" + str(self.number_of_generations) + "_Pop_" + str(
            self.population_size) + "_" + str(int(self.solution_start_time)) + "_plot.png")
        plt.show()
