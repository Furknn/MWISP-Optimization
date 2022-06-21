import numpy.random

from Graph import Graph


class Chromosome:
    def __init__(self, string: list[int], graph: Graph):
        self.string = string
        self.feasibilty = self.CheckFeasibility(graph)
        self.objective_val = self.EvalFunc(graph) if self.feasibilty else 0.0

    def CheckFeasibility(self, graph: Graph):
        for i in range(len(self.string)):
            for j in range(i + 1, len(self.string)):
                if graph.edges[i][j] == 1:
                    x_i = int(self.string[i])
                    x_j = int(self.string[j])
                    if not x_i + x_j <= 1:
                        return False
        return True

    def EvalFunc(self, graph: Graph):
        chromosome_eval = 0.0
        for i in range(len(self.string)):
            x = int(self.string[i])
            chromosome_eval += x * graph.node_weights[i]
        return chromosome_eval


class MWISPGenetic:
    best_chromosome = None
    current_population = list[Chromosome]

    def __init__(self, graph: Graph, numOfGen: int, popSize: int, crossProb: float):
        self.graph = graph
        self.number_of_generations = numOfGen
        self.population_size = popSize
        self.crossover_prob = crossProb
        self.mutation_prob = 1 / graph.num_nodes

    def Solve(self):
        self.current_population = []
        self.GenerateInitialPopulation()
        self.best_chromosome = self.BestFromPopulation()
        for generation in range(self.number_of_generations):
            parents = self.BinaryTournamentSelection()
            offsprings = self.UniformCrossover(parents)
            self.current_population = self.Mutate(offsprings)
            new_best_chromosome = self.BestFromPopulation()
            if new_best_chromosome.objective_val > self.best_chromosome.objective_val:
                self.best_chromosome = new_best_chromosome
            # Generation log

    def GenerateInitialPopulation(self):
        while len(self.current_population) < self.population_size:
            chromosome_str = [0] * self.graph.num_nodes
            chromosome_str[numpy.random.randint(0, self.graph.num_nodes)] = 1
            chromosome = Chromosome(chromosome_str, self.graph)
            if chromosome.feasibilty:
                self.current_population.append(chromosome)

    def BestFromPopulation(self):
        best = self.current_population[0]
        for chromosome in self.current_population:
            if chromosome.objective_val > best.objective_val:
                best = chromosome
        return best

    def BinaryTournamentSelection(self):
        parents = []
        while len(parents) < self.population_size:
            contender1 = self.current_population[numpy.random.randint(0, self.population_size)]
            contender2 = self.current_population[numpy.random.randint(0, self.population_size)]
            parents.append(contender1 if contender1.objective_val > contender2.objective_val else contender2)
        return parents

    def UniformCrossover(self, parents):
        offsprings = []
        while len(offsprings) < self.population_size:
            parent1 = parents[numpy.random.randint(0, self.population_size)]
            parent2 = parents[numpy.random.randint(0, self.population_size)]
            offspring_str = []
            for i in range(len(parent1.string)):
                offspring_str.append(
                    parent1.string[i] if numpy.random.uniform(0, 1) < self.crossover_prob else parent2.string[i])
            offspring = Chromosome(offspring_str, self.graph)

            # Repair ?

            offsprings.append(offspring)

        return offsprings

    def Mutate(self, offsprings):
        mutated_offsprings = []
        for offspring in offsprings:
            mutated_offspring = offspring
            dice = numpy.random.uniform(0, 1)
            if dice < self.mutation_prob:
                gene = numpy.random.randint(0, self.graph.num_nodes)
                mutated_offspring.string[gene] = 0 if mutated_offspring.string[gene] == 1 else 1
                mutated_offsprings.append(Chromosome(mutated_offspring.string, self.graph))
            else:
                mutated_offsprings.append(mutated_offspring)
        return mutated_offsprings
