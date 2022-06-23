import os

from Graph import Graph
from MWISPGenetic import MWISPGenetic
from MWISPGreedy import MWISPGreedy
from MWISPMIP import MWISPMIP


def solve_mips(path):
    # Solve all graphs under folder using mip solver
    # can be used like: solve_mips("/.Generated_Graphs")
    graph_dirs = os.listdir(path=path)

    for graph_dir in graph_dirs:
        graph = Graph(path + "/" + graph_dir)
        mip = MWISPMIP(graph=graph)
        mip.Solve()


def solve_greedy(path):
    # Solve all graphs under folder using greedy solver
    # can be used like: solve_greedy("/.Generated_Graphs")
    graph_dirs = os.listdir(path=path)

    for graph_dir in graph_dirs:
        graph = Graph(path + "/" + graph_dir)
        mip = MWISPGreedy(graph=graph)
        mip.Solve()


def solve_genetic(path, generations: list, popSizes: list):
    # Solve all graphs under folder using genetic solver
    # can be used like: solve_genetic("./Generated_Graphs", [50, 100, 150], [50, 100])
    graph_dirs = os.listdir(path=path)

    for graph_dir in graph_dirs:
        for generation in generations:
            for popSize in popSizes:
                graph = Graph(path + "/" + graph_dir)
                mip = MWISPGenetic(graph=graph, numOfGen=generation, popSize=popSize, crossProb=0.5)
                mip.Solve()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solve_mips("/.Generated_Graphs")
    solve_greedy("/.Generated_Graphs")
    solve_genetic("./Generated_Graphs", [50, 100, 150], [50, 100])
