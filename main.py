import os

from Graph import Graph
from MWISPMIP import MWISPMIP


def solve_mips(path):
    graph_dirs = os.listdir(path=path)

    for graph_dir in graph_dirs:
        graph = Graph(path+"/"+graph_dir)
        mip = MWISPMIP(graph=graph)
        mip.Solve()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solve_mips("/home/furknn/Desktop/Combinatorial Optimization/Optimization-Term-Project-1-Python/Generated_Graphs")
