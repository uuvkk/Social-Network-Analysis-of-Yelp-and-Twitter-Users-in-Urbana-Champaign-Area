import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

def draw_graph(graph):
    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # There are graph layouts like shell, spring, spectral and random.
    # Shell layout usually looks better, so we're choosing it.
    # I will show some examples later of other layouts
    graph_pos = nx.shell_layout(G)

    # draw nodes, edges and labels
    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    # show graph
    plt.savefig("/home/dtao2/Dropbox/graph.png")
    plt.show()


if __name__ == "__main__":
    # draw example
    # graph is a list of tuples of nodes. Each tuple defining the
    # connection between 2 nodes

    
    # graph = [(20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 20)]

    # draw_graph(graph)

    G = nx.read_adjlist("../output/adj_list.txt")
    
    # print(G.edges())

    graph_pos = nx.shell_layout(G)
    
    nx.draw_networkx_nodes(G, graph_pos, node_size=100, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    plt.savefig("/home/dtao2/Dropbox/graph.png")
    plt.show()
