import io
import re
import traceback

import networkx as nx
from Bio import Phylo
from Graph import DAG, Node
from NetworkParser import NetworkParser, NetworkParserError
from collections import deque
import dendropy



def add_parallel_edge(parent: Node, child: Node, network: DAG):
    """
    Adds a parallel edge between parent and child in network
                   parent
                     |
                    nd1
                     | \
                     |  |
                     | /
                    nd2
                     |
                   child
    """
    if [parent, child] not in network.edges:
        raise ValueError(
            f"Edge {parent.get_name()}->{child.get_name()} does not exist in network {network.print_adjacency()}")
    inserted_node1, inserted_node2 = Node(), Node()
    network.add_uid_node(inserted_node1)
    network.add_uid_node(inserted_node2)
    network.removeEdge([parent, child])
    network.addEdges([
        [parent, inserted_node2],
        [inserted_node2, inserted_node1],
        [inserted_node1, child]
    ], as_list=True)

    network.addEdges([[inserted_node2, inserted_node1]], as_list=True)


def bfs(network: DAG):
    q = [network.root()[0]]
    visited = set()
    while q:
        cur = q.pop()
        yield cur
        visited.add(cur)

        for child in network.get_children(cur):
            if child not in visited and child not in q:
                q.append(child)


def get_leaf_name_set(network: DAG):
    return set([node.get_name() for node in network.get_leaves()])

def convert_to_networkx(network: DAG):
    G = nx.DiGraph()
    edges = []
    for edge in network.get_edges():
        edges.append((edge[0].get_name(), edge[1].get_name()))
    G.add_edges_from(edges)
    return G
def plot_network(network: DAG):
    import matplotlib.pyplot as plt
    G = convert_to_networkx(network)
    indegrees = G.in_degree()
    edge_colors = ['red' if indegrees[edge[1]] > 1 else 'black' for edge in G.edges()]
    nx.draw_networkx(G, edge_color=edge_colors, with_labels=True)
    plt.savefig("network.png")


def remove_binary_nodes(net: DAG):
    """Modified based on DAG.prune_excess_nodes()"""

    def prune(net: DAG) -> bool:
        root = net.root()[0]
        q = deque([root])
        net_updated = False

        while q:
            cur = q.pop()  # pop right for bfs

            for neighbor in net.get_children(cur):
                current_node: Node = neighbor
                previous_node: Node = cur
                node_removed = False

                # There could be a chain of nodes with in/out degree = 1. Resolve the whole chain before moving on to search more nodes
                while net.in_degree(current_node) == net.out_degree(current_node) == 1:
                    net.remove_edge([previous_node, current_node])

                    previous_node = current_node
                    temp = net.get_children(current_node)[0]
                    net.remove_node(current_node)
                    current_node = temp
                    node_removed = True

                # We need to connect cur to its new successor
                if node_removed:
                    # self.removeEdge([previous_node, current_node])
                    net.add_edges([cur, current_node])
                    current_node.set_parent([cur])
                    net_updated = True

                # Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(current_node)

        return net_updated

    while True:
        update = prune(net)
        if not update:
            break


def print_topology_newick(net: DAG):
    newick = re.sub(r':\[.*?\]', '', net.newick())
    newick = newick.replace(" ", "")
    print(newick)

def is_tree(graph: DAG):
    visited = set()
    stack = [graph.root()[0]]   
    while stack:
        node = stack.pop()
        if node in visited:
            return False # cycle detected
        visited.add(node)

        for child in graph.get_children(node):
            if child in visited:
                return False # cycle detected
            stack.append(child)

    return len(visited) == len(graph.nodes)



def preorder_traversal(tree: DAG):
    if not is_tree(tree):
        raise Exception("Not a valid tree")

    root = tree.root()
    if root is None:
        return []

    result = []
    stack = [root[0]]

    while stack:
        node = stack.pop()
        result.append(node)

        stack.extend(reversed(tree.get_children(node)))

    return result

def postorder_traversal(net: DAG):
    root = net.root()[0]
    stack = [root]
    searched_nodes = []
    node2index = {root: 0}

    while stack:
        top_node = stack[-1]
        index = node2index[top_node]
        if index == net.out_degree(top_node):
            searched_nodes.append(stack.pop())
        else:
            it = iter(net.get_children(top_node))
            for _ in range(index):
                next(it)
            child = next(it)

            if child in searched_nodes:
                node2index[top_node] = index + 1
            else:
                stack.append(child)
                node2index[child] = 0
    return searched_nodes





def init_node_heights(graph: DAG):
    nodes = postorder_traversal(graph)
    for node in nodes:
        if not node.attribute_value('t'):
            node.add_attribute('t', 0)

        for par in node.get_parent(return_all=True):
            branch_length = node.length()[par][0]
            if not par.attribute_value('t'):
                par.add_attribute('t', branch_length + node.attribute_value('t'))





def has_cycle(network: DAG):
    def dfs(node, visited, stacked):
        if node in stacked:
            return True
        if node in visited:
            return False

        visited.add(node)
        stacked.add(node)

        for child in network.get_children(node):
            if dfs(child, visited, stacked):
                return True

        stacked.remove(node)
        return False

    visited_nodes = set()
    stacked_nodes = set()
    root_node = network.root()[0]

    return dfs(root_node, visited_nodes, stacked_nodes)












