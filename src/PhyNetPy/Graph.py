""" 
Author : Mark Kessler
Last Edit : 2/20/24
First Included in Version : 1.0.0
Approved for Release: NO
"""

from collections import defaultdict, deque
import copy
import math
import random
from typing import Callable
from Node import Node
import numpy as np




def random_object(mylist : list, rng):
    """
    Select a random item from a list using an rng object (for testing consistency and debugging purposes)

    Args:
        mylist (list): a list of any type
        rng : the random rng object

    Returns:
        ? : an item from mylist
    """
    rand_index = rng.integers(0, len(mylist))
    return mylist[rand_index]

def newickSubstring(children, node):
    """
    Returns the newick string for a subtree beginning at node

    ie a whole tree (A, (B, C)D)E with newickSubstring(D) returns (B, C)D

    node-- the root node (type Node) that starts the computation
    children-- a mapping from parent nodes to a list of their children
    
    """
    if node in children.keys():

        retStr = "("

        for child in children[node][0]:
            if not children[node][1]:
                retStr += newickSubstring(children, child)
            else:
                retStr += child.get_name()
                if child.length() is not None:
                    #TODO: NETWORKS?
                    retStr += ":" + str(list(child.length().values())[0])
            retStr += ", "

        retStr = retStr[:-2]  # Get rid of spurious comma

        retStr += ")" + node.get_name()
        if node.length() is not None:
            retStr += ":" + str(list(node.length().values())[0])
    else:
        retStr = node.get_name()
        if node.length() is not None:
            retStr += ":" + str(list(node.length().values())[0])

    return retStr


class GraphError(Exception):
    """
    This exception is raised when a graph is malformed, or if a graph operation fails.
    """

    def __init__(self, message="Error with a Graph Instance"):
        self.message = message
        super().__init__(self.message)



class DAG():
    """
    This class represents a directed graph containing nodes of any data type, and edges.
    An edge is a list [a,b] where a and b are nodes in the graph, and the direction of the edge
    is from a to b. [a,b] is NOT the same as [b,a], as this is a DIRECTED graph.

    Allowances:
    1) You may create cycles-- BUT we have provided a method to check if this graph object is acyclic
    2) You may have multiple roots. Be mindful of whether this graph is connected and what root you wish to operate on
    3) You may end up with floater nodes/edges, ie this may be an unconnected graph with multiple connected components. 
       We will provide in the next release a method to check for whether your graph object is one single connected component. 
       We have also provided methods to remove such artifacts.
    
    
    """
    

    def __init__(self, edges = None, nodes = None, weights = None) -> None:
        """
        Initialize a Directed Acyclic Graph (DAG) object.
        You may initialize with any combination of edges/nodes/weights, or provide none at all.

        Args:
            edges (list, optional): A list of Node 2-tuples. Defaults to None.
            nodes (list, optional): A list of Node objs. Defaults to None.
            weights (list, optional): Unused as of right now. Defaults to None.
        """
        
        # Map nodes to the number of children they have
        self.out_degrees : dict[Node, int] = defaultdict(int)
        
        # Map nodes to the number of parents they have
        self.in_degrees : dict[Node, int] = defaultdict(int)
        
        # List of nodes in the graph with in degree of 0. Note that there could be floater nodes w/ out degree 0
        self.roots : list[Node] = []
        
        # List of nodes in the graph with out degree of 0. Note that there could be floater nodes w/ in degree 0
        self.leaves : list[Node] = []
        
        # Outgroup is a node that is a leaf and whose parent is the root node.
        self.outgroup : Node = None
        
        # Blob storage for anything that you want to associate with this network. Just give it a string key!
        self.items : dict[str, object] = {}
        
        # Map of nodes to their parents
        self.parent_map : dict[Node, list[Node]] = defaultdict(list) 
        
        # Map of nodes to their children
        self.child_map : dict[Node, list[Node]] = defaultdict(list)
        
        #Map of names of nodes to nodes
        self.node_names : dict[str, Node] = {}
        
        if edges is None:
            self.edges = []
        else:
            self.edges = edges
        
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
            for node in self.nodes:
                self.node_names[node.get_name()] = node
            
        if weights is None:
            self.weights = []
        else:
            self.weights = weights

        # Initialize the unique id count
        self.UID = 0
        
        # If edges and nodes provided, then start bookkeeping!
        if nodes is not None and edges is not None:
            
            #Free floater nodes/edges are *technically* allowed
            for edge in self.edges:
                self.in_degrees[edge[1]] += 1
                self.out_degrees[edge[0]] += 1
                self.parent_map[edge[1]].append(edge[0])
                self.child_map[edge[0]].append(edge[1])
            
            self.leaves = [node for node in self.nodes if self.out_degrees[node] == 0]
            self.roots = [node for node in self.nodes if self.in_degrees[node] == 0]
            
    def add_nodes(self, nodes):
        """
        Args:
            nodes -- List[Node] or Node
        
        If nodes is a list of nodes, then add each node point to the list
        If nodes is simply a node, then just add the one node to the nodes list.
        
        """
        if type(nodes) == list:
            for node in nodes:
                if node not in self.nodes:
                    self.nodes.append(node)
                    self.node_names[node.get_name()] = node
        else:
            if nodes not in self.nodes:
                self.nodes.append(nodes)
                self.node_names[nodes.get_name()] = nodes
        return

    def update_node_name(self, node : Node, name : str):
        """
        Rename a node and update the bookkeeping.

        Args:
            node (Node): a node in the graph
            name (str): the new name for the node.
        """
        if node.get_name() is not None:
            del self.node_names[node.get_name()]
        node.set_name(name)
        self.node_names[name] = node
        
    def add_edges(self, edges)->None:
        """
        If edges is a list of lists, 
        then add each list (edge) to the list of edges
        
        If edges is simply a list, then just add the list to the edge array.

        Args:
            edges (?): Either a list[list[Node]] or a list[Node].
    
        """
        as_list : bool = True
        if len(edges) > 0:
            if type(edges[0]) is list:
                pass
            elif type(edges[0]) is Node:
                if len(edges) != 2:
                    raise GraphError("'edges' parameter has node elements but \
                                     is not of length 2.")
                as_list = False
            else:
                raise GraphError("'edges' parameter needs to be a list of Node\
                                 arrays, or a singular Node array (of size 2)")                                 
                                 
        if as_list:
            for edge in edges:
                if type(edge) is not list:
                    raise GraphError("")
                self.edges.append(edge) 
                self.out_degrees[edge[0]] += 1
                self.in_degrees[edge[1]] += 1
                self.child_map[edge[0]].append(edge[1])
                self.parent_map[edge[1]].append(edge[0])
                edge[1].add_parent(edge[0])  
                self.reclassify_node(edge[0], True, True)
                self.reclassify_node(edge[1], False, True)    
        else:
            if type(edge) is tuple:
                edge = list(edge)
            self.edges.append(edges)
            edges[1].add_parent(edges[0])
            self.out_degrees[edges[0]] += 1
            self.in_degrees[edges[1]] += 1
            self.child_map[edges[0]].append(edges[1])
            self.parent_map[edges[1]].append(edges[0])
            self.reclassify_node(edges[0], True, True)
            self.reclassify_node(edges[1], False, True)   
        return

    def remove_node(self, node : Node):
        """
        Removes node from the list of nodes.
        Also prunes all edges from the graph that are connected to the node
        
        Args:
            node (Node): a node in the graph
        """
        
        if node in self.nodes:
            for edge in self.edges:
                if node in edge:
                    self.remove_edge(edge)
                        
            self.nodes.remove(node)
            del self.in_degrees[node]
            del self.out_degrees[node]
            del self.child_map[node]
            del self.parent_map[node]  
            del self.node_names[node.get_name()]     
    
    def remove_edge(self, edge : list[Node]):
        """
        Args:
            edge (list[Node]): an edge to remove from the graph
            
        Removes edge from the list of edges. Does not delete nodes with no edges
        """
        
        if edge in self.edges:
            self.edges.remove(edge)
            edge[1].remove_parent(edge[0])
            self.out_degrees[edge[0]] -= 1
            self.in_degrees[edge[1]] -= 1
            
            try:
                self.parent_map[edge[1]].remove(edge[0])
            except:
                pass
        
            try:
                self.child_map[edge[0]].remove(edge[1])
            except:
                pass
            
            self.reclassify_node(edge[0], True, False)
            self.reclassify_node(edge[1], False, False)
        
    def reclassify_node(self, node : Node, is_par : bool, is_addition : bool):
        """
        

        Args:
            node (Node): A node in the graph
            is_par (bool): flag that tells the method whether the node is being operated on as a parent (true) or child (false)
            is_addition (bool): flag that tells the method whether the node arg is an addition (true) or subtraction(false)
        """
        if is_addition:
            if is_par:
                # If out degree now = 1, then the node was previously a leaf and is not anymore
                if self.out_degrees[node] == 1:
                    try:
                        self.leaves.remove(node)
                    except:
                        pass
                if self.in_degrees[node] == 0:
                    if node not in self.roots:
                        self.roots.append(node)
            else:
                # If in_degree now = 1, then the node was previously a root and is not anymore
                if self.in_degrees[node] == 1:
                    try:
                        self.roots.remove(node)
                    except:
                        pass
                if self.out_degrees[node] == 0:
                    if node not in self.leaves:
                        self.leaves.append(node)
                        pars = self.get_parents(node)
                        if len(pars) == 1 and self.roots == 1:
                            if pars[0] in self.roots:
                                self.outgroup = node
        else:
            if is_par:
                # if out degree is now = 0, then the node is now a leaf
                if self.out_degrees[node] == 0:
                    self.leaves.append(node)
                    pars = self.get_parents(node)
                    if len(pars) == 1 and self.roots == 1:
                        if pars[0] in self.roots:
                            self.outgroup = node
            else:
                # if in degree is now = 0, the node is now a root
                if self.in_degrees[node] == 0:
                    self.roots.append(node)
                    
    def get_outgroup(self):
        return self.outgroup
        
    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges
        
    def get_item(self, key : str):
        return self.items[key]
    
    def put_item(self, key : str, item):
        if key not in self.items:
            self.items[key] = item

    def add_uid_node(self, node : Node):
        """
        Ensure an added node has a unique name that hasn't been used before/is not currently in use for this graph

        Args:
            node (Node): _description_
        """
        if node not in self.nodes:
            self.nodes.append(node)
            self.update_node_name(node, "UID_" + str(self.UID))
            self.UID += 1
    
    def in_degree(self, node: Node):
        return self.in_degrees[node]

    def out_degree(self, node: Node):
        return self.out_degrees[node]

    def in_edges(self, node: Node):
        return [[par, node] for par in self.parent_map[node]]

    def out_edges(self, node: Node):
        return [[node, child] for child in self.child_map[node]]
    
    def root(self) -> list[Node]:
        """
        Finds the root(s) of this DAG
    
        """
        return [root for root in self.roots if self.out_degrees[root] != 0]

    def get_parents(self, node: Node) -> list[Node]:
        """
        Returns a list of the parents of node

        node-- A Node Object
        """
        return self.parent_map[node]
        
    def get_children(self, node: Node) -> list[Node]:
        """
        Returns a list of the children of a node. For a tree, this 
        list should be of length 2. For a network, this number may only be one

        node-- A Node Object
        """
        return self.child_map[node]

    def get_leaves(self) -> list[Node]:
        """
        returns the list of leaves in the graph, aka the set of nodes where there are no incoming edges
        """
        
        return [leaf for leaf in self.leaves if self.in_degrees[leaf] != 0]

    def has_node_named(self, name : str) -> Node:
        """
        Check whether the graph has a node with a certain name.
        Strings must be exactly equal (ie, same white space, capitalization, etc.)

        Args:
            name (str): the name to search for

        Returns:
            Node: the node with the given name
        """
        try:
            return self.node_names[name]
        except:
            return None

    def print_graph(self):
        """
        Prints node information for each node in the graph
        """
        for node in self.nodes:
            if type(node) is Node:
                print(node.asString())

    def pretty_print_edges(self)-> None:
        for edge in self.edges:
            print(f"<{edge[0].get_name()}, {edge[1].get_name()}>")
            print(edge)
            print("------")
        
    def remove_floaters(self) -> None:
        """
        Remove all nodes from the graph that have in degree = out degree = 0
        
        Inputs : No inputs
        Returns : None
        """
        
        floaters = [node for node in self.nodes if self.in_degrees[node] == 0 and self.out_degrees[node] == 0]
        for floater in floaters:
            self.nodes.remove(floater)
            if floater in self.in_degrees.keys():
                del self.in_degrees[floater]
            if floater in self.out_degrees.keys():
                del self.out_degrees[floater]
            if floater in self.child_map.keys():
                del self.child_map[floater]
            if floater in self.parent_map.keys():
                del self.parent_map[floater]
       
    def remove_excess_branch(self):
        """
        Any typical root node will have 2 outgoing edges. 
        If there's only 1, then we can just delete the edge and there will be a new root node.
        
         root        root
         /  \   vs     |
        c1  c2         c1
        """
        root = self.root()[0]
        if self.out_degrees[root] == 1:
            self.remove_edge([root, self.get_children(root)[0]])
        
    def prune_excess_nodes(self) -> None:
        """
        Again, think about refactoring this.
        
        Routine that removes nodes with in/out = 1 from a graph. These nodes are not interesting and serve no purpose.
        """
        
        root = self.root()[0]
        
        q = deque()
        q.appendleft(root)
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                current_node : Node = neighbor
                previous_node : Node = cur
                node_removed = False
                
                #There could be a chain of nodes with in/out degree = 1. Resolve the whole chain before moving on to search more nodes
                while self.in_degree(current_node) == self.out_degree(current_node) == 1:
                    self.remove_edge([previous_node, current_node])
                    
                    previous_node = current_node
                    temp = self.get_children(current_node)[0]
                    self.remove_node(current_node)
                    current_node = temp
                    node_removed = True
                
                #We need to connect cur to its new successor
                if node_removed:
                    #self.remove_edge([previous_node, current_node])
                    self.add_edges([cur, current_node])    
                    current_node.set_parent([cur])
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(current_node)
                    
    def generate_branch_lengths(self) -> None:
        """
        Assumes that each node in the graph does not yet have a branch length associated with it,
        but has a defined "t" attribute.

        Raises:
            Exception: If a node is encountered that does not have a "t" value defined in its attribute dictionary
        """
        root = self.root()[0]
        root.add_length(0, None)
        
        # stack for dfs
        q = deque()
        q.append(root)
        visited = set()

        while len(q) != 0:
            cur = q.pop()

            for neighbor in self.get_children(cur):
                if neighbor not in visited:
                    t_par = cur.attribute_value("t")
                    t_nei = neighbor.attribute_value("t")
                    
                    #Handle case that a "t" value doesn't exist
                    if t_par is None or t_nei is None:
                        raise GraphError("Assumption that t attribute exists for all nodes has been violated")
                    neighbor.add_length(t_par - t_nei, cur)
                
                    q.append(neighbor)
                    visited.add(neighbor)

    def mrca(self, set_of_nodes: set)-> Node:
        """
        Computes the Least Common Ancestor of a set of graph nodes

        Args:
            set_of_nodes (set): A set of Node objs that should be a subset of the set of leaves. 

        Returns:
            Node: The node that is the LCA of the set of leaves that was passed in.
        """
        format_set = set()
        for item in set_of_nodes:
            if type(item) is str:
                node_version = self.has_node_named(item)
                if node_version is None:
                    raise GraphError("Wrong name passed into LCA calculation")
                else:
                    format_set.add(node_version)
            elif type(item) is Node:
                format_set.add(item)
            else:
                raise GraphError("Wrong item type passed into LCA calculation")
        
        
        set_of_nodes = format_set
                
        leaf_2_parents = {} # mapping from each node in set_of_nodes to a mapping from ancestors to dist from node.
        
        for leaf in set_of_nodes:
            #Run bfs upward from each node 
            node_2_lvl : dict = {}
            
            # queue for bfs
            q = deque()
            q.append(leaf)
            visited = set()
            node_2_lvl[leaf] = 0

            while len(q) != 0:
                cur = q.popleft()

                #Seach cur's parents
                for neighbor in self.get_parents(cur):
                    if neighbor not in visited:
                        node_2_lvl[neighbor] = node_2_lvl[cur] + 1
                        q.append(neighbor)
                        visited.add(neighbor)
            
            leaf_2_parents[leaf] = node_2_lvl
        
        
        #Compare each leaf's parents
        intersection = set(self.nodes)
        for leaf, par_level in leaf_2_parents.items():
            intersection = intersection.intersection(set(par_level.keys()))
        
        
    
        additive_level = {} # Map potential LCA's to cumulative distance from all the nodes
        
        for node in intersection: # A LCA has to be in each node's ancestor set
            lvl = 0
            for leaf in set_of_nodes:
                try:
                    lvl += leaf_2_parents[leaf][node]
                except KeyError:
                    continue
            
            additive_level[node] = lvl
        
        #The LCA is the node that minimizes cumulative distance
        return min(additive_level, key=additive_level.get)
                      
    def leaf_descendants(self, node : Node)->tuple:
        """
        Compute the set of all leaf nodes that are descendants of the parameter node. Uses DFS to find paths to leaves

        Args:
            node (Node): The node for which to compute leaf children

        Returns:
            tuple: The list of all leaves
        """
        root = node

        # stack for dfs
        q = deque()
        q.appendleft(root)
        leaves = set()

        while len(q) != 0:
            cur = q.popleft()
        
            if self.out_degrees[cur] == 0:
                leaves.add(cur)
                
            for neighbor in self.get_children(cur): #Continue path to a leaf
                q.append(neighbor)
        
        return leaves    
        
    def diff_subtree_edges(self, rng)-> list[list[Node]]:
        """
        Returns 2 random edges such that neither edge is in the same "direct" subtree.

        Args:
            rng (random.default.rng): an rng object.
        
        Returns:
            list[list[Node]]: a list of 2 edges such that neither edge is downstream of the other.
        """
        #Grab a random edge
        first_edge = random_object(self.edges, rng)
        
        #Find another edge while excluding descendants of the first edge
        first_edge_subtree = self.leaf_descendants(first_edge[1])
        second_edge = random_object([edge for edge in self.edges if len(set(self.leaf_descendants(edge[1])).intersection(set(first_edge_subtree))) == 0], rng)
        
        return [first_edge, second_edge]
    
    def subgenome_count(self, n : Node)->int:
        """
        THINK ABOUT REFACTORING!
        
        
        Given a node in this graph, return the subgenome count.
         
        Args:
            n (Node): Any node in the graph. It is an error to input a node that is not in the graph.

        Returns:
            int: subgenome count
        """
        if n not in self.nodes:
            raise GraphError("Input node is not in the graph")
        
        if self.root()[0] == n:
            return 1
        else:
            parents = self.get_parents(n)
            if len(parents) > 0:
                return sum([self.subgenome_count(parent) for parent in self.get_parents(n)])
            else:
                return 0
    
    def edges_downstream_of_node(self, n : Node)-> list[list[Node]]:
        """
        Returns the set (as a list) of edges that are in the subgraph of a node.

        Args:
            n (Node): A node in a graph.
        Returns:
            edges (list[list[Node]]): The set of all edges in the subgraph of n.
        """
        if n not in self.nodes:
            raise GraphError("Input node is not in the graph.")
        
        q = deque()
        q.appendleft(n)
        
        edges : list[list[Node]] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
                edges.append([cur, neighbor])
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return edges
    
    def edges_to_subgenome_count(self, downstream_node : Node = None, delta : float = math.inf, start_node : Node = None):
        """
        Maps edges to their subgenome counts.
        
        Args:
            downstream_node (Node, optional): No edges will be included in the map that are in a subgraph of this node. Defaults to None.
            delta (float, optional): Only include edges in the mapping that have subgenome counts <= delta. Defaults to math.inf.
            start_node (Node, optional): Provide a node only if you don't want to start at the root. Defaults to None, which will result in starting at the root.

        Raises:
            GraphError: If the graph has more than one root to start.

        Returns:
            dict[tuple[Node], int]: a map from edges (as tuples for hashability) to subgenome counts
        """
    
        if start_node is None:
            start_nodes = self.root()
            if len(self.root()) != 1:
                raise GraphError("Please specify a start node for this network, there is more than one root (or none)")
            start_node = start_nodes[0]
            
        q = deque()
        q.appendleft(start_node)
        
        edges_2_sub = {tuple(edge) : 0 for edge in self.edges}
        
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
                edges_2_sub[(cur, neighbor)] += 1
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
    
        
        partition : dict[int, list[list[Node]]] = {}
        for edge, value in edges_2_sub.items():
            if value not in partition.keys():
                partition[value] = [list(edge)]
            else:
                partition[value].append(list(edge))
        
        #Filter out invalid keys
        
        filter1 = {key : value for (key, value) in partition.items() if key <= delta}
        
        #Filter out edges that would create a cycle from param edge
        if downstream_node is not None:
            filter2 = {}
            for subct, edges in filter1.items():
                for target in edges:
                    downstream_edges = self.edges_downstream_of_node(downstream_node)
                    if target not in downstream_edges:
                        if subct not in filter2.keys():
                            filter2[subct] = [target]
                        else:
                            filter2[subct].append(target)
            return filter2
        else:
            return filter1

    def leaf_descendants_all(self) -> dict[Node, set[Node]]:
        """
        Map each node in the graph to its set of leaf descendants
        Returns:
            dict[Node, set[Node]]: map from graph nodes to their leaf descendants
        """
        desc_map : dict[Node, set[Node]] = {}
        
        #Mutates desc_map
        leaf_desc_help(self, self.root()[0], self.get_leaves(), desc_map)
        
        return desc_map
        
    def newick(self):
        return newick_help(self, self.root()[0], set()) + ";"
    
    def is_cyclic_util(self, v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True

        for neighbor in self.get_children(v):
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    def is_acyclic(self) -> bool:
        """
        Checks if each of this graph's connected components is acyclic

        Returns:
            bool: True if acyclic, False if cyclic. 
        """
        
        #Maintain structures for checking nodes that are visited or in the recursive stack
        visited = {node : False for node in self.nodes}
        rec_stack = {node : False for node in self.nodes}

        #Call recursive dfs on each root node / each connected component
        for node in self.root():
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return False

        return True
    
    def bfs_dfs(self, start_node : Node = None, dfs : bool = False, is_connected : bool = False, accumulator : Callable = None, accumulated = None) -> dict[Node, int]:
        """
        General bfs-dfs routine, with the added utility of checking whether or not this graph is made up of multiple 
        connected components.

        Args:
            start_node (Node, optional): Give a node to start the search from. Defaults to None, in which case the search will start at the root.
            dfs (bool, optional): Flag that specifies whether to use bfs or dfs. Defaults to False (bfs), if true is passed, will run dfs.
            is_connected (bool, optional): Flag that, if enabled, will check for the connected component status. Defaults to False (won't run).
            accumulator (Callable, optional): A function that takes the currently searched Node in the graph and does some sort of bookkeeping.
            accumulated (Any): Any type of structure that stores the data given by the accumulator function.

        Returns:
            dict[Node, int]: Distance mapping from nodes to their distance from the start node.
        """
        q : deque = deque()
        visited : set[Node] = set()
        
        
        if start_node is not None:
            q.append(start_node)
            dist = {start_node : 0}
            visited.add(start_node)
        else:
            root : Node = self.root()[0]
            q.append(root)
            dist = {root : 0}
            visited.add(root)
        
        while len(q) != 0:
            
            if dfs: #LIFO
                cur : Node = q.popleft() #Adding to left, so popleft is LIFO behavior
            else: #FIFO
                cur : Node = q.pop() #Popright is FIFO behavior
            
            if accumulator is not None and accumulated is not None:
                accumulated = accumulator(cur, accumulated)
            
            for neighbor in self.get_children(cur):
                dist[neighbor] = dist[cur] + 1
                q.appendleft(neighbor)
                visited.add(neighbor)
        
        if is_connected:
            if len(set(self.nodes).difference(visited)) != 0:
                print("GRAPH HAS MORE THAN 1 CONNECTED COMPONENT")
            else:
                print("GRAPH IS FULLY CONNECTED")
        
        return dist, accumulated
        
    def reset_outgroup(self, new_outgroup : Node)->None:
        """
        Change the root of the network such that new_outgroup (a leaf) is now the outgroup of the network.
        
         
        
        Args:
            new_outgroup (Node): _description_

        Raises:
            GraphError: _description_
        """
        if new_outgroup == self.get_outgroup():
            return
        
        pars = self.get_parents(new_outgroup)
        if len(pars) != 1:
            raise GraphError("Could not set the outgroup to the given node. Violates assumption 1")
        
        #Assumption 2: parent of new outgroup must not be a parent to a reticulation node
        children : list[Node] = self.get_children(pars[0])
        
        for child in children:
            if child.is_reticulation():
                raise GraphError("Could not set the outgroup to the given node. Violates assumption 2")
        
        #Check Assumption 1, that the path to the root must not contain any reticulation nodes
        path_to_root = self.rootpaths(pars[0])
        if len(path_to_root) != 1:
            raise GraphError("Could not set the outgroup to the given node. Violates assumption 1")
        
        #Valid outgroup, now switch roots (by reversing all the edges on the path to the root)
        for edge in path_to_root:
            self.remove_edge(edge)
            self.add_edges([edge[1], edge[0]]) #Reverse the edge
        
        self.prune_excess_nodes() #There may be excess nodes as a result of the edge flipping process
        
        
        #These better match
        print(self.outgroup)
        print(new_outgroup.get_name())
        
    def rootpaths(self, start : Node):
        paths : list[list[list[Node]]] = [] #A list of paths, each path is a list of edges (which are lists of nodes)
        for par in self.get_parents(start):
            for path in self.rootpaths(par):
                paths.append(path.append([par, start]))
        return paths
    
    def subtree_copy(self, retic_node : Node):
        """
        Make a copy of a subnetwork of this DAG, rooted at @retic_node, with unique node names
        
        Args:
            retic_node (Node): A node in net that is a reticulation node

        Returns:
            DAG: A subnetwork of the DAG being operated on
        """
    
        q = deque()
        q.appendleft(retic_node)
        nodes = []
        edges = []
        
        new_node = Node(name = retic_node.get_name() + "_copy")
        nodes.append(new_node)
        net_2_mul = {retic_node : new_node}
        

        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                new_node = Node(name = neighbor.get_name() + "_copy")
                nodes.append(new_node)
                net_2_mul[neighbor] = new_node
                edges.append([net_2_mul[cur], new_node])
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return DAG(edges = edges, nodes=nodes) 
    
        
        
        
#### HELPER FUNCTIONS ####

def leaf_desc_help(net : DAG, node : Node, leaves : list[Node], desc_map : dict[Node, set[Node]]):
    """
    Helper function for "leaf_descedants_all".  

    Args:
        net (DAG): _description_
        node (Node): _description_
        leaves (list[Node]): _description_
        desc_map (dict[Node, set[Node]]): _description_

    Returns:
        _type_: _description_
    """
    if node not in desc_map.keys():
        if node in leaves:
            desc_map[node] = {node}
        else:
            desc_map[node] = set()
            for child in net.get_children(node):
                desc_map[node] = desc_map[node].union(leaf_desc_help(net, child, leaves, desc_map))
                    
    return desc_map[node]    


def newick_help(net : DAG, node : Node, processed_retics : set[Node]):
    """
    Helper function to "newick".

    Args:
        net (DAG): _description_
        node (Node): _description_
        processed_retics (set[Node]): _description_

    Returns:
        _type_: _description_
    """
        
    if node in net.leaves:
        return node.get_name()
    else:
        if net.in_degrees[node] == 2 and node in processed_retics:
            if node.get_name()[0] != "#":
                return "#" + node.get_name()
            return node.get_name()
        else:
            if net.in_degrees[node] == 2:
                processed_retics.add(node)
                if node.get_name()[0] != "#":
                    node_name = "#" + node.get_name()
                else:
                    node_name = node.get_name()
            else:
                node_name = node.get_name()    
                
            substr = "("
            for child in net.get_children(node):
                substr += newick_help(net, child, processed_retics)
                substr += ","
            substr = substr[0:-1]
            substr += ")"
            substr += node_name
            
            return substr

