import random
import numpy as np
from Node import Node
from Graph import DAG
import math
import copy


class BirthDeathSimError(Exception):
    """
        This exception is thrown whenever something irrecoverably wrong happens
        during the process of generating trees
    """

    def __init__(self, message="Something went wrong simulating a tree"):
        self.message = message
        super().__init__(self.message)


def numLiveSpecies(nodes):
    """
        Returns the number of live lineages in a list of node objects

        nodeList -- an array/set of Node objects
        """
    return len(liveSpecies(nodes))


def randomSpeciesSelection(nodes):
    """
        Returns a random live Node from an array/set. The Node returned
        will be operated on during a birth or death event

        nodeList -- an array/set of Node objects
        """
    liveNodes = liveSpecies(nodes)
    randomInt = random.randint(0, len(liveNodes) - 1)
    return liveNodes[randomInt]


def liveSpecies(nodes):
    """
        Returns a subset of Nodes that represent live lineages

        nodeList -- an array/set of Node objects
        """
    return [node for node in nodes if node.attribute_value_if_exists("live") is True]


class Yule:
    """
        The Yule class represents a pure birth model for simulating
        trees of a fixed (n) amount of extant taxa.

        gamma-- the birth rate. A larger birth rate will result in shorter 
                branch lengths and a younger tree.
        
        n-- number of extant taxa at the end of simulation

        time-- if conditioning on time, the age of the tree to be simulated
        
        """

    def __init__(self, gamma, n, time):

        # birth rate
        self.gamma = gamma

        # goal number of taxa
        self.N = n

        # goal time
        self.time = time

        if self.N < 3:
            # choosing <3 because a tree with 2 taxa is always the same (branch lengths aside)
            raise BirthDeathSimError("Please generate a tree with at least 3 taxa")

        # current number of live lineages, always starts at 2
        self.lin = 2

        # helper var for labeling internal nodes
        self.internalCount = 1

        # amount of time elapsed during the simulation of a tree
        self.elapsedTime = 0

        # a list of trees generated under this model
        self.generatedTrees = []

    def drawWaitingTime(self):
        """
                Draw a waiting time until the next speciation event from 
                a memory-less exponential distribution.

                Since each lineage is equally likely for each event 
                under the Yule Model, the waiting time is given by the parameter 
                numlineages * birthRate or .lin*.gamma
                """
        scale = 1 / (self.lin * self.gamma)
        return round(np.random.exponential(scale), 2)

    def event(self, nodes, edges, condition="N"):
        """
                A speciation event occurs. Select a living lineage.

                Then add an internal "dead" node with branch length := t_parent + drawnWaitTime
                Set the parent to the chosen node as that internal node.

                nodes-- an array of nodes that represents the current state of the tree

                edges-- an array of 2-tuples (as arrays) that represents the current state of the tree
                
        """

        # select random live lineage to branch from
        specNode = randomSpeciesSelection(nodes)

        # keep track of the old parent, we need to disconnect edges
        oldParent = specNode.get_parent()

        # calculate the branch length to the internal node
        nextTime = self.drawWaitingTime()
        branchLen = 0
        if condition == "N":
            branchLen = self.elapsedTime + nextTime - specNode.get_parent().attribute_value_if_exists("t")
            self.elapsedTime += nextTime
        elif condition == "T" and self.elapsedTime + nextTime <= self.time:
            branchLen = self.elapsedTime + nextTime - specNode.get_parent().attribute_value_if_exists("t")
            self.elapsedTime += nextTime
        elif condition == "T" and self.elapsedTime + nextTime > self.time:
            return -1

        # create the new internal node
        newInternal = Node([branchLen], parent_nodes=[specNode.get_parent()], attr={"t": self.elapsedTime, "live": False},
                           name="internal" + str(self.internalCount))
        self.internalCount += 1

        # set the extent species parent to be its direct ancestor
        specNode.set_parent(newInternal)

        # there's a new live lineage
        self.lin += 1
        newLabel = "spec" + str(self.lin)

        # create the node for the new extent species
        newSpecNode = Node(parent_nodes=[newInternal], attr={"live": True}, name=newLabel)

        # add the newly created nodes
        nodes.append(newSpecNode)
        nodes.append(newInternal)

        # add the newly created branches, and remove the old connection (oldParent)->(specNode)
        edges.append([newInternal, newSpecNode])
        edges.append([newInternal, specNode])
        edges.append([oldParent, newInternal])
        edges.remove([oldParent, specNode])

        return nodes, edges

    def generateTree(self, condition="N"):
        """
                Simulate one tree under the model. Starts with a root and 2 living lineages
                and then continuously runs speciation (in this case birth only) 
                events until there are exactly self.N live species.

                After the nth event, draw one more time and fill out the remaining
                branch lengths.
        """

        # Set up the tree with 2 living lineages and an "internal" root node
        node1 = Node([0], attr={"t": 0, "label": "root", "live": False}, name="root")
        node2 = Node(parent_nodes=[node1], attr={"live": True}, name="spec1")
        node3 = Node(parent_nodes=[node1], attr={"live": True}, name="spec2")

        nodes = [node1, node2, node3]
        edges = [[node1, node2], [node1, node3]]

        # until the tree contains N extant taxa, keep having speciation events
        if condition == "N":
            while numLiveSpecies(nodes) < self.N:
                self.event(nodes, edges)

            # populate remaining branches with branch lengths according to
            # Eq 5.1? Just taking sigma_n for now
            nextTime = self.drawWaitingTime()

            for node in liveSpecies(nodes):
                node.add_attribute("t", self.elapsedTime + nextTime)
                if len(node.get_parent(True)) != 0:
                    node.set_length(self.elapsedTime + nextTime - node.get_parent().attribute_value_if_exists("t"))
                else:
                    node.set_length(0)

            # return the simulated tree
            tree = copy.deepcopy(DAG())
            tree.addEdges(edges)
            tree.addNodes(nodes)

            # reset the elapsed time to 0, and the number of live branches to 2
            # for correctness generating future trees
            self.elapsedTime = 0
            self.lin = 2

        elif condition == "T":
            while self.elapsedTime < self.time:
                status = self.event(nodes, edges, "T")
                if status == -1:
                    break

            for node in liveSpecies(nodes):
                node.add_attribute("t", self.time)
                if len(node.get_parent(True)) != 0:
                    node.set_length(self.time - node.get_parent().attribute_value_if_exists("t"))
                else:
                    node.set_length(0)

            tree = DAG()
            tree.addEdges(edges)
            tree.addNodes(nodes)

            # reset the elapsed time to 0, and the number of live branches to 2
            # for correctness generating future trees
            self.elapsedTime = 0
            self.lin = 2
        else:
            raise BirthDeathSimError("Condition parameter was not time ('T') or number of taxa ('N')")

        return tree

    def clearGenerated(self):
        """
                empty out the generated tree array
                """
        self.generatedTrees = []

    def generate_trees(self, num_trees):
        """
                The sequential version of generating a set number of trees.

                numTrees-- number of trees to generate and place into the generatedTrees database

                Outputs: the array of generated trees, includes all that have been previously generated
        """
        for dummy in range(num_trees):
            self.generatedTrees.append(self.generateTree())

        return self.generatedTrees


class CBDP:
    """
    Constant Rate Birth Death Process tree simulation
    """

    def __init__(self, gamma, mu, n, sample=1):

        # Eq 15 from https://www.sciencedirect.com/science/article/pii/S0022519309003300#bib24
        self.gamma = gamma / sample
        self.mu = mu - gamma * (1 - (1 / sample))

        self.sample = sample

        # probabilities of speciation or extinction event
        self.pBirth = self.gamma / (self.gamma + self.mu)
        self.pDeath = self.mu / (self.gamma + self.mu)
        self.N = n

        self.generatedTrees = []

    def Qinv(self, r):
        """
                Draw a time from the Qinv distribution from
                https://academic.oup.com/sysbio/article/59/4/465/1661436#app2

                r-- r[0] from the n-1 samples from [0,1]

                Returns: the time t, which is the age of a new simulated tree
        """
        term1 = (1 / self.gamma - self.mu)
        term2 = 1 - ((self.mu / self.gamma) * math.pow(r, 1 / self.N))
        term3 = 1 - math.pow(r, 1 / self.N)
        return term1 * math.log(term2 / term3)

    def Finv(self, r, t):
        """
        Draw a sample speciation time from the Finv distribution from
        https://academic.oup.com/sysbio/article/59/4/465/1661436#app2

                r-- r_i, from the sampled values from [0,1]
                t-- the age of the tree determined by Qinv(r[0])

                Returns: s_i from r_i
        """
        term1 = (1 / self.gamma - self.mu)
        term2 = self.gamma - (self.mu * math.exp(-1 * t * (self.gamma - self.mu)))
        term3 = 1 - math.exp(-1 * t * (self.gamma - self.mu))
        return term1 * math.log((term2 - self.mu * r * term3) / (term2 - self.gamma * r * term3))

    def generateTree(self):
        """
                Simulate a single tree under the Constant Rate Birth Death Selection Model.
                Follows the algorithm laid out by: https://academic.oup.com/sysbio/article/59/4/465/1661436#app2
                (Hartmann, Wong, Stadler)

                Returns: A tree with n taxa chosen from the proper distributions.
        """

        # step 1
        r = [random.random() for _ in range(self.N)]

        # step 2
        t = self.Qinv(r[0])

        # step 3
        s = {self.Finv(r[i], t): (i + .5) for i in range(1, self.N)}

        # step 4 setup

        sKeys = list(s.keys())

        nodes = []
        edges = []

        # set up leaf nodes and internal nodes in proper order (fig 5)
        for j in range(2 * self.N - 1):
            if j % 2 == 0:
                # leaf node
                leaf = Node(attr={"t": 0}, name="T" + str(int(j / 2) + 1))
                nodes.append(leaf)
            else:
                internal = Node(attr={"t": sKeys[int((j - 1) / 2)]}, name="internal" + str(int((j - 1) / 2)))
                nodes.append(internal)

        # step 4
        for i in range(2 * self.N - 1):
            # for each node, connect it to the correct parent
            new_edge = self.connect(i, nodes)
            if new_edge is not None:
                edges.append(new_edge)

        # add edges and nodes to a tree
        tree = DAG()
        tree.addEdges(edges)
        tree.addNodes(nodes)
        
        tree.generate_branch_lengths()

        return tree

    @staticmethod
    def connect(index, nodes):
        """
                nodes-- a list of nodes (list[i] is the ith node along a horizontal
                axis that alternates between species and internal s_i nodes/speciation events)

                index-- the node to connect to its parent in the tree

                Given the nodes and a node to connect, create a new edge.

                The parent node is defined to be the closest to nodes[index] in terms
                of time and proximity in the list. There are two candidates, the left and right 
                candidate. Each candidate is the nearest element in the list such that the time 
                attribute is larger than nodes[index]. The parent is the minimum of the 
                two candidates.

                Returns: the edge from nodes[index] to its correct parent

                """

        # find right candidate
        copyIndex = index + 1
        rightCandidate = None

        while copyIndex < len(nodes):
            # search in the list to the right (ie increase the index)
            if nodes[copyIndex].attribute_value_if_exists("t") > nodes[index].attribute_value_if_exists("t"):
                rightCandidate = nodes[copyIndex]
                break
            copyIndex += 1

        # find left candidate
        copyIndex = index - 1
        leftCandidate = None
        while copyIndex >= 0:
            # search in the left part of the list
            if nodes[copyIndex].attribute_value_if_exists("t") > nodes[index].attribute_value_if_exists("t"):
                leftCandidate = nodes[copyIndex]
                break
            copyIndex -= 1

        # take the minimum time (leaves being at time 0, root being at max time)
        if leftCandidate is None and rightCandidate is None:
            # We're running this on the root
            return
        elif leftCandidate is None:
            selection = rightCandidate
        elif rightCandidate is None:
            selection = leftCandidate
        else:
            comp = rightCandidate.attribute_value_if_exists("t") - leftCandidate.attribute_value_if_exists("t")
            if comp >= 0:
                selection = leftCandidate
            else:
                selection = rightCandidate

        # create new edge
        nodeT = nodes[index].attribute_value_if_exists("t")
        futureT = selection.attribute_value_if_exists("t")
        newEdge = [selection, nodes[index]]

        # set the branch length of the current node
        nodes[index].set_length(futureT - nodeT, selection)
        #nodes[index].set_parent([selection])

        return newEdge

    def sampleTrees(self, m):
        """
                Generate m trees and add them to the list of generated trees

                Returns: the list of all generated trees from this run and any prior
                         uncleared runs.
                """
        for dummy in range(m):
            self.generatedTrees.append(self.generateTree())

        return self.generatedTrees

    def clearGenerated(self):
        """
                Clear out the generated trees list
                """
        self.generatedTrees = []
