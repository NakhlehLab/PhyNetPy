import random
import numpy as np
from Node import Node
from Graph import DAG
import math
import copy
from threading import Thread
from threading import Lock
import time
from collections import deque

class BirthDeathSimError(Exception):
        """
        This exception is thrown whenever something irrecoverably wrong happens
        during the process of generating trees
        """

        def __init__(self, message = "Something went wrong simulating a tree"):
                self.message = message
                super().__init__(self.message)


def numLiveSpecies(nodeList):
        """
        Returns the number of live lineages in a list of node objects

        nodeList -- an array/set of Node objects
        """
        return len([node for node in nodeList if node.attrLookup("live") == True])

def randomSpeciesSelection(nodeList):
        """
        Returns a random live Node from an array/set. The Node returned
        will be operated on during a birth or death event

        nodeList -- an array/set of Node objects
        """
        liveNodes = liveSpecies(nodeList)
        randomInt = random.randint(0,len(liveNodes)-1)        
        return liveNodes[randomInt]

def liveSpecies(nodeList):
        """
        Returns a subset of Nodes that represent live lineages

        nodeList -- an array/set of Node objects
        """
        return [node for node in nodeList if node.attrLookup("live") == True]


class treeGen(Thread):
        def __init__(self, thread_name, thread_ID, yuleObj, tasks):
                Thread.__init__(self)
                self.thread_name = thread_name
                self.thread_ID = thread_ID
                self.model = yuleObj
                self.tasks = tasks
 
        # helper function to execute the threads
        def run(self):
                self.model.generateTrees(self.tasks)
                #print("Thread "+ str(self.thread_ID) + " has completed")
                
                

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

                #birth rate
                self.gamma = gamma

                #goal number of taxa
                self.N = n

                #goal time
                self.time = time

                if self.N < 3:
                        #choosing <3 because a tree with 2 taxa is always the same (branch lengths aside)
                        raise BirthDeathSimError("Please generate a tree with at least 3 taxa")


                #current number of live lineages, always starts at 2
                self.lin = 2

                #helper var for labeling internal nodes
                self.internalCount = 1

                #amount of time elapsed during the simulation of a tree
                self.elapsedTime = 0

                #a list of trees generated under this model
                self.generatedTrees = []
                self.parallelTrees = ThreadSafeList()
                


        def drawWaitingTime(self):
                """
                Draw a waiting time until the next speciation event from 
                a memoryless exponential distribution.

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

                #select random live lineage to branch from
                specNode = randomSpeciesSelection(nodes)

                #keep track of the old parent, we need to disconnect edges
                oldParent = specNode.getParent()

                #calculate the branch length to the internal node
                nextTime = self.drawWaitingTime()
                if condition == "N":
                        branchLen = self.elapsedTime + nextTime - specNode.getParent().attrLookup("t")
                        self.elapsedTime += nextTime
                elif condition == "T" and self.elapsedTime + nextTime <= self.time:
                        branchLen = self.elapsedTime + nextTime - specNode.getParent().attrLookup("t")
                        self.elapsedTime += nextTime
                elif condition == "T" and self.elapsedTime + nextTime > self.time:
                        return -1
                        # branchLen = self.time - specNode.getParent().attrLookup("t")
                        # self.elapsedTime = self.time

                #create the new internal node
                newInternal = Node(branchLen, parNode=[specNode.getParent()], attr={"t":self.elapsedTime, "live":False}, name = "internal" + str(self.internalCount))
                self.internalCount+= 1

                #set the extent species parent to be its direct ancestor
                specNode.setParent(newInternal)

                #there's a new live lineage
                self.lin+=1
                newLabel = "spec" + str(self.lin)

                #create the node for the new extent species
                newSpecNode = Node(parNode = [newInternal], attr = {"live":True}, name = newLabel)

                #add the newly created nodes
                nodes.append(newSpecNode)
                nodes.append(newInternal)

                #add the newly created branches, and remove the old connection (oldParent)->(specNode)
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

                #Set up the tree with 2 living lineages and an "internal" root node
                node1 = Node(0, attr = {"t":0, "label": "root", "live":False}, name = "root")
                node2 = Node(parNode=[node1], attr={"live":True}, name="spec1")
                node3 = Node(parNode=[node1], attr={"live":True}, name="spec2")

                nodes = [node1, node2, node3]
                edges = [[node1, node2], [node1, node3]]

                #until the tree contains N extant taxa, keep having speciation events
                if(condition == "N"):
                        while numLiveSpecies(nodes) < self.N:
                                self.event(nodes, edges)
                        
                        #populate remaining branches with branch lengths according to
                        #Eq 5.1? Just taking sigma_n for now
                        nextTime = self.drawWaitingTime()

                        for node in liveSpecies(nodes):
                                node.addAttribute("t", self.elapsedTime + nextTime)
                                if len(node.getParent(True)) != 0:
                                        node.setBranchLength(self.elapsedTime + nextTime - node.getParent().attrLookup("t"))
                                else:
                                        node.setBranchLength(0)
                

                        #return the simulated tree
                        tree = copy.deepcopy(DAG())
                        tree.addEdges(edges)
                        tree.addNodes(nodes)

                        #reset the elapsed time to 0, and the number of live branches to 2 
                        #for correctness generating future trees
                        self.elapsedTime = 0
                        self.lin = 2
                
                elif condition == "T":
                        while self.elapsedTime < self.time:
                                status = self.event(nodes, edges, "T")
                                if status == -1:
                                        break
                        
                        for node in liveSpecies(nodes):
                                node.addAttribute("t", self.time)
                                if len(node.getParent(True)) != 0:
                                        node.setBranchLength(self.time - node.getParent().attrLookup("t"))
                                else:
                                        node.setBranchLength(0)

                        tree = copy.deepcopy(DAG())
                        tree.addEdges(edges)
                        tree.addNodes(nodes)

                        #reset the elapsed time to 0, and the number of live branches to 2 
                        #for correctness generating future trees
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
        
        def generateNTreesParallel(self, numTrees, numThreads):
                """
                Return a list of numTrees number of simulated trees.
                Runs using numThreads number of threads, for exact results
                numThreads should be a divisor of numTrees.

                numTrees-- number of trees to generate

                numThreads-- number of threads to use

                Outputs: A list (length numThreads) of lists, each inner list being the results of one
                thread of execution. The total number of trees is still numTrees.
        
                """
                threads = []
                ids = 0
                for dummy in range(numThreads):
                        #create a thread with a unique name and id, and give it the right number of tasks
                        newThread = treeGen("thread" + str(ids), ids , self, int(numTrees / numThreads))
                        threads.append(newThread)
                        ids+=1

                        #run the thread and append the results
                        newThread.start()
                
                for thread in threads:
                        thread.join()
                
                return self.parallelTrees
        

        def generateNTreesSeq(self, numTrees):
                """
                The sequential version of generating a set number of trees.

                numTrees-- number of trees to generate and place into the generatedTrees database

                Outputs: the array of generated trees, includes all that have been previously generated
                """
                for dummy in range(numTrees):
                        self.generatedTrees.append(self.generateTree())

                return self.generatedTrees
        

        def generateTrees(self, tasks):
                """
                Parallel helper function that generates a set number of trees and 
                places them into an array

                tasks-- number of trees to generate

                Outputs: the array of trees
                """
                
                for dummy in range(tasks):
                        self.parallelTrees.append(self.generateTree())
                
                

class CBDP:

        def __init__(self, gamma, mu, n, sample=1):

                #Eq 15 from https://www.sciencedirect.com/science/article/pii/S0022519309003300#bib24
                self.gamma = gamma / sample
                self.mu = mu - gamma*(1 - (1/sample))

                self.sample = sample

                #probabilities of speciation or extinction event
                self.pBirth = self.gamma / (self.gamma + self.mu)
                self.pDeath = self.mu / (self.gamma + self.mu)
                self.N = n

                self.generatedTrees = []


        def Qinv(self, r):
                """
                Draw a time from the Qinv distribution from https://academic.oup.com/sysbio/article/59/4/465/1661436#app2

                r-- r[0] from the n-1 samples from [0,1]

                Returns: the time t, which is the age of a new simulated tree
                """
                term1 = (1 / self.gamma - self.mu) 
                term2 = 1 - ((self.mu / self.gamma) * math.pow(r, 1 / self.N))
                term3 = 1 - math.pow(r, 1 / self.N)
                return term1 * math.log(term2 / term3)
        
        def Finv(self, r, t):
                """
                Draw a sample speciation time from the Finv distribution from  https://academic.oup.com/sysbio/article/59/4/465/1661436#app2

                r-- r_i, from the sampled values from [0,1]
                t-- the age of the tree determined by Qinv(r[0])

                Returns: s_i from r_i
                """
                term1 = (1 / self.gamma - self.mu)
                term2 = self.gamma - (self.mu * math.exp(-1*t*(self.gamma - self.mu)))
                term3 = 1 - math.exp(-1*t*(self.gamma - self.mu))
                return term1 * math.log((term2 - self.mu*r*term3) /(term2 - self.gamma*r*term3))

        def generateTree(self):
                """
                Simulate a single tree under the Constant Rate Birth Death Selection Model.
                Follows the algorithm laid out by: https://academic.oup.com/sysbio/article/59/4/465/1661436#app2
                (Hartmann, Wong, Stadler)

                Returns: A tree with n taxa chosen from the proper distributions.
                """

                #step 1
                r = [random.random() for dummy in range(self.N)]

                #step 2
                t = self.Qinv(r[0])
                print(t)

                #step 3
                s = {self.Finv(r[i], t): (i + .5) for i in range(1, self.N)}
                
                #step 4 setup

                sKeys = list(s.keys())
                

                nodes = []
                edges = []

                #set up leaf nodes and internal nodes in proper order (fig 5)
                for j in range(2*self.N - 1):
                        if j % 2 == 0:
                                #leaf node
                                leaf = Node(attr = {"t":0}, name= "species" + str(int(j/2)))
                                nodes.append(leaf)
                        else:
                                internal = Node(attr={"t": sKeys[int((j-1)/2)]}, name= "internal" + str(int((j-1)/2)))
                                nodes.append(internal)
                
                #step 4
                for i in range(2*self.N - 1):
                        #for each node, connect it to the correct parent 
                        edges.append(self.connect(i, nodes))
                
                #add edges and nodes to a tree
                tree = copy.deepcopy(DAG())
                tree.addEdges(edges)
                tree.addNodes(nodes)

                return tree
        

        def connect(self, index, nodes):
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

                #find right candidate
                copyIndex = index + 1
                rightCandidate = None
                
                while copyIndex < len(nodes):
                        #search in the list to the right (ie increase the index)
                        if nodes[copyIndex].attrLookup("t") > nodes[index].attrLookup("t"):
                                rightCandidate = nodes[copyIndex]
                                break
                        copyIndex += 1


                #find left candidate
                copyIndex = index - 1
                leftCandidate = None
                while copyIndex >= 0:
                        #search in the left part of the list 
                        if nodes[copyIndex].attrLookup("t") > nodes[index].attrLookup("t"):
                                leftCandidate = nodes[copyIndex]
                                break
                        copyIndex -= 1
                
                #take the minimum time (leaves being at time 0, root being at max time)
                if leftCandidate == None and rightCandidate == None:
                        #We're running this on the root
                        return
                elif leftCandidate == None:
                        selection = rightCandidate
                elif rightCandidate == None:
                        selection = leftCandidate
                else:
                        comp = rightCandidate.attrLookup("t") - leftCandidate.attrLookup("t")
                        if comp >= 0:
                                selection = leftCandidate
                        else:
                                selection = rightCandidate

                #create new edge
                nodeT = nodes[index].attrLookup("t")
                futureT = selection.attrLookup("t")
                newEdge = [selection, nodes[index]]

                #set the branch length of the current node
                nodes[index].setBranchLength(futureT - nodeT)
                nodes[index].setParent(selection)

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
                        













class ThreadSafeList():
    # constructor
    def __init__(self):
        # initialize the list
        self._list = list()
        # initialize the lock
        self._lock = Lock()
 
    # add a value to the list
    def append(self, value):
        # acquire the lock
        with self._lock:
            # append the value
            self._list.append(value)
 
    # remove and return the last value from the list
    def pop(self):
        # acquire the lock
        with self._lock:
            # pop a value from the list
            return self._list.pop()
 
    # read a value from the list at an index
    def get(self, index):
        # acquire the lock
        with self._lock:
            # read a value at the index
            return self._list[index]
 
    # return the number of items in the list
    def length(self):
        # acquire the lock
        with self._lock:
            return len(self._list)
 
        # add items to the list
    def add_items(safe_list):
        for i in range(100000):
                safe_list.append(i)
 




sim = Yule(.05, 6, 30)

#sim.generateTree("T").printGraph()

sim2 = CBDP(.05, .01, 6)
sim2.generateTree().printGraph()

# startSeq = time.perf_counter()
# sim.generateNTreesSeq(500000)
# endSeq = time.perf_counter()
# print(len(sim.generatedTrees))
# sim.clearGenerated()


# startPar = time.perf_counter()
# sim.generateNTreesParallel(500000, 4)
# endPar = time.perf_counter()

# print(sim.parallelTrees.length())

# print("SEQUENTIAL TIME:" + str(endSeq-startSeq))
# print("PARALLEL TIME:" + str(endPar-startPar))


