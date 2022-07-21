import random
import numpy as np
from Node import Node
from Graph import DAG
import math
import copy
from threading import Thread
from threading import Lock
import time


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
        
        """

        def __init__(self, gamma, n):

                #birth rate
                self.gamma = gamma

                #goal number of taxa
                self.N = n

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
        
        def event(self, nodes, edges):
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
                branchLen = self.elapsedTime + nextTime - specNode.getParent().attrLookup("t")
                self.elapsedTime += nextTime

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


        def generateTree(self):
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
                
                



                        



# class CBDP:

#         def __init__(self, gamma, mu, n):
#                 self.gamma = gamma
#                 self.mu = mu
#                 self.pBirth = self.gamma / (self.gamma + self.mu)
#                 self.pDeath = self.mu / (self.gamma + self.mu)
#                 self.N = n


#         def drawWaitingTime(self, numLineages):
#                 scale = 1 / (numLineages * (self.mu + self.gamma))
#                 return np.random.exponential(scale)
        
#         def event(self):
#                 """
#                 A speciation event occurs. Return 1 if it is a birth event,
#                 or 0 for a death event.
#                 """
#                 draw = random.random()

#                 if draw < (1 - self.pBirth):
#                         return 0 #death
#                 else:
#                         return 1 #birth




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
 




sim = Yule(.05, 10)

startSeq = time.perf_counter()
sim.generateNTreesSeq(500000)
endSeq = time.perf_counter()
print(len(sim.generatedTrees))
sim.clearGenerated()


startPar = time.perf_counter()
sim.generateNTreesParallel(500000, 4)
endPar = time.perf_counter()

print(sim.parallelTrees.length())

print("SEQUENTIAL TIME:" + str(endSeq-startSeq))
print("PARALLEL TIME:" + str(endPar-startPar))


