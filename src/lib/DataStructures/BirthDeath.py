import random
import numpy as np
from Node import Node
from Graph import DAG
import math
import copy
from threading import Thread
from threading import Lock
import time


def numLiveSpecies(nodeList):
        return len([node for node in nodeList if node.attrLookup("live") == True])

def randomSpeciesSelection(nodeList):
        liveNodes = liveSpecies(nodeList)
        randomInt = random.randint(0,len(liveNodes)-1)        
        return liveNodes[randomInt]

def liveSpecies(nodeList):
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
                return self.model.generateTrees(self.tasks)
                #print("Thread "+ str(self.thread_ID) + " has completed")
                
                

class Yule:

        def __init__(self, gamma, n):
                self.gamma = gamma
                self.N = n
                self.lin = 2
                self.internalCount = 1
                self.elapsedTime = 0
                self.generatedTrees = []


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

                return tree
        
        def clearGenerated(self):
                self.generatedTrees = []
        
        def generateNTreesParallel(self, numTrees, numThreads):
                """
                Return a list of numTrees number of simulated trees.

                Not yet parallelized, m threads would grant the ability to sim
                m trees at once, reducing runtime for large trees or a large number of trees
                """
                ids = 0
                totalTrees = []
                for dummy in range(numThreads):
                        newThread = treeGen("thread" + str(ids), ids , self, int(numTrees / numThreads))
                        ids+=1
                        totalTrees.append(newThread.run())
                
                return totalTrees
        

        def generateNTreesSeq(self, numTrees):
                for dummy in range(numTrees):
                        self.generatedTrees.append(self.generateTree())

                return self.generatedTrees
        
        def generateTrees(self, tasks):
                trees = []
                for dummy in range(tasks):
                        trees.append(self.generateTree())
                
                return trees



                        



class CBDP:

        def __init__(self, gamma, mu, n):
                self.gamma = gamma
                self.mu = mu
                self.pBirth = self.gamma / (self.gamma + self.mu)
                self.pDeath = self.mu / (self.gamma + self.mu)
                self.N = n


        def drawWaitingTime(self, numLineages):
                scale = 1 / (numLineages * (self.mu + self.gamma))
                return np.random.exponential(scale)
        
        def event(self):
                """
                A speciation event occurs. Return 1 if it is a birth event,
                or 0 for a death event.
                """
                draw = random.random()

                if draw < (1 - self.pBirth):
                        return 0 #death
                else:
                        return 1 #birth



sim = Yule(.05, 10)

startSeq = time.perf_counter()
seqTrees = sim.generateNTreesSeq(10000)
endSeq = time.perf_counter()
print(len(seqTrees))
sim.clearGenerated()


startPar = time.perf_counter()
parTrees = sim.generateNTreesParallel(10000, 10)
endPar = time.perf_counter()

print(len(parTrees[0]))

print("SEQUENTIAL TIME:" + str(endSeq-startSeq))
print("PARALLEL TIME:" + str(endPar-startPar))


