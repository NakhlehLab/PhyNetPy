import random
import numpy as np
from Node import Node


def numLiveSpecies(nodeList):
        return len([node for node in nodeList if node.attrLookup("live") == True])

def randomSpeciesSelection(nodeList):
        liveNodes = [node for node in nodeList if node.attrLookup("live") == True]
        randomInt = random.randint(0,len(liveNodes))
        return liveNodes[randomInt]

class Yule:

        def __init__(self, gamma, n):
                self.gamma = gamma
                self.N = n
                self.lin = 2


        def drawWaitingTime(self):
                scale = 1 / (self.lin * self.gamma)
                return np.random.exponential(scale)
        
        def event(self, nodes):
                """
                A speciation event occurs. Select a living lineage.

                Then add an internal "dead" node with branch length := t_parent + drawnWaitTime
                Set the parent to the chosen node as that internal node.
                
                """

                #select random live lineage to branch from
                specNode = randomSpeciesSelection(nodes)

                #calculate the branch length to the internal node
                branchLen = specNode.getParent().attrLookup("t") + self.drawWaitingTime()

                #create the new internal node
                newInternal = Node(parNode=[specNode.getParent()], attr={"t":branchLen, "live":False})

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

                return nodes


        def generateTree(self):
                node1 = Node(attr = {"t":0, "label": "root", "live":False})
                node2 = Node(parNode=[node1], attr={"live":True}, name="spec1")
                node3 = Node(parNode=[node1], attr={"live":True}, name="spec2")

                nodes = [node1, node2, node3]

                while numLiveSpecies(nodes) < self.N:
                        self.event(nodes)
                        



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


        def generateTree(self, numTaxa):
                root = Node(0, )
                nodes = [root]

                while nodes
