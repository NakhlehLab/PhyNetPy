import math


class Node:

        """
        Node wrapper class for storing data in a graph object

        propose/reject/accept structure not parallelizable at this time.
        """
        
        
        def __init__(self, height, attr):
                self.height = height
                self.tempHeight = None
                self.attributes = attr
                
        
        def addAttribute(self, key, value):
                self.attributes[key] = value
        
        def propose(self, newValue):
                """
                Stores the current value in a temporary holder while the
                newValue gets tested for viability 
                """
                self.tempHeight = self.height
                self.height = newValue
        
        def accept(self):
                """
                If the proposed change is good, accept it by flushing the data 
                out of the temp container, symbolically cementing .height as the 
                official height
                """
                self.tempHeight = None
        
        def reject(self):
                """
                If the proposed change is bad, reset the official height to what it
                was (the contents of the temp container).

                Flush the temp container of all data.
                """

                self.height = self.tempHeight
                self.tempHeight = None
        
        def branchLen(self, otherNode):
                """
                Defines either the weight of the edge between this node and otherNode, or
                simply the difference in time between two nodes.
                """

                return math.abs(otherNode.getHeight()-self.height)

        def getHeight(self):
                return self.height