import math


class Node:

        """
        Node wrapper class for storing data in a graph object

        propose/reject/accept structure not parallelizable at this time.
        """
        
        
        def __init__(self, branchLen=None, parNode=None, attr=None, isReticulation=False, name = None):
                self.branchLength = branchLen
                self.tempLen = None
                self.attributes = attr
                self.isReticulation = isReticulation
                self.parent = parNode
                self.label = name
                
        
        def addAttribute(self, key, value):
                self.attributes[key] = value
        
        def propose(self, newValue):
                """
                Stores the current value in a temporary holder while the
                newValue gets tested for viability 
                """
                self.tempLen = self.branchLength
                self.branchLength = newValue
        
        def accept(self):
                """
                If the proposed change is good, accept it by flushing the data 
                out of the temp container, symbolically cementing .height as the 
                official height
                """
                self.tempLen = None
        
        def reject(self):
                """
                If the proposed change is bad, reset the official height to what it
                was (the contents of the temp container).

                Flush the temp container of all data.
                """

                self.branchLength = self.tempLen
                self.tempLen = None
        
        def branchLen(self):
                """
                Defines either the weight of the edge between this node and its parent, or
                simply the difference in time between two nodes.
                """

                return self.branchLength

        def asString(self):
                myStr = "Node " + str(self.label) + ": "
                if self.branchLength != None:
                        myStr += str(self.branchLength) + " "
                if self.parent != None:
                        myStr += " has parent " + str(self.parent.name)
                
                return myStr


        