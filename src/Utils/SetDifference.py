class SetDifference:
        setA = set()
        setB = set()

        def __init__(self, seta, setb):
                self.setA = seta
                self.setB = setb
        
        def inANotInB(self):
                """
                Returns the elements of Set A that are not present in Set B
                """
                return set([elem for elem in self.setA if elem not in self.setB])
        
        def inBNotInA(self):
                """
                Returns the elements of Set B that are not present in Set A
                """
                return set([elem for elem in self.setB if elem not in self.setA])
        
        def getBoth(self):
                """
                Returns a 2 element list, each element being A-B, B-A respectively
                """
                return [self.inANotInB(), self.inBNotInA()]