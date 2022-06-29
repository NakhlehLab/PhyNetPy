class Counter:
        count = 0

        def __init__(self):
                self.count = 0
        
        def zero(self):
                self.count = 0
        
        def increment(self):
                self.count+=1
        
        def decrement(self):
                self.count-=1
        
        def getCount(self):
                return self.count