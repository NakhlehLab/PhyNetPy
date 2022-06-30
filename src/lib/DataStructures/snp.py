class SNAP:

        def __init__(self, values = []):
                self.matrix = values

        
        def MetropolisHastings(self, out="text.txt"):
                result = MetroHastings(self.matrix).run()
                result.writeToFile(out)
                
