from state import StateNode
import MHSettings
import math


class PopulationSize(StateNode):

#     popSize ## GammaDistribution
    gammaMean = MHSettings._POP_SIZE_MEAN / MHSettings.GAMMA_SHAPE;
#     operators
#     opWeights

    
    def __init__(self):
        self.popSize = GammaDistribution(MHSettings.GAMMA_SHAPE, self.gammaMean)
        self.operators = Operator[]{new ChangePopSizeParam(this), new ScalePopSizeParam(this)}
        self.opWeights = MHSettings.PopSize_Op_Weights
    

    def getGammaMean(self):
        return self.gammaMean
    

    def setGammaMean(self, mean):
        if(not MHSettings._ESTIMATE_POP_SIZE):
            #throw new RuntimeException("Don't allow to change gamma mean parameter for population size estimation");
            print("oopsies")

        if(not MHSettings._ESTIMATE_POP_PARAM):
            mean = MHSettings._POP_SIZE_MEAN / MHSettings.GAMMA_SHAPE
        self.gammaMean = mean
        self.popSize = GammaDistribution(MHSettings.GAMMA_SHAPE, self.gammaMean)
    

    def propose(self):
        self.operator = self.getOp(self.operators, self.opWeights)
        return self.operator.propose()

    def undo(self):
        if(self.operator == None):
                #throw new IllegalArgumentException("null operator");
                print("oopsies")
        self.operator.undo()
    

    def accept(self):
        return None

    def reject(self):
        return None

    def logDensity(self):
        return -math.log(self.gammaMean)
    

    def mayViolate(self):
        return False
    
    
    def isValid(self):
        return True
    

    def toString(self):
        return str(self.gammaMean)
    

    def density(self, popSize):
        return self.popSize.density(popSize)
    


