import random
from UltrametricNetwork import UltrametricNetwork
import MHSettings
from PopulationSize import PopulationSize

class StateNode:

    operator = None
    dirty = False

    def getOperation(self):
        return self.operator
    

    def setDirty(self, dirty):
        self.dirty = dirty
    

    def isDirty(self):
        return self.dirty
    

    def getOp(self, operators, weights):
        if(len(operators) < len(weights)):
            #throw new RuntimeException("Operator-Weight pair doesn't match " + Arrays.toString(weights))
            print("error")
        
        rand = random.random()
        for i in range(len(weights)):
            if(rand < weights[i]):
                return operators[i]
            
        
        print("RETURNING NONE, ERROR IN GETOP FUNC")
        ## throw new RuntimeException("Error propose Operation " + rand) ## should never reach here
    
class State:

    ## inferred or fixed topologies/parameters

    popSizeParamWeight = 0.005
    moveNode = None
    

    def __init__(self, network, trees, markerSeqs, poissonParameter, species2alleles, BAGTRModel):

        if(SNAPPLikelihood.pseudoLikelihood != None):
            SNAPPLikelihood.pseudoLikelihood.cache.clear()
            
        

        self.geneTrees = []
        for i in range(len(markerSeqs)):
            if(trees == None):
                self.geneTrees.append(UltrametricTree(markerSeqs[i]))
            else:
                self.geneTrees.append(UltrametricTree(trees[i], markerSeqs[i]))
            
        
        self.speciesNet = UltrametricNetwork(network, self.geneTrees, markerSeqs, species2alleles, BAGTRModel)
        self.populationSize = PopulationSize()
        self.priorDistribution = SpeciesNetPriorDistribution(poissonParameter, self.populationSize)
        ##self.gtOpWeight = 1.0 - Math.min(0.3, Math.max(0.1, 8.0 / (.self.geneTrees.size() + 8.0)))
    

    def toString(self):
        return "[" + self.speciesNet.getNetwork().getRoot().getRootPopSize() + "]" + self.speciesNet.getNetwork().toString() + "\nTopology: " + Networks.getTopologyString(self.speciesNet.getNetwork()) + "\nGamma Mean: " + self.populationSize.toString()
    

    def toNetworkString(self):
        return "[" + self.speciesNet.getNetwork().getRoot().getRootPopSize() + "]" + self.speciesNet.getNetwork().toString()
    

    
    def toList(self):
        mylist = []
        mylist.append(self.speciesNet.getNetwork().toString())
        # /*for(UltrametricTree t : .self.geneTrees) {
        #     list.add(t.toString())
        # }*/
        if(MHSettings._ESTIMATE_POP_SIZE):
            mylist.append(self.populationSize.toString())
        
        return mylist
    

    def propose(self):
        rand = random.random()

        if (MHSettings._ESTIMATE_POP_SIZE and MHSettings._ESTIMATE_POP_PARAM and rand < self.popSizeParamWeight):
            self.moveNode = self.populationSize
        else:
            ## always do network operation (jiafan)
            self.moveNode = self.speciesNet
        
        self.geneTrees = []

        logHR = self.moveNode.propose()
        self.moveNode.setDirty(True)
        if(self.getOperation().getName().contains("Scale-All")):
            # /*for(UltrametricTree ut : self.geneTrees) {
            #     ut.setDirty(True)
            # }*/
            logHR = MHSettings.INVALID_MOVE
        

        if (self.getOperation().getName().contains("Add-Reticulation") and
                self.speciesNet.getNetwork().getReticulationCount() > MHSettings._NET_MAX_RETI):
            logHR = MHSettings.INVALID_MOVE
        elif (self.moveNode.mayViolate()):
            if(self.speciesNet.isDirty() and not self.priorDistribution.isValid(self.speciesNet.getNetwork())):
                if(MHSettings.DEBUG_MODE):
                    #System.err.println(self.getOperation())
                    print("oopsies")
                logHR = MHSettings.INVALID_MOVE
            elif(not self.speciesNet.isValid()):
                logHR = MHSettings.INVALID_MOVE
            
        
        return logHR
    

    #Undo the proposal, restore the last state
    def undo(self, logAlpha):
        self.moveNode.undo()
        self.speciesNet.reject()
        ##for (UltrametricTree gt : self.geneTrees) gt.reject()

        self.moveNode._operator._rejCounter+=1
        self.moveNode._operator._rejCorrectionCounter+=1
        if (logAlpha != MHSettings.INVALID_MOVE):
            self.moveNode._operator.optimize(logAlpha)
        self.moveNode = None
    

    # accept the proposal
    def accept(self, logAlpha):
        self.speciesNet.accept()
        ##for (UltrametricTree gt : self.geneTrees) gt.accept()

        self.moveNode._operator._acCounter+=1
        self.moveNode._operator._acCorrectionCounter+=1
        if (logAlpha != MHSettings.INVALID_MOVE):
            self.moveNode._operator.optimize(logAlpha)
        self.moveNode = None
    

    def getOperation(self):
        return self.moveNode.getOperation()
    

    def calculatePrior(self):
        return self.priorDistribution.logPrior(self.speciesNet.getNetwork())
    

    def calculateLikelihood(self):
        logL = self.speciesNet.logDensity()
        # /*for(UltrametricTree gt : self.geneTrees) {
        #     logL += gt.logDensity()
        # }*/
        return logL


    def calculateScore(self):
        return self.calculatePrior() + self.calculateLikelihood()
    

    def recalculateLikelihood(self):
        logL = self.speciesNet.recomputeLogDensity()
        # /*for(UltrametricTree gt : self.geneTrees) {
        #     logL += gt.logDensity()
        # }*/
        return logL
    

    def numOfReticulation(self):
        return self.speciesNet.getNetwork().getReticulationCount()
    

    ##  function should only be called under DEBUG_MODE
    def isValidState(self):
        if(not MHSettings.DEBUG_MODE): 
            return True

        if(not self.priorDistribution.isValid(self.speciesNet.getNetwork())):
            #System.err.println("Invalid network")
            return False
        elif(not self.speciesNet.isValid()):
            #System.err.println("Invalid temporal constraints")
            return False
        elif(not self.speciesNet.isUltrametric()):
            #System.err.println("Invalid network - not ultrametric")
            return False
        else:
            # /*for(UltrametricTree gt : self.geneTrees) {
            #     if(gt.isValid()) continue
            #     System.err.println("Invalid gene tree - not ultrametric")
            #     return False
            # }*/
            return True
   

    def getNetwork(self):
        if(Double.isNaN(self.speciesNet.getNetwork().getRoot().getRootPopSize())):
            return self.speciesNet.getNetwork().toString()
        else:
            return "[" + self.speciesNet.getNetwork().getRoot().getRootPopSize() + "]" + self.speciesNet.getNetwork().toString()
    

    def getNetworkObject(self):
        return self.speciesNet.getNetwork()
    

    def getUltrametricNetworkObject(self):
        return self.speciesNet
    
