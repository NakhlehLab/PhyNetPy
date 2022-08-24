import MHSettings
import math
import random
import PopulationSize

class State:

    ## inferred or fixed topologies/parameters
    ##private double _gtOpWeight;
    _popSizeParamWeight = 0.005
    _moveNode = None


    def __init__(self, network, trees, markerSeqs, poissonParameter, species2alleles, BAGTRModel):

        if(None): ## SNAPPLikelihood.pseudoLikelihood != null (HARDCODED TO NULL FOR NOW)
            SNAPPLikelihood.pseudoLikelihood.cache.clear()
        

        self.geneTrees = []
        for i in range(len(markerSeqs)):
            if(trees == None):
                self.geneTrees.append(UltrametricTree(markerSeqs[i]))
            else:
                self.geneTrees.append(UltrametricTree(trees[i], markerSeqs[i]))
            
        
        self.speciesNet = UltrametricNetwork(network, self.geneTrees, markerSeqs, species2alleles, BAGTRModel)
        self.populationSize = PopulationSize()
        self.priorDistribution = SpeciesNetPriorDistribution(poissonParameter, self.populationSize);
        ##_gtOpWeight = 1.0 - Math.min(0.3, Math.max(0.1, 8.0 / (this._geneTrees.size() + 8.0)));
    

    
    
    def toString(self):
        return "[" + self.speciesNet.getNetwork().getRoot().getRootPopSize() + "]" + self.speciesNet.getNetwork().toString() + "\nTopology: " + Networks.getTopologyString(self.speciesNet.getNetwork()) + "\nGamma Mean: " + self.populationSize.toString()
    

    def toNetworkString(self):
        return "[" + self.speciesNet.getNetwork().getRoot().getRootPopSize() + "]" + self.speciesNet.getNetwork().toString()
    

    def toList(self):
        result = []
        result.append(self.speciesNet.getNetwork().toString())
        #for(UltrametricTree t : this._geneTrees) {
        #    list.add(t.toString());
        #}*/
        if(MHSettings._ESTIMATE_POP_SIZE):
            result.append(self.populationSize.toString())
        
        return result
    

    
    def propose(self):
        randNum = random.random()

        if(MHSettings._ESTIMATE_POP_SIZE and MHSettings._ESTIMATE_POP_PARAM and randNum < self.popSizeParamWeight):
            self._moveNode = self.populationSize
        
        else:
            ##always do network operation (jiafan)
            self._moveNode = self.speciesNet
        
        self.geneTrees = []

        logHR = self._moveNode.propose()
        self._moveNode.setDirty(True)
        if(self.getOperation().get_name().contains("Scale-All")):
            ##for(UltrametricTree ut : _geneTrees) {
            ##    ut.setDirty(true);
            ##}
            logHR = MHSettings.INVALID_MOVE
        

        if(self.getOperation().get_name().contains("Add-Reticulation") and
                self.speciesNet.getNetwork().getReticulationCount() > MHSettings._NET_MAX_RETI):
                logHR = MHSettings.INVALID_MOVE
        elif (self._moveNode.mayViolate()):
                if(self.speciesNet.isDirty() and not self.priorDistribution.isValid(self.speciesNet.getNetwork())):
                        ##if(MHSettings.DEBUG_MODE):
                        ##        ##System.err.println(self.getOperation())
                        logHR = MHSettings.INVALID_MOVE
                elif(not self.speciesNet.isValid()):
                        logHR = MHSettings.INVALID_MOVE
            
        return logHR
    

    
    def undo(self, logAlpha):
        """
        Restores the last state
        """
        self._moveNode.undo()
        self.speciesNet.reject()
        ##for (UltrametricTree gt : _geneTrees) gt.reject();

        self._moveNode._operator._rejCounter += 1
        self._moveNode._operator._rejCorrectionCounter += 1
        if (logAlpha != MHSettings.INVALID_MOVE):
            self._moveNode._operator.optimize(logAlpha)
        self._moveNode = None

    
    def accept(self, logAlpha):
        self.speciesNet.accept()
        ##for (UltrametricTree gt : _geneTrees) gt.accept();

        self._moveNode._operator._acCounter += 1
        self._moveNode._operator._acCorrectionCounter += 1
        if (logAlpha != MHSettings.INVALID_MOVE):
            self._moveNode._operator.optimize(logAlpha)
        self._moveNode = None
    

    
    def getOperation(self):
        return self._moveNode.getOperation()
    

   
    def calculatePrior(self):
        return self.priorDistribution.logPrior(self.speciesNet.getNetwork())
    


    def calculateLikelihood(self):
        logL = self.speciesNet.logDensity()
        # /*for(UltrametricTree gt : _geneTrees) {
        #     logL += gt.logDensity();
        # }*/
        return logL
    

    def calculateScore(self):
        return self.calculatePrior() + self.calculateLikelihood()
    

    def recalculateLikelihood(self):
        logL = self.speciesNet.recomputeLogDensity()
        # /*for(UltrametricTree gt : _geneTrees) {
        #     logL += gt.logDensity();
        # }*/
        return logL
    

    def numOfReticulation(self):
        return self.speciesNet.getNetwork().getReticulationCount()
    

    ## this function should only be called under DEBUG_MODE
    def isValidState(self):
        if(not MHSettings.DEBUG_MODE):
                return True

        if(not self.priorDistribution.isValid(self.speciesNet.getNetwork())):
            #System.err.println("Invalid network");
            return False
        elif (not self.speciesNet.isValid()):
            #System.err.println("Invalid temporal constraints");
            return False
        elif(not self.speciesNet.isUltrametric()):
            #System.err.println("Invalid network - not ultrametric");
            return False
        else:
        #     /*for(UltrametricTree gt : _geneTrees) {
        #         if(gt.isValid()) continue;
        #         System.err.println("Invalid gene tree - not ultrametric");
        #         return false;
        #     }*/
            return True
        
    

    def getNetwork(self):
        if(math.isnan(self.speciesNet.getNetwork().getRoot().getRootPopSize())):
            return self.speciesNet.getNetwork().toString()
        else:
            return "[" + self.speciesNet.getNetwork().getRoot().getRootPopSize() + "]" + self.speciesNet.getNetwork().toString()
   
   
    def getNetworkObject(self):
        return self.speciesNet.getNetwork()
    

    def getUltrametricNetworkObject(self):
        return self.speciesNet
    

