from cmath import nan
import math
import MHSettings

class UltrametricNetwork(stateNode):

 ## STILL A BUNCH TO FIX FOR THIS CLASS
    

    logGeneTreeNetwork = None
    logLtemp = None

    numSites = 0
    

    def __init__(self, s, gts, markerSeqs, s2a, BAGTRModel):
        self.markers = markerSeqs
        self.numSites = markerSeqs.get(0).getSiteCount()
        self.geneTrees = gts
        self.species2alleles = s2a
        self.alleles2species = None
        self.BAGTRModel = BAGTRModel
        self.numThreads = MHSettings._NUM_THREADS

        #populate reverse map
        if (self.species2alleles != None):
            self.alleles2species = {}
            for key in self.species2alleles.keys():
                for allele in self.species2alleles[key]:
                    self.alleles2species[allele] = key
                

        init = (s == None)
        popSize = nan

        if(not init):
            if(s.startsWith("[")):
                popSize = Double.parseDouble(s.substring(1, s.indexOf("]")))
                ##MHSettings._POP_SIZE_MEAN = popSize
                s = s.substring(s.indexOf("]") + 1)
            
            self.network = Networks.readNetwork(s) ## adopt topology only
            taxa = set()
            for leaf in self.network.getLeaves(): 
                taxa.add(leaf.getName())
            
            if(self.species2alleles != None and (not taxa.containsAll(self.species2alleles.keySet()) or not self.species2alleles.keySet().containsAll(taxa))):
                #System.err.println("The starting network doesn't match the taxaMap")
                #System.err.println(self.network.toString() + "\n v.s. \n" + Arrays.toString(self.species2alleles.keySet().toArray()))
                #System.err.println("The starting network will be set to the MDC tree given gene trees")
                init = True
            

        if(init):
            mdc = MDCInference_Rooted()
            trees = []
            for t in self.geneTrees:
                trees.append([t.getTree(), 1.0])
            
            #Solution sol = (self.alleles2species == None) ?
                #       mdc.inferSpeciesTree(trees, False, 1, False, true, -1).get(0) :
                #   mdc.inferSpeciesTree(trees, self.alleles2species, False, 1, False, true, -1).get(0)
            
            startingTree= Trees.generateRandomBinaryResolution(sol._st)
            s = startingTree.toNewick()
            self.network = Networks.readNetwork(s) ## adopt topology only
        
        constraints = TemporalConstraints.getTemporalConstraints(gts, self.species2alleles, self.alleles2species)
        self.initNetHeights(self.popSize, constraints)

        if(MHSettings.SAMPLE_SPLITTING):
            self.initSplitting()
        else:
            self.splittings = None
        

        if(SNAPPLikelihood.useApproximateBayesian):
            self.abcData = {}
        

        self.setOperators()
    

    def setOperators(self):
        if(MHSettings._MCMC):
            self.operators = new Operator[]{
                    new ChangePopSize(this),
                    new ScalePopSize(this),
                    new ScaleAll(_geneTrees, this),
                    new ScaleTime(this), new ScaleRootTime(this), new ChangeTime(this),
                    new SlideSubNet(this), new SwapNodes(this), new MoveTail(this),
                    new AddReticulation(this),
                    new FlipReticulation(this), new MoveHead(this),
                    new DeleteReticulation(this),
                    new ChangeInheritance(this)
            }

            if (MHSettings._ESTIMATE_POP_SIZE):
                self.treeOpWeights = MHSettings.getOperationWeights(MHSettings.Net_Tree_Op_Weights)
                self.netOpWeights = MHSettings.getOperationWeights(MHSettings.Net_Op_Weights)
            else:
                self.treeOpWeights = MHSettings.getOperationWeights(
                        MHSettings.Net_Tree_Op_Weights, 3, MHSettings.Net_Tree_Op_Weights.length)
                self.netOpWeights = MHSettings.getOperationWeights(
                        MHSettings.Net_Op_Weights, 3, MHSettings.Net_Op_Weights.length)
            
        else:
            self.operators = new Operator[]{
                    new ChangePopSize(this),
                    new ScalePopSize(this),
                    new ScaleAll(_geneTrees, this),
                    new ScaleTime(this), new ScaleRootTime(this), new ChangeTime(this),
                    new SlideSubNet(this), new SwapNodes(this), new MoveTail(this),
                    new AddReticulation(this),
                    new FlipReticulation(this), new MoveHead(this),
                    new DeleteReticulation(this),
                    new ChangeInheritance(this),
                    new ReplaceReticulation(this)
            }
            if (MHSettings._ESTIMATE_POP_SIZE):
                self.treeOpWeights = MHSettings.getOperationWeights(MHSettings.Search_Net_Tree_Op_Weights)
                self.netOpWeights = MHSettings.getOperationWeights(MHSettings.Search_Net_Op_Weights)
            else:
                self.treeOpWeights = MHSettings.getOperationWeights(
                        MHSettings.Search_Net_Tree_Op_Weights, 3, MHSettings.Search_Net_Tree_Op_Weights.length)
                self.netOpWeights = MHSettings.getOperationWeights(
                        MHSettings.Search_Net_Op_Weights, 3, MHSettings.Search_Net_Op_Weights.length)
            
        

    

#     ## used only for debug only
#      UltrametricNetwork(String s, List<MarkerSeq> markerSeqs, Map<String, List<String>> species2alleles, BiAllelicGTR BAGTRModel) {
#         self.network = Networks.readNetworkWithRootPop(s)
#         self.species2alleles = species2alleles
#         if(self.species2alleles != None) {
#             self.alleles2species = new HashMap<>()
#             for(String key: self.species2alleles.keySet()){
#                 for(String allele: self.species2alleles.get(key)){
#                     self.alleles2species.put(allele, key)
#                 }
#             }
#         }
#         self.geneTrees = None
#         self.markers = markerSeqs
#         self.numSites = markerSeqs.get(0).getSiteCount()
#         initNetHeights()
#         self.BAGTRModel = BAGTRModel

#         if(MHSettings.SAMPLE_SPLITTING) {
#             initSplitting()
#         } else {
#             self.splittings = None
#         }
#     }

#      UltrametricNetwork(String s) {
#         self.network = Networks.readNetworkWithRootPop(s)
#         initNetHeights()

#     }

    def getNetwork(self):
        return self.network
    

    def setNetwork(self, network):
        self.network = network
    

    def getReticulationCount(self):
        return self.network.getReticulationCount()
    

    def getInternalNodeCount(self):
        return 2 * self.network.getReticulationCount() + self.network.getLeafCount() - 1
    

    def isUltrametric(self):
        for node in Networks.postTraversal(self.network):
            height = node.getData().getHeight()
            if(node.isLeaf()):
                if(height != MHSettings.DEFAULT_NET_LEAF_HEIGHT):
                    #System.err.println(height + " vs " + MHSettings.DEFAULT_NET_LEAF_HEIGHT)
                    return False
                
            else:
                for child in node.getChildren():
                    temp = child.getData().getHeight() + child.getParentDistance(node)
                    if(math.abs(temp - height) > 0.000001):
                        #System.err.println(node.getName() + " - " + height + " vs " + temp)
                        return False
        return True
    

    #/************ State node methods ************/
    
    def propose(self):
        logHR = 0.0

        if(self.network.getReticulationCount() == 0):
            self.operator = getOp(self.operator, self.treeOpWeights)
        else:
            self.operator = getOp(self.operator, self.netOpWeights)
        
        logHR = self.operator.propose()

        if(MHSettings.SAMPLE_SPLITTING):
            ## experimental!
            Networks.autoLabelNodes(self.network)
            for i in range(len(self.splittings)):
                self.splittings.get(i).propose()
            
        

        if(SNAPPLikelihood.useApproximateBayesian):
            self.abcDataPrev = dict(self.abcData) 
            self.abcData = {}
        

        return logHR
    

    def undo(self):
        if(self.operator == None):
                #complain
                print("You done goofed") ##throw some exception

        self.operator.undo()

        if(MHSettings.SAMPLE_SPLITTING):
            ## experimental!
            for i in range(len(self.splittings)):
                self.splittings.get(i).undo()
            

        if(SNAPPLikelihood.useApproximateBayesian):
            self.abcData = dict(self.abcDataPrev)
            self.abcDataPrev = {}
            _logGeneTreeNetwork = self.computeLikelihood() ## approximate bayesian
   

    def logDensity(self):
        if(self.logGeneTreeNetwork == None):
            self.logLtemp = self.computeLikelihood()
            return MHSettings.sum(self.logLtemp)
        
        ## network changed
        if(self.dirty):
            if(self.network.getReticulationCount() > MHSettings._NET_MAX_RETI)
                return MHSettings.INVALID_MOVE
            else:
                self.logLtemp = self.computeLikelihood()
                return MHSettings.sum(self.logLtemp)
        else:
            self.logLtemp = MHSettings.copy(self.logGeneTreeNetwork)
        
        return MHSettings.sum(self.logLtemp)
    

    def recomputeLogDensity(self):
        return MHSettings.sum(self.computeLikelihood())
    

    def mayViolate(self):
        return self.operator.mayViolate()
    
    
    def accept(self):
        self.dirty = False
        if(self.logLtemp != None):
            self.logGeneTreeNetwork = self.logLtemp
        
        self.logLtemp = None

        if(MHSettings.SAMPLE_SPLITTING):
            for i in range(len(self.splittings)):
                self.splittings.get(i).accept()
        
        if(SNAPPLikelihood.useApproximateBayesian):
            self.abcDataPrev = {}
        
    

    def reject(self):
        self.dirty = False
        self.logLtemp = None

        if(MHSettings.SAMPLE_SPLITTING):
            for i in range(len(self.splittings)):
                self.splittings.get(i).reject()
           

    def isValid(self):
        if(self.network.getRoot().getData().getHeight() > MHSettings.NET_MAX_HEIGHT):
                return False

        if(MHSettings._Forbid_Net_Net):
            for nodeObj in self.network.bfs():
                node = nodeObj #cast?
                if(node.isNetworkNode()):
                    for parentObj in node.getParents():
                        parent = parentObj #cast??
                        if(parent.isNetworkNode()):
                            return False
       

        if(MHSettings._Forbid_Net_Triangle):
            for nodeObj in self.network.bfs():
                node = nodeObj
                if(node.isNetworkNode()):
                        ##wont work in python
                    it = node.getParents().iterator()
                    parent1 = it.next()
                    parent2 = it.next()
                    if(parent1.hasParent(parent2) or parent2.hasParent(parent1)):
                        return False
                    
        # /*Map<String, Double> constraints = TemporalConstraints.getTemporalConstraints(_geneTrees, self.species2alleles, self.alleles2species)
        # Map<String, Double> lowerBound = TemporalConstraints.getLowerBounds(self.network)
        # for(String key : lowerBound.keySet()) {
        #     if(constraints.get(key) < lowerBound.get(key)) {
        #         return False
        #     }
        # }*/
        return True
    

    #/************ moves *************/

    def getLowerAndUpperBound(self, node):
        bounds = [-math.inf, math.inf]
        for child in node.getChildren():
            bounds[0] = math.max(bounds[0], child.getData().getHeight())
        
        for par in node.getParents():
            bounds[1] = math.min(bounds[1], par.getData().getHeight())
        
        return bounds
    

    ## for debug only
    def getOldHeights(self):
        heights = []
        for node in Networks.postTraversal(self.network):
            heights.append(node.getData().getHeight())
        
        return heights
    

    #/************ Likelihood computation **************/

    def computeLikelihood(self):
        likelihoodArray = []
        ##likelihoodArray[0] = SNAPPLikelihood.computeSNAPPLikelihood(self.network, self.alleles2species, self.markers, self.BAGTRModel)

        if(not MHSettings.SAMPLE_SPLITTING):

            if (SNAPPLikelihood.usePseudoLikelihood):
                likelihoodArray[0] = SNAPPLikelihood.computeSNAPPPseudoLikelihood(self.network, self.alleles2species, self.markers, self.BAGTRModel) ## pseudo likelihood
            elif(SNAPPLikelihood.useApproximateBayesian):
                likelihoodArray[0] = SNAPPLikelihood.computeApproximateBayesian(self.network, self.alleles2species, self.markers, self.BAGTRModel, self.abcData) ## approximate bayesian
            else:
                likelihoodArray[0] = SNAPPLikelihood.computeSNAPPLikelihood(self.network, self.markers.get(0)._RPatterns, self.BAGTRModel) ## normal likelihood
        else:
            likelihoodArray[0] = SNAPPLikelihoodSampling.computeSNAPPLikelihoodST(self.network, self.splittings, self.markers.get(0)._RPatterns, self.BAGTRModel)
        
        return likelihoodArray
    

    #/************** init nodes **************/
    def initNetHeights(self, popSize, constraints):
        if(math.isnan(popSize)):
            self.initNetHeights2(constraints)
            return
        
        for node in Networks.postTraversal(self.network):
            if(node.isLeaf()):
                node.setData(NetNodeInfo(MHSettings.DEFAULT_NET_LEAF_HEIGHT))
                
            
            height = math.inf
            for child in node.getChildren():
                height = math.min(height, child.getParentDistance(node) + child.getData().getHeight())
            
            node.setData(NetNodeInfo(height))
        
        setPopSize = MHSettings.varyPopSizeAcrossBranches()
        for node in Networks.postTraversal(self.network):
            for par in node.getParents():
                node.setParentDistance(par, par.getData().getHeight() - node.getData().getHeight())
                if(node.getParentSupport(par) == node.NO_POP_SIZE and setPopSize):
                    node.setParentSupport(par, popSize)
                
        self.network.getRoot().setRootPopSize(popSize)
    

    def initNetHeights2(constraints):
        restrictedNodes = TemporalConstraints.getNodeRestriction(self.network)
        stack = stack() ## python stack implementation?
        stack.add([self.network.getRoot(), MHSettings.DEFAULT_NET_ROOT_HEIGHT])
        while(not stack.isEmpty()):
            tup = stack.pop()
            height = tup[1]
            if(restrictedNodes != None and restrictedNodes.containsKey(tup[0]) and constraints != None):
                if(tup[0].isNetworkNode()):
                        print("throw new RuntimeException('Invalid netwok node')")
                for key in restrictedNodes.get(tup[0]):
                    height = math.min(height, constraints.get(key))
                
            height *= MHSettings.NET_INTI_SCALE
            tup.Item1.setData(NetNodeInfo(height))
            for child in tup[0].getChildren():
                if(child.isLeaf()):
                    child.setData(NetNodeInfo(MHSettings.DEFAULT_NET_LEAF_HEIGHT))
                else:
                    stack.add([child, height])
                
        
        setPopSize = MHSettings.varyPopSizeAcrossBranches()
        for node in Networks.postTraversal(self.network):
            for par in node.getParents():
                node.setParentDistance(par, par.getData().getHeight() - node.getData().getHeight())
                if node.isNetworkNode():
                    node.setParentProbability(par, 0.5)
                
                if(node.getParentSupport(par) == node.NO_POP_SIZE and setPopSize): 
                    node.setParentSupport(par, MHSettings._POP_SIZE_MEAN)
                
        self.network.getRoot().setRootPopSize(MHSettings._POP_SIZE_MEAN)
    

    def initNetHeights3(self):
        for node in Networks.postTraversal(self.network):
            if(node.getData() == None):
                node.setData(NetNodeInfo(0.0))
            for par in node.getParents():
                dist = node.getParentDistance(par)
                if(par.getData() == None):
                    par.setData(NetNodeInfo(node.getData().getHeight() + dist))
             

    def initSplitting(self):
        self.splittings = []
        for pattern in self.markers.get(0)._RPatterns.keySet():
            count = self.markers.get(0)._RPatterns.get(pattern)[0]
            for k in range(count):
                self.splittings.append(Splitting(this, self.species2alleles, pattern, self.BAGTRModel)) #this
      

        if(len(self.splittings) != self.numSites):
            print("throw new RuntimeException('Number of splittings != number of sites.'")


    def updateWeights(self, arr, start):
        cutoff = arr[start-1]
        for i in range(len(arr)):
            arr[i] = (i < start)? 0 : (arr[i] - cutoff) / (1.0 - cutoff) ##wtf
        
        if(math.abs(arr[arr.length-1] - 1.0) > 0.0000001):
            print("throw new IllegalArgumentException(Arrays.toString(arr))")
