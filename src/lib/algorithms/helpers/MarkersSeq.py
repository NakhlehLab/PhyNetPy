import math


class MarkerSeq:

    taxaNames = [] #sorted
    sequences = [] #corresponds to _taxaNames

    # state counts for each sequence
    stateCounts = []
    maxStateCount = -1
    # state codes for the sequences, a matrix of #taxa X #sites
    counts = []

    m_dataType = Nucleotide()

    # weights of sites, default value is 1
    siteWeights = None
    # weight over the columns of a matrix
    patternWeight = [] #?
    # Probabilities associated with leaves when the characters are uncertain
    tipLikelihoods = [] # #taxa X #sites X #states
    usingTipLikelihoods = False

    # pattern state encodings, a matrix of #patterns X #taxa
    sitePatterns = []
    # maps site nr to pattern nr, an array of #sites
    patternIndex = []

    # From AscertainedAlignment
    isAscertained = False
    excludedPatterns = set()
    excludefrom = 0 # first site to condition on
    excludeto = 0 # last site to condition on (but excluding this site
    excludeevery = 1 # interval between sites to condition on
    cache = {}
    dominant = None
    polyploid = None

    def __init__(self, sequences, name):
        
        self.name = name
        self.aln = sequences
        for key in sequences.keys():
            if (self.taxaNames.contains(key)):
                #throw new RuntimeException("Duplicate taxon found in alignment: " + key)
                print("oopsies")
            
            self.taxaNames.append(key)
        
        self.taxaNames.sort() #sorted in ascending order whatever that means for the taxaNames??

        for taxon in self.taxaNames:
            seq = Sequence(taxon, sequences[taxon]) ##TODO: sequence
            self.sequences.add(seq)
            self.counts.add(seq.getSequence(self.m_dataType))
            if(seq.getStateCount() == -1):
                #throw new RuntimeException("state count has not been initialized yet " + taxon + " " + sequences.get(taxon))
                print("oopsies")
            
            self.stateCounts.add(seq.getStateCount())
            self.maxStateCount = math.max(self.maxStateCount, seq.getStateCount())
        
        if (self.counts.size() == 0):
            #throw new RuntimeException("Sequence data expected, but none found")
            print("oopsies")
        
        #self.calcPatterns()
    

    

    def setCache(self, cache):
        self.cache = cache
        return

    def getCache(self):
        return self.cache

    def getAlignment(self):
        return self.aln

    def getName(self):
        return self.name 

    def setupAscertainment(self, src, to, every):
        self.isAscertained = True
        self.excludefrom = src
        self.excludeto = to
        self.excludeevery = every
        self.excludedPatterns = set()
        for i in range(src, to, every):
            patternIndex = self.patternIndex[i]
            # reduce weight, so it does not confuse the tree likelihood
            self.patternWeight[patternIndex] = 0
            self.excludedPatterns.append(patternIndex)
   


#     /**
#      * SiteComparator is used for ordering the sites,
#      * which makes it easy to identify patterns.
#      */
#     class SiteComparator implements Comparator<int[]> {
#         @Override
#         def compare(int[] o1, int[] o2) {
#             for (int i = 0 i < o1.length i++) {
#                 if (o1[i] > o2[i]) return 1
#                 if (o1[i] < o2[i]) return -1
#             }
#             return 0
#         }
#     }

#     /**
#      * calculate patterns from sequence data
#      */
    #def calcPatterns(self):
        # /*int taxonCount = _counts.size()
        # int siteCount = _counts.get(0).size()
        # # convert data to transposed int array
        # int[][] data = new int[siteCount][taxonCount]
        # for (int i = 0 i < taxonCount i++) {
        #     List<Integer> sites = _counts.get(i)
        #     for (int j = 0 j < siteCount j++) {
        #         data[j][i] = sites.get(j)
        #     }
        # }
        # # sort data
        # SiteComparator comparator = new SiteComparator()
        # Arrays.sort(data, comparator)
        # # count patterns in sorted data
        # int patterns = 1
        # int[] weights = new int[siteCount]
        # weights[0] = 1
        # for (int i = 1 i < siteCount i++) {
        #     if (_usingTipLikelihoods || comparator.compare(data[i - 1], data[i]) != 0) {
        #         # In the case where we're using tip probabilities, we need to treat each
        #         # site as a unique pattern, because it could have a unique probability vector.
        #         patterns++
        #         data[patterns - 1] = data[i]
        #     }
        #     weights[patterns - 1]++
        # }
        # # reserve memory for patterns
        # _patternWeight = new int[patterns]
        # _sitePatterns = new int[patterns][taxonCount]
        # for (int i = 0 i < patterns i++) {
        #     _patternWeight[i] = weights[i]
        #     _sitePatterns[i] = data[i]
        # }
        # # find patterns for the sites
        # _patternIndex = new int[siteCount]
        # for (int i = 0 i < siteCount i++) {
        #     int[] sites = new int[taxonCount]
        #     for (int j = 0 j < taxonCount j++) {
        #         sites[j] = _counts.get(j).get(i)
        #     }
        #     _patternIndex[i] = Arrays.binarySearch(_sitePatterns, sites, comparator)
        # }*/
    

    def getTaxaNames(self):
        return self.taxaNames
    

    def getStateCounts(self):
        return self.stateCounts
    

    def getDataType(self):
        return self.m_dataType
    

    def getCounts(self):
        return self.counts
    

    def getTaxonSize(self):
        return self.taxaNames.size()
    

    def getTaxonIndex(self, taxon):
        return self.taxaNames.indexOf(taxon) ##python equivalent?
    

    def getMaxStateCount(self):
        return self.maxStateCount
    

    def getStateSet(self, state):
        return self.m_dataType.getStateSet(state)
    

    def getPatternCount(self):
        return self.sitePatterns.length


    def getPattern(self, patternIndex):
        return self.sitePatterns[patternIndex]
    

    def getPattern(self, taxonIndex, patternIndex):
        return self.sitePatterns[patternIndex][taxonIndex]
    

    def getPatternWeight(self, patternIndex):
        return self.patternWeight[patternIndex]
    

    def getPatternIndex(self, site):
        return self.patternIndex[site]
    

    def getSiteCount(self):
        #return _patternIndex.length
        return self.aln.entrySet().iterator().next().getValue().length() #hmmmm
    

    def getPatternWeights(self):
        return self.patternWeight
    

    def getTipLikelihoods(self, taxon, idx):
        return None


    def getTotalWeight(self):
        return sum(self.patternWeight)
    


    #Methods from AscertainedAlignment ???
#     def getExcludedPatternIndices(self):
#         return self.excludedPatterns
    

#     def getExcludedPatternCount(self):
#         return self.excludedPatterns.size() ##?
    

#     def getAscertainmentCorrection(self, patternLogProbs):
#         excludeProb = 0
#         includeProb = 0
#         returnProb = 1.0
#         for pattern in self.excludedPatterns:
#             excludeProb += math.exp(patternLogProbs[pattern])
        
#         if (includeProb == 0.0):
#             returnProb -= excludeProb
#         elif(excludeProb == 0.0):
#             returnProb = includeProb
#         else:
#             returnProb = includeProb - excludeProb
        
#         return math.log(returnProb)
    


#     public static String getSequence(MarkerSeq data, int taxonIndex) {

#         int[] states = new int[data.getPatternCount()]
#         for (int i = 0 i < data.getPatternCount() i++) {
#             int[] sitePattern = data.getPattern(i)
#             states[i] = sitePattern[taxonIndex]
#         }
#         try {
#             return data.getDataType().stateToString(states)
#         } catch (Exception e) {
#             e.printStackTrace()
#             System.exit(1)
#         }
#         return null
#     }

#     # test
#     public static void main(String[] args) {
#         Map<String, String> input = new HashMap<>()
#         input.put("A", "ATATCG")
#         input.put("B", "ATATG-")
#         input.put("C", "CTAT-G")
#         MarkerSeq aln = new MarkerSeq(input)
#         System.out.println(aln.getTaxonSize() == 3)
#         List<String> taxa = aln.getTaxaNames()
#         System.out.println(taxa.get(0) == "A" && taxa.get(1) == "B" && taxa.get(2) == "C")
#         System.out.println(aln.getMaxStateCount() == 4)
#         System.out.println(aln.getPatternCount() == 5)
#         System.out.println(Arrays.toString(aln.getPatternWeights())) # 1,1,1,1,2
#         System.out.println(Arrays.toString(aln.getStateSet(0))) # TFFF
#         System.out.println(Arrays.toString(aln.getStateSet(17))) # TTTT
#         System.out.println(Arrays.toString(aln.getPattern(4))) # T T T -> 3, 3, 3
#     }

# }