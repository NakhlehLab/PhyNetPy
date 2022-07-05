class Sequence:

    def __init__(self, taxon, sequenceData):
        self.totalCount = -1
        self.taxon = taxon
        self.sequenceData = sequenceData
    

    def getSequence(self, dataType):
        sequence = dataType.stringToState(self.sequenceData)
        self.totalCount = dataType.getStateCount()
        return sequence
    

#     /**
#      * @return the taxon of this sequence as a string.
#      */
    def getTaxon(self):
        return self.taxon
    

#     /**
#      * @return the data of this sequence as a string.
#      */
    def getData(self):
        return self.sequenceData
    

#     /**
#      * @return return the state count
#      */
    def getStateCount(self):
        return self.totalCount
    

#     /**
#      * @return the sequence in the collection with the given taxon, or null if its not in the collection.
#      */
    def getSequenceByTaxon(self, taxon, sequences):
        for seq in sequences:
            if (seq.getTaxon().equals(taxon)):
                return seq
        return None
    

