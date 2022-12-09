class TestNetworkBuilder():
    
    def test_simple_complete_tree(self):
        """
        This computes the likelihood of a complete nexus file, with no missing/gap characters
        in the data matrix. The nexus file contains a standard binary tree. Simplest possible working case.
        """
        assert True == True
        
    def test_network(self):
        """
        This is the case in which there is a network passed in through the nexus file.
        This is an error, since Felsenstein's algorithm does not support networks. Handle accordingly
        """ 
        assert True == True
    
    def test_gap_chars(self):
        """
        Pass in a nexus file in which there are characters in the data matrix that are the gap_chars, and ensure that the likelihood
        is computed appropriately.
        """
        assert True == True
    
    def test_match_chars(self):
        """
        Pass in a nexus file in which there are characters in the data matrix that are the match chars, and ensure that the likelihood
        is computed appropriately.
        """
        assert True == True
        
    def test_missing_chars(self):
        """
        Pass in a nexus file in which there are characters in the data matrix that are the missing chars ('?'), and ensure that the likelihood
        is computed appropriately.
        """
        assert True == True
    
    def test_malformed_nexus(self):
        """
        Pass in a malformed nexus file, and be sure to gracefully throw the correct error.
        """
        assert True == True