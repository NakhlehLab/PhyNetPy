from abc import abstractmethod, ABC


##############################################
### Abstract Base Class for Model Visitors ###
##############################################

class Strategy(ABC):
    
    @abstractmethod
    def compute_at_leaf(self, n : 'ModelNode') -> None:
        pass
    
    @abstractmethod
    def compute_at_internal(self, n : 'ModelNode') -> None:
        pass

    @abstractmethod
    def compute_at_reticulation(self, n : 'ModelNode') -> None:
        pass

    @abstractmethod
    def compute_at_root(self, n : 'ModelNode') -> None:
        pass
    
    
    