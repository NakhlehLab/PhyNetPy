from abc import abstractmethod, ABC


##############################################
### Abstract Base Class for Model Visitors ###
##############################################

class Visitor(ABC):
    
    @abstractmethod
    def visit_leaf(self, n : 'ModelNode') -> None:
        pass
    
    @abstractmethod
    def visit_internal(self, n : 'ModelNode') -> None:
        pass

    @abstractmethod
    def visit_reticulation(self, n : 'ModelNode') -> None:
        pass

    @abstractmethod
    def visit_root(self, n : 'ModelNode') -> None:
        pass
    
    def visit(self, n : 'ModelNode') -> None:
        """Dispatch to the correct visit method based on node type."""
        dispatch = {
            "leaf": self.visit_leaf,
            "internal": self.visit_internal,
            "root": self.visit_root,
            "reticulation": self.visit_reticulation,
        }
        return dispatch[n.get_node_type()](n)
    


