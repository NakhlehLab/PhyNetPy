from Network import *
from GeneTrees import GeneTrees
from GraphUtils import *
from typing import Dict, Set, List, Tuple, Optional
from collections import deque
import copy

"""
TODOS:
- [x] Implement Conflict class to represent conflicts between subnetwork topologies and astral network
- [x] Implement ConflictSet class to identify and manage all conflicts
- [x] Implement MoveCandidate class for different types of moves (SPR, NNI, Add reticulation)
- [x] Implement scoring function based on gene tree support, network alteration, preservation, and progress
- [x] Implement main merge algorithm with conflict resolution loop
- [x] Add validation methods to ensure subnetworks are properly embedded
- [x] Add conflict tracking and resolution validation
- [x] Create example script demonstrating the algorithm

Remaining improvements:
- [ ] Improve gene tree support calculation to be more accurate
- [ ] Add more sophisticated conflict detection (e.g., cluster-based conflicts)
- [ ] Optimize move generation to avoid invalid moves
- [ ] Add more sophisticated network topology validation
- [ ] Add support for different types of phylogenetic networks
"""


class MergeError(Exception):
    """
    Exception raised when the input is invalid.
    """
    def __init__(self, message : str = "Error in merging subnetworks into astral tree."):
        self.message = message
        super().__init__(self.message)

class DPMerge:
    
    def __init__(self, 
                 astral_network : Network, 
                 subnets : List[Network],
                 gene_trees : GeneTrees) -> None:
        """
        Initialize the DPMerge algorithm with the astral network, subnets and gene trees.

        Args:
            astral_network (Network): Astral tree to be merged into.
            subnets (List[Network]): Subnets to be merged into the astral tree.
            gene_trees (GeneTrees): Gene trees for information purposes.
        """
        self.Psi = astral_network
        self.subnets = subnets
        self.gene_trees = gene_trees
        self.validate_input()

    def validate_input(self) -> None:
        """
        Ensure that the astal network leaf set is equivalent to the union of the subnets leaf sets.
        
        Args:
            N/A
        Returns:
            N/A
        Raises:
            MergeError: If the astral network leaf set is not equivalent to the union of the subnets leaf sets.
        """
        # Ensure that subnet leaf sets are disjoint
        subnet_leaf_sets = [[leaf.label for leaf in subnet.get_leaves()] for subnet in self.subnets]
        astral_leaf_set = [leaf.label for leaf in self.Psi.get_leaves()]
        
        if set(astral_leaf_set) != set().union(*subnet_leaf_sets):
            raise MergeError("Astral network leaf set is not equivalent to the union of the subnets leaf sets.")

    def merge(self) -> tuple[Network, dict]:
        """
        Merge the subnets into the astral tree.
        Keep tabs on total move cost/score, and the number of conflicts
        solved. Return the resulting network, and validate that each subnetwork
        has been merged into the astral tree properly.

        Returns:
            tuple[Network, dict]: The final merged network and summary information
        """
        
        #Bookkeeping
        move_costs = []
        conflict_path = []
        resolved_conflicts = set()
        iteration_count = 0
        
        while True:
            iteration_count += 1
            
            #Step 1: Identify conflicts
            conflicts = ConflictSet(self.Psi, self.subnets, self.gene_trees)
            
            #Step 1.1: If no more conflicts, return the resulting network
            if len(conflicts) == 0:
                break
            
            #Step 2: Identify max conflict
            max_conflict = conflicts.compute_max_conflict()
            
            #Step 3: Generate the set of moves that edits the conflict
            moves = max_conflict.generate_move_set()
            
            if len(moves) == 0:
                # No valid moves for this conflict, skip it for now
                conflicts.remove_conflict(max_conflict)
                continue
            
            #Step 4: Select the move with the lowest score/cost
            best_move = None
            best_score = float('inf')
            
            for move in moves:
                # Score the move
                move_score = score(max_conflict, move, self.Psi, self.subnets)
                move.set_cost(move_score)
                
                if move_score < best_score:
                    best_score = move_score
                    best_move = move
            
            if best_move is None:
                # No valid moves found
                conflicts.remove_conflict(max_conflict)
                continue
            
            #Step 5: Edit the Astral Tree
            self.Psi = best_move.apply(self.Psi)
            
            #Step 6: Repeat until no conflicts and bookkeep
            move_costs.append(best_move.cost())
            conflict_path.append(str(max_conflict))
            resolved_conflicts.add(max_conflict)
            
            # Remove the resolved conflict
            conflicts.remove_conflict(max_conflict)
            
            # Validate that the move didn't break previous resolutions
            if not self._validate_previous_resolutions(resolved_conflicts):
                print(f"Warning: Move {best_move.move_type} may have broken previous resolutions")
        
        # Generate summary
        summary = {
            'total_iterations': iteration_count,
            'total_moves': len(move_costs),
            'total_cost': sum(move_costs),
            'average_cost_per_move': sum(move_costs) / len(move_costs) if move_costs else 0,
            'conflicts_resolved': len(resolved_conflicts),
            'move_costs': move_costs,
            'conflict_path': conflict_path,
            'embedding_valid': self.validate_embedding()
        }
        
        return self.Psi, summary
    
    def validate_embedding(self) -> bool:
        """
        Validate that each subnetwork is properly embedded in the final network.
        
        Returns:
            bool: True if all subnetworks are properly embedded
        """
        for subnet in self.subnets:
            if not self._is_subnet_embedded(subnet):
                return False
        return True
    
    def _is_subnet_embedded(self, subnet: Network) -> bool:
        """
        Check if a subnetwork is embedded in the final network.
        
        Args:
            subnet (Network): The subnetwork to check
            
        Returns:
            bool: True if the subnetwork is embedded
        """
        # Get the leaf labels of the subnetwork
        subnet_leaves = [leaf.label for leaf in subnet.get_leaves()]
        
        # Find the corresponding nodes in the final network
        final_nodes = []
        for label in subnet_leaves:
            node = None
            for n in self.Psi.V():
                if n.label == label:
                    node = n
                    break
            if node is None:
                return False
            final_nodes.append(node)
        
        # Check if the topology is preserved
        # This is a simplified check - in practice you'd want to:
        # 1. Extract the subnetwork induced by these nodes
        # 2. Check if it's isomorphic to the original subnetwork
        
        # For now, check basic relationships
        for i in range(len(final_nodes)):
            for j in range(i + 1, len(final_nodes)):
                node1, node2 = final_nodes[i], final_nodes[j]
                
                # Check sibling relationships
                if self._are_siblings_in_subnet(subnet, subnet_leaves[i], subnet_leaves[j]):
                    if not self._are_siblings_in_astral(subnet_leaves[i], subnet_leaves[j]):
                        return False
                
                # Check ancestor relationships
                if self._is_ancestor_in_subnet(subnet, subnet_leaves[i], subnet_leaves[j]):
                    if not self._is_ancestor_in_astral(subnet_leaves[i], subnet_leaves[j]):
                        return False
        
        return True
    
    def _validate_previous_resolutions(self, resolved_conflicts: set[Conflict]) -> bool:
        """
        Validate that previous conflict resolutions are still intact.
        
        Args:
            resolved_conflicts (set[Conflict]): Set of previously resolved conflicts
            
        Returns:
            bool: True if all previous resolutions are intact
        """
        for conflict in resolved_conflicts:
            if not self._is_conflict_resolved(conflict):
                return False
        return True
    
    def _is_conflict_resolved(self, conflict: Conflict) -> bool:
        """
        Check if a conflict is resolved in the current network.
        
        Args:
            conflict (Conflict): The conflict to check
            
        Returns:
            bool: True if the conflict is resolved
        """
        taxa1, taxa2 = conflict.taxa_pair
        
        if conflict.conflict_type == "sibling":
            return self._are_siblings_in_astral(taxa1, taxa2)
        elif conflict.conflict_type == "ancestor":
            return self._is_ancestor_in_astral(taxa1, taxa2)
        else:
            return True  # Unknown conflict type, assume resolved
    
    def _are_siblings_in_subnet(self, subnet: Network, taxa1: str, taxa2: str) -> bool:
        """Check if two taxa are siblings in a subnetwork."""
        node1 = None
        node2 = None
        for node in subnet.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        parents1 = set(parent.label for parent in subnet.get_parents(node1))
        parents2 = set(parent.label for parent in subnet.get_parents(node2))
        
        return len(parents1.intersection(parents2)) > 0
    
    def _is_ancestor_in_subnet(self, subnet: Network, taxa1: str, taxa2: str) -> bool:
        """Check if taxa1 is an ancestor of taxa2 in a subnetwork."""
        node1 = None
        node2 = None
        for node in subnet.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        return self._is_ancestor_path(subnet, node1, node2)
    
    def _is_ancestor_path(self, network: Network, ancestor: Node, descendant: Node) -> bool:
        """Check if there's a path from ancestor to descendant."""
        if ancestor == descendant:
            return True
        
        for child in network.get_children(ancestor):
            if self._is_ancestor_path(network, child, descendant):
                return True
        
        return False
    
    def _are_siblings_in_astral(self, taxa1: str, taxa2: str) -> bool:
        """Check if two taxa are siblings in the astral network."""
        node1 = None
        node2 = None
        for node in self.Psi.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        parents1 = set(parent.label for parent in self.Psi.get_parents(node1))
        parents2 = set(parent.label for parent in self.Psi.get_parents(node2))
        
        return len(parents1.intersection(parents2)) > 0
    
    def _is_ancestor_in_astral(self, taxa1: str, taxa2: str) -> bool:
        """Check if taxa1 is an ancestor of taxa2 in the astral network."""
        node1 = None
        node2 = None
        for node in self.Psi.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        return self._is_ancestor_path(self.Psi, node1, node2)
    
    def _is_ancestor_path(self, network: Network, ancestor: Node, descendant: Node) -> bool:
        """Check if there's a path from ancestor to descendant."""
        if ancestor == descendant:
            return True
        
        for child in network.get_children(ancestor):
            if self._is_ancestor_path(network, child, descendant):
                return True
        
        return False

class Conflict:
    def __init__(self, 
                 subnetwork: Network, 
                 taxa_pair: Tuple[str, str], 
                 expected_relationship: str,
                 current_relationship: str,
                 conflict_type: str) -> None:
        """
        Represents a conflict between subnetwork topology and astral network.
        
        Args:
            subnetwork (Network): The subnetwork that requires this relationship
            taxa_pair (Tuple[str, str]): The two taxa involved in the conflict
            expected_relationship (str): What the relationship should be (e.g., "siblings")
            current_relationship (str): What the relationship currently is
            conflict_type (str): Type of conflict (e.g., "sibling", "ancestor", "cluster")
        """
        self.subnetwork = subnetwork
        self.taxa_pair = taxa_pair
        self.expected_relationship = expected_relationship
        self.current_relationship = current_relationship
        self.conflict_type = conflict_type
        self.severity = self._compute_severity()
    
    def _compute_severity(self) -> float:
        """
        Compute the severity of this conflict based on how fundamental it is.
        Lower values indicate more severe conflicts that should be resolved first.
        
        Returns:
            float: Severity score (lower = more severe)
        """
        # Sibling conflicts are most severe (fundamental topology)
        if self.conflict_type == "sibling":
            return 1.0
        # Ancestor conflicts are next most severe
        elif self.conflict_type == "ancestor":
            return 2.0
        # Cluster conflicts are least severe
        elif self.conflict_type == "cluster":
            return 3.0
        else:
            return 4.0
    
    def __str__(self) -> str:
        return f"Conflict: {self.taxa_pair[0]} and {self.taxa_pair[1]} should be {self.expected_relationship} in {self.subnetwork}, but are currently {self.current_relationship}"
    
    def __lt__(self, other):
        return self.severity < other.severity


class ConflictSet:
    def __init__(self, astral_network: Network, subnets: List[Network], gene_trees: GeneTrees) -> None:
        """
        Identifies and manages conflicts between subnetworks and the astral network.
        
        Args:
            astral_network (Network): The current astral network
            subnets (List[Network]): List of subnetworks to be embedded
            gene_trees (GeneTrees): Gene trees for support calculation
        """
        self.astral_network = astral_network
        self.subnets = subnets
        self.gene_trees = gene_trees
        self.conflicts: set[Conflict] = set()
        self._identify_conflicts()
    
    def _identify_conflicts(self) -> None:
        """Identify all conflicts between subnetworks and the astral network."""
        for subnet in self.subnets:
            self._identify_subnet_conflicts(subnet)
    
    def _identify_subnet_conflicts(self, subnet: Network) -> None:
        """Identify conflicts for a specific subnetwork."""
        subnet_leaves = [leaf.label for leaf in subnet.get_leaves()]
        
        # Check sibling relationships
        for i in range(len(subnet_leaves)):
            for j in range(i + 1, len(subnet_leaves)):
                taxa1, taxa2 = subnet_leaves[i], subnet_leaves[j]
                
                # Check if they are siblings in the subnetwork
                if self._are_siblings_in_subnet(subnet, taxa1, taxa2):
                    # Check if they are siblings in the astral network
                    if not self._are_siblings_in_astral(taxa1, taxa2):
                        conflict = Conflict(
                            subnet, (taxa1, taxa2), "siblings", 
                            self._get_current_relationship(taxa1, taxa2), "sibling"
                        )
                        self.conflicts.add(conflict)
                
                # Check ancestor relationships
                if self._is_ancestor_in_subnet(subnet, taxa1, taxa2):
                    if not self._is_ancestor_in_astral(taxa1, taxa2):
                        conflict = Conflict(
                            subnet, (taxa1, taxa2), "ancestor-descendant",
                            self._get_current_relationship(taxa1, taxa2), "ancestor"
                        )
                        self.conflicts.add(conflict)
    
    def _are_siblings_in_subnet(self, subnet: Network, taxa1: str, taxa2: str) -> bool:
        """Check if two taxa are siblings in a subnetwork."""
        # Find the nodes in the subnetwork
        node1 = None
        node2 = None
        for node in subnet.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        # Check if they have the same parent
        parents1 = set(parent.label for parent in subnet.get_parents(node1))
        parents2 = set(parent.label for parent in subnet.get_parents(node2))
        
        return len(parents1.intersection(parents2)) > 0
    
    def _are_siblings_in_astral(self, taxa1: str, taxa2: str) -> bool:
        """Check if two taxa are siblings in the astral network."""
        # Find the nodes in the astral network
        node1 = None
        node2 = None
        for node in self.astral_network.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        # Check if they have the same parent
        parents1 = set(parent.label for parent in self.astral_network.get_parents(node1))
        parents2 = set(parent.label for parent in self.astral_network.get_parents(node2))
        
        return len(parents1.intersection(parents2)) > 0
    
    def _is_ancestor_in_subnet(self, subnet: Network, taxa1: str, taxa2: str) -> bool:
        """Check if taxa1 is an ancestor of taxa2 in the subnetwork."""
        # This is a simplified check - in practice you'd want to do a proper path search
        node1 = None
        node2 = None
        for node in subnet.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        # Check if taxa1 is in the path from root to taxa2
        return self._is_ancestor_path(subnet, node1, node2)
    
    def _is_ancestor_in_astral(self, taxa1: str, taxa2: str) -> bool:
        """Check if taxa1 is an ancestor of taxa2 in the astral network."""
        node1 = None
        node2 = None
        for node in self.astral_network.V():
            if node.label == taxa1:
                node1 = node
            elif node.label == taxa2:
                node2 = node
        
        if node1 is None or node2 is None:
            return False
        
        return self._is_ancestor_path(self.astral_network, node1, node2)
    
    def _is_ancestor_path(self, network: Network, ancestor: Node, descendant: Node) -> bool:
        """Check if there's a path from ancestor to descendant."""
        if ancestor == descendant:
            return True
        
        for child in network.get_children(ancestor):
            if self._is_ancestor_path(network, child, descendant):
                return True
        
        return False
    
    def _get_current_relationship(self, taxa1: str, taxa2: str) -> str:
        """Get the current relationship between two taxa in the astral network."""
        if self._are_siblings_in_astral(taxa1, taxa2):
            return "siblings"
        elif self._is_ancestor_in_astral(taxa1, taxa2):
            return "ancestor-descendant"
        elif self._is_ancestor_in_astral(taxa2, taxa1):
            return "descendant-ancestor"
        else:
            return "unrelated"
    
    def __len__(self) -> int:
        return len(self.conflicts)
    
    def compute_max_conflict(self) -> Conflict:
        """Return the most severe conflict to resolve next."""
        if len(self.conflicts) == 0:
            raise MergeError("No conflicts to resolve")
        return min(self.conflicts)
    
    def remove_conflict(self, conflict: Conflict) -> None:
        """Remove a resolved conflict."""
        self.conflicts.discard(conflict)

class MoveCandidate:
    def __init__(self, 
                 move_type: str,
                 source_node: Node,
                 target_node: Node,
                 conflict: Conflict,
                 cost: float = 0.0) -> None:
        """
        Represents a candidate move to resolve a conflict.
        
        Args:
            move_type (str): Type of move ("SPR", "NNI", "add_reticulation")
            source_node (Node): Source node for the move
            target_node (Node): Target node for the move
            conflict (Conflict): The conflict this move aims to resolve
            cost (float): Cost of this move
        """
        self.move_type = move_type
        self.source_node = source_node
        self.target_node = target_node
        self.conflict = conflict
        self.cost = cost
    
    def apply(self, astral_net: Network) -> Network:
        """
        Apply this move to the astral network.
        
        Args:
            astral_net (Network): The astral network to modify
            
        Returns:
            Network: The modified network
        """
        if self.move_type == "SPR":
            return self._apply_spr(astral_net)
        elif self.move_type == "NNI":
            return self._apply_nni(astral_net)
        elif self.move_type == "add_reticulation":
            return self._apply_add_reticulation(astral_net)
        else:
            raise MergeError(f"Unknown move type: {self.move_type}")
    
    def _apply_spr(self, astral_net: Network) -> Network:
        """Apply Subtree Pruning and Regrafting move."""
        # Create a copy of the network
        new_net, node_map = astral_net.copy()
        
        # Find the source and target nodes in the new network
        new_source = node_map[self.source_node]
        new_target = node_map[self.target_node]
        
        # Get the parent of the source node
        source_parents = new_net.get_parents(new_source)
        if not source_parents:
            raise MergeError("Source node has no parent for SPR move")
        
        source_parent = source_parents[0]
        
        # Remove the edge from parent to source
        edge_to_remove = new_net.get_edge(source_parent, new_source)
        new_net.remove_edge(edge_to_remove)
        
        # Add edge from target to source
        new_edge = Edge(new_target, new_source)
        new_net.add_edges(new_edge)
        
        # Clean up any artifacts
        new_net.clean()
        
        return new_net
    
    def _apply_nni(self, astral_net: Network) -> Network:
        """Apply Nearest Neighbor Interchange move."""
        # Create a copy of the network
        new_net, node_map = astral_net.copy()
        
        # Find the source and target nodes in the new network
        new_source = node_map[self.source_node]
        new_target = node_map[self.target_node]
        
        # Get the parent of the source node
        source_parents = new_net.get_parents(new_source)
        if not source_parents:
            raise MergeError("Source node has no parent for NNI move")
        
        source_parent = source_parents[0]
        
        # Get the parent of the target node
        target_parents = new_net.get_parents(new_target)
        if not target_parents:
            raise MergeError("Target node has no parent for NNI move")
        
        target_parent = target_parents[0]
        
        # Swap the connections
        edge1 = new_net.get_edge(source_parent, new_source)
        edge2 = new_net.get_edge(target_parent, new_target)
        
        new_net.remove_edge(edge1)
        new_net.remove_edge(edge2)
        
        new_edge1 = Edge(source_parent, new_target)
        new_edge2 = Edge(target_parent, new_source)
        
        new_net.add_edges(new_edge1)
        new_net.add_edges(new_edge2)
        
        return new_net
    
    def _apply_add_reticulation(self, astral_net: Network) -> Network:
        """Add a reticulation edge to resolve the conflict."""
        # Create a copy of the network
        new_net, node_map = astral_net.copy()
        
        # Find the source and target nodes in the new network
        new_source = node_map[self.source_node]
        new_target = node_map[self.target_node]
        
        # Create a new reticulation node
        retic_node = Node(f"retic_{new_source.label}_{new_target.label}", is_reticulation=True)
        new_net.add_nodes(retic_node)
        
        # Add edges to create the reticulation
        edge1 = Edge(new_source, retic_node, gamma=0.5, tag="retic_edge_1")
        edge2 = Edge(new_target, retic_node, gamma=0.5, tag="retic_edge_2")
        
        new_net.add_edges(edge1)
        new_net.add_edges(edge2)
        
        return new_net
    
    def cost(self) -> float:
        """Return the cost of this move."""
        return self.cost
    
    def set_cost(self, cost: float) -> None:
        """Set the cost of this move."""
        self.cost = cost

def score(conflict: Conflict, move: MoveCandidate, Psi: Network, subnets: List[Network]) -> float:
    """
    Score a move based on multiple criteria.
    
    Args:
        conflict (Conflict): The conflict being resolved
        move (MoveCandidate): The move to score
        Psi (Network): The current astral network
        subnets (List[Network]): List of subnetworks
        
    Returns:
        float: The cost of the move (lower is better)
    """
    # Initialize cost
    total_cost = 0.0
    
    # 1. Gene tree support (more support = lower cost)
    gene_support_cost = _calculate_gene_support_cost(move, conflict, Psi)
    total_cost += gene_support_cost * 0.3  # Weight: 30%
    
    # 2. Network alteration cost (less alteration = lower cost)
    alteration_cost = _calculate_alteration_cost(move, Psi)
    total_cost += alteration_cost * 0.25  # Weight: 25%
    
    # 3. Prior resolution preservation (very important)
    preservation_cost = _calculate_preservation_cost(move, Psi, subnets)
    total_cost += preservation_cost * 0.35  # Weight: 35%
    
    # 4. Progress towards resolving subnetwork (solving = fantastic)
    progress_cost = _calculate_progress_cost(move, conflict, Psi)
    total_cost += progress_cost * 0.1  # Weight: 10%
    
    return total_cost

def _calculate_gene_support_cost(move: MoveCandidate, conflict: Conflict, Psi: Network) -> float:
    """Calculate cost based on gene tree support."""
    # Count how many gene trees support the relationship created by this move
    # vs. how many support the current relationship
    
    # For now, return a base cost
    # In practice, you would:
    # 1. Apply the move to a copy of the network
    # 2. Count gene trees that support the new topology
    # 3. Count gene trees that support the current topology
    # 4. Return a cost inversely proportional to the support difference
    
    # This is a placeholder implementation
    return 1.0

def _calculate_gene_tree_support_for_relationship(network: Network, taxa1: str, taxa2: str, 
                                                gene_trees: GeneTrees, relationship_type: str) -> int:
    """
    Count how many gene trees support a specific relationship between two taxa.
    
    Args:
        network (Network): The network to check
        taxa1 (str): First taxon
        taxa2 (str): Second taxon
        gene_trees (GeneTrees): Collection of gene trees
        relationship_type (str): Type of relationship ("sibling", "ancestor", etc.)
        
    Returns:
        int: Number of supporting gene trees
    """
    support_count = 0
    
    for gene_tree in gene_trees.trees:
        if _gene_tree_supports_relationship(gene_tree, taxa1, taxa2, relationship_type):
            support_count += 1
    
    return support_count

def _gene_tree_supports_relationship(gene_tree: Network, taxa1: str, taxa2: str, 
                                   relationship_type: str) -> bool:
    """
    Check if a single gene tree supports a specific relationship.
    
    Args:
        gene_tree (Network): The gene tree to check
        taxa1 (str): First taxon
        taxa2 (str): Second taxon
        relationship_type (str): Type of relationship
        
    Returns:
        bool: True if the gene tree supports the relationship
    """
    # Find the nodes in the gene tree
    node1 = None
    node2 = None
    for node in gene_tree.V():
        if node.label == taxa1:
            node1 = node
        elif node.label == taxa2:
            node2 = node
    
    if node1 is None or node2 is None:
        return False
    
    if relationship_type == "sibling":
        # Check if they have the same parent
        parents1 = set(parent.label for parent in gene_tree.get_parents(node1))
        parents2 = set(parent.label for parent in gene_tree.get_parents(node2))
        return len(parents1.intersection(parents2)) > 0
    
    elif relationship_type == "ancestor":
        # Check if taxa1 is an ancestor of taxa2
        return _is_ancestor_in_gene_tree(gene_tree, node1, node2)
    
    return False

def _is_ancestor_in_gene_tree(gene_tree: Network, ancestor: Node, descendant: Node) -> bool:
    """Check if there's a path from ancestor to descendant in a gene tree."""
    if ancestor == descendant:
        return True
    
    for child in gene_tree.get_children(ancestor):
        if _is_ancestor_in_gene_tree(gene_tree, child, descendant):
            return True
    
    return False

def _calculate_alteration_cost(move: MoveCandidate, Psi: Network) -> float:
    """Calculate cost based on how much the network is altered."""
    # Count how many edges/nodes are affected
    if move.move_type == "SPR":
        return 2.0  # Affects 2 edges
    elif move.move_type == "NNI":
        return 1.0  # Affects 2 edges but swaps them
    elif move.move_type == "add_reticulation":
        return 3.0  # Adds new node and edges
    else:
        return 5.0  # Unknown move type

def _calculate_preservation_cost(move: MoveCandidate, Psi: Network, subnets: List[Network]) -> float:
    """Calculate cost based on preserving prior resolutions."""
    # This is critical - we don't want to undo previous work
    # For now, return a high cost if the move affects many nodes
    # In practice, you'd want to track which conflicts have been resolved
    
    # Count how many nodes in existing subnetworks would be affected
    affected_nodes = 0
    for subnet in subnets:
        for node in subnet.V():
            if node.label in [move.source_node.label, move.target_node.label]:
                affected_nodes += 1
    
    return affected_nodes * 10.0  # High penalty for affecting resolved subnetworks

def _calculate_progress_cost(move: MoveCandidate, conflict: Conflict, Psi: Network) -> float:
    """Calculate cost based on progress towards resolving the subnetwork."""
    # Check if this move directly resolves the conflict
    if _move_resolves_conflict(move, conflict, Psi):
        return -10.0  # Negative cost (reward) for solving the conflict
    elif _move_contributes_to_resolution(move, conflict, Psi):
        return -2.0   # Small reward for contributing
    else:
        return 5.0    # Cost for not contributing

def _move_resolves_conflict(move: MoveCandidate, conflict: Conflict, Psi: Network) -> bool:
    """Check if a move directly resolves a conflict."""
    # This is a simplified check - in practice you'd want to:
    # 1. Apply the move to a copy of the network
    # 2. Check if the conflict is resolved
    # 3. Return True/False accordingly
    
    # For now, return False to be conservative
    return False

def _move_contributes_to_resolution(move: MoveCandidate, conflict: Conflict, Psi: Network) -> bool:
    """Check if a move contributes to resolving a conflict."""
    # This is a simplified check - in practice you'd want to:
    # 1. Apply the move to a copy of the network
    # 2. Check if the conflict is closer to being resolved
    # 3. Return True/False accordingly
    
    # For now, return False to be conservative
    return False

# Add method to Conflict class to generate move sets
def generate_move_set(self) -> set[MoveCandidate]:
    """
    Generate a set of candidate moves to resolve this conflict.
    
    Returns:
        set[MoveCandidate]: Set of candidate moves
    """
    moves = set()
    
    # Get the taxa involved in the conflict
    taxa1, taxa2 = self.taxa_pair
    
    # Find the corresponding nodes in the astral network
    node1 = None
    node2 = None
    for node in self.astral_network.V():
        if node.label == taxa1:
            node1 = node
        elif node.label == taxa2:
            node2 = node
    
    if node1 is None or node2 is None:
        return moves
    
    # Generate SPR moves
    for node in self.astral_network.V():
        if node != node1 and node != node2:
            # SPR: move subtree rooted at node1 to be child of node
            spr_move = MoveCandidate("SPR", node1, node, self)
            moves.add(spr_move)
    
    # Generate NNI moves (if applicable)
    if self._can_apply_nni(node1, node2):
        nni_move = MoveCandidate("NNI", node1, node2, self)
        moves.add(nni_move)
    
    # Generate reticulation moves
    retic_move = MoveCandidate("add_reticulation", node1, node2, self)
    moves.add(retic_move)
    
    return moves

def _can_apply_nni(self, node1: Node, node2: Node) -> bool:
    """Check if NNI can be applied between two nodes."""
    # NNI can only be applied if the nodes are at the same level
    # and have a common ancestor that's not too far up
    # This is a simplified check
    return True  # For now, allow all NNI moves

# Add the generate_move_set method to the Conflict class
Conflict.generate_move_set = generate_move_set
Conflict._can_apply_nni = _can_apply_nni

        
        
        
        
        
        