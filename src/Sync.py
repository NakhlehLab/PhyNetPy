#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --                                                              
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

""" 
Author : Mark Kessler
Last Stable Edit : 3/20/26
First Included in Version : 0.3.2

Docs   - [ ]
Tests  - [ ]
Design - [ ]

Example Usage:

my_model : Model

with Sync(my_model):
    my_model.execute_move(nni())
    my_model.execute_move(spr())
    my_model.execute_move(add_reticulation())
    my_model.execute_move(gene_flow_swap())

#Can assume now that my_model.network shares the exact same topology as the ModelNodes that make up the model

"""
from .ModelGraph import *

class Sync:
    """Atomic model / network reconciliation context manager.

    ``Sync`` wraps a sequence of topology moves on a :class:`~.ModelGraph.Model`
    so that they appear atomic with respect to both the ``Model`` and its
    underlying :class:`~.Network.Network`:

    * On normal exit the network is **reconciled** -- nodes and edges in
      ``model.network`` are rebuilt from the current ``ModelNode`` topology so
      that the two representations agree.
    * On exception the model and network are **rolled back** to the snapshot
      captured at ``__enter__``, leaving the caller's state untouched.

    Typical use::

        with Sync(my_model):
            my_model.execute_move(nni())
            my_model.execute_move(spr())
            my_model.execute_move(add_reticulation())
        # after the block, my_model.network reflects all three moves
        # (or none of them, if one raised)
    """

    def __init__(self, model : Model) -> None:
        """Create a synchroniser bound to ``model``.

        Args:
            model (Model): The model whose ``ModelNode`` topology and
                ``network`` attribute should be kept in lock-step on exit.
        """
        self.net = model.network
        self.model = model

    def __enter__(self):
        """
        You may assume that as you enter the with flow, that the ModelNode topology is in sync with the
        network that the model has a reference to.
        """
        backup, old_to_new = self.net.copy()
        self._rollback_net = backup
        self._rollback_map = {old_to_new[n]: m for n, m in self.model.network_node_map.items()}
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """
        If there are any errors executing a move within the block, the model and network should be unchanged 
        from the start of the with block (in other words, moves within the with block are "atomic" with respect
        to how the user experiences it)

        The easiest way to implement this is to let the model make moves, and then at the end, use the node mapping 
        to reconcile the network. 

        If there are nodes that no longer exist in the modelgraph that used to exist in the network, it is safe to delete
        the old node in the network.

        If there are new nodes, then a node should be added to the network.
        """
        if exc_type is not None:
            self.model.network = self._rollback_net
            self.model.network_node_map = dict(self._rollback_map)
            self.net = self._rollback_net
            return False
        self._reconcile()
        return False

    def _reconcile(self):
        """
        This function should rebuild self.net so that self.net and the ModelNode structure agree 
        and contain all relevant stored data.
        
        Args:
            N/A
        Returns:
            N/A (alters self.net)
        """
        model = self.model
        net = self.net
        for n in list(net.V()):
            net.remove_nodes(n)
        model.network_node_map.clear()

        phylo: list[ModelNode] = []
        for key in ("root", "internal", "reticulation", "leaf"):
            phylo.extend(model.nodetypes.get(key, []))
        phylo = list(dict.fromkeys(phylo))

        mn_to_nn: dict[ModelNode, Node] = {}
        for mn in phylo:
            nn = Node(mn.get_name(), is_reticulation=isinstance(mn, ReticulationNode))
            mn_to_nn[mn] = nn
            net.add_nodes(nn)
            model.network_node_map[nn] = mn

        for mn in phylo:
            for par in (mn.parents or []):
                if par not in mn_to_nn:
                    continue
                src = mn_to_nn[par]
                dest = mn_to_nn[mn]
                if isinstance(mn, ReticulationNode):
                    b1, b2 = mn.branches()
                    if b1.parent_id == par.get_name():
                        br = b1
                    elif b2.parent_id == par.get_name():
                        br = b2
                    else:
                        br = b1
                    net.add_edges(Edge(src, dest, length=br.length, gamma=br.inheritance_probability))
                elif isinstance(mn, (LeafNode, InternalNode)):
                    br = mn.branch()
                    net.add_edges(Edge(src, dest, length=br.length, gamma=br.inheritance_probability))
                else:
                    net.add_edges(Edge(src, dest))
        model.update_network()