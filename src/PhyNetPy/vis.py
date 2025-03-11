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

from Network import *
from GraphUtils import *



class View:
    
    def __init__(self, net : Network) -> None:
        self.grid : dict[Node, tuple[float]] = {}
        self.net : Network = net
        self.n = len(self.net.get_leaves())
        self.x_step = 10 * self.n
        self.y_step = self.x_step * math.log2(2 * self.n)
    
    
    def _generate_grid_locations(self):
        
        #Step 0: Initialize array of leaves 
        sorted_leaves : list[Node] = []
        
        #Step 1: Compute clusters and then sort in ascending order by size
        clusters = get_all_clusters(self.net, self.net.root())
        sorted_clusters = sorted(clusters, key = lambda x: len(x))
        
        #Step 2: Starting with the smallest clusters, put the members next to 
        #        each other in the array.
        for cluster in sorted_clusters:
            for node in cluster:
                if node not in sorted_leaves:
                    pass
                
        #Step 3: Give grid coordinates to leaves
        for i in range(len(sorted_leaves)):
            x_coord = self.x_step * (i + 1)
            y_coord = self.y_step * (i + 1)
            self.grid[sorted_leaves[i]] = (x_coord, y_coord)




