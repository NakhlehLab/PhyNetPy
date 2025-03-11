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
Module that contains classes and functions that assist developers while using
PhyNetPy.

Release Version: 2.0.0

Author: Mark Kessler
"""

from __future__ import annotations
import base64
from io import BytesIO
from Network import *
from Newick import *
import webbrowser
import networkx as nx
from pyvis.network import Network as pyvisn
from lxml.html import builder as E
import lxml
import matplotlib.pyplot as plt
from networkx import Graph as xGraph



class Logger:
    """
    Class that logs method output in html formatting, for easily tracking 
    network status and construction at various states of development.
    
    In the future, all output of all types will be directed here and displayed.
    Formatting output helps debugging.
    """
    
    def __init__(self, id : int) -> None:
        """
        Initialize a logger instance with an integer ID value (on user to make
        it unique).

        Args:
            id (int): A unique id for this logger instance.
        Returns:    
            N/A
        """
        self.networkx_objs : list[xGraph] = []
        self.comments : list[str] = []
        self.id : int = id
        
    def log(self, net : Network, comment : str = "") -> None:
        """
        Log a network, and optionally attach a comment to it (such as iteration
        number, a network move that was applied to it, etc.)

        Args:
            net (Network): A Network
            comment (str, optional): An optional comment that
                                     assists to give program state context for
                                     the given Network. Defaults to "".
        Returns:
            N/A
        """
        self.networkx_objs.append(net.to_networkx())  
        self.comments.append(comment)
    
    def to_html(self) -> None:
        """
        Generate an html document containing all logged networks as viewables, 
        for viewing in a browser window.
        
        Args:
            N/A
        Returns:
            N/A
        """
        
        # to open/create a new html file in the write mode 
        f = open(f'PhyNetPy/src/PhyNetPy/Log-Output/logout{self.id}.html', 'w') 
        
        # Send each network to a pyvis object and generate a paragraph full
        # of html networks and related commentary.
        html_str = "<p>"
       
        for G, comment in zip(self.networkx_objs, self.comments):
            for layer, nodes in enumerate(nx.topological_generations(G)):
                # `multipartite_layout` expects the layer as a node attribute, so add the
                # numeric layer value as a node attribute
                for node in nodes:
                    G.nodes[node]["layer"] = layer
            
            # Compute the multipartite_layout using the "layer" node attribute
            pos = nx.multipartite_layout(G, subset_key="layer")

            fig, ax = plt.subplots()
            nx.draw_networkx(G, pos=pos, ax=ax)
            ax.set_title("DAG layout in topological order")
            fig.tight_layout()
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

            html_img = f'<img src=\'data:image/png;base64,{encoded}\'>'
            
            if comment is not None:
                html_str += comment
                html_str += " <br> "
            html_str += html_img
            html_str += "----------------------------------- <br>"
            
        
        html_str += "</p>"
            
        
        #Make html content
        html = E.HTML(
                  E.HEAD(
                    E.TITLE("--------Network Log Output--------")
                  ),
                  E.BODY(
                    E.P("Starting Logs:", style="font-size: 30pt;"),
                    lxml.html.fromstring(html_str), # type: ignore
                    E.P("----------End Log Output----------", 
                        style="font-size: 30pt;")
                  )
                )   
               
        # writing the code into the file 
        f.write(str(lxml.html.tostring(html))) # type: ignore
        
        # close the file 
        f.close() 
        
        # open html file 
        webbrowser.open(f'PhyNetPy/src/PhyNetPy/Log-Output/logout{self.id}.html', new = 1)
        