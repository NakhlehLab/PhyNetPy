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
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0  

Docs   - [x]
Tests  - [ ]
Design - [x]

This module is a wrapper for the PhyloNet software. It is used to run PhyloNet
from within Python code.
"""

from subprocess import *

def jarWrapper(*args) -> list[str]:
    """
    Wrapper function for running PhyloNet from within Python code.
    
    Args:
        *args (tuple): List of arguments to pass to PhyloNet.
    Returns:
        list[str]: list of lines of output from PhyloNet.
    """
    process = Popen(['java', '-jar'] + list(args), stdout = PIPE, stderr = PIPE)
    ret = []
    
    while process.poll() is None:
        line = process.stdout.readline()
        if line != '' and line.endswith('\n'):
            ret.append(line[:-1])
            
    stdout, stderr = process.communicate()
    
    ret += stdout.split('\n')
    if stderr != '':
        ret += stderr.split('\n')
    ret.remove('')
    
    return ret

def run(nex_file_loc : str) -> list[str]:
    """
    Run PhyloNet on a nexus file.
    
    Args:
        nex_file_loc (str): Location of the nexus file to run PhyloNet on.
    Returns:
        list[str]: List of lines of output from PhyloNet.
    """
    return jarWrapper("PhyloNetv3_8_2.jar", nex_file_loc)