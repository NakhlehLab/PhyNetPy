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
##############################################################################

"""
Logger stub module for ModelMove compatibility.

Author: Mark Kessler
First Included in Version: 1.1.0
"""


class Logger:
    """
    Simple debug logger stub for ModelMove compatibility.
    
    This is a minimal implementation that satisfies the import requirements
    of ModelMove.py. Can be extended with actual logging functionality
    if needed.
    """
    
    def __init__(self, debug_id: str = None) -> None:
        """
        Initialize a Logger instance.
        
        Args:
            debug_id (str, optional): Identifier for debugging. Defaults to None.
        """
        self.debug_id = debug_id
        self.messages: list[str] = []
    
    def log(self, message: str) -> None:
        """
        Log a message (no-op by default).
        
        Args:
            message (str): The message to log.
        """
        self.messages.append(message)
    
    def debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message (str): The debug message.
        """
        self.log(f"[DEBUG] {message}")
    
    def info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message (str): The info message.
        """
        self.log(f"[INFO] {message}")
    
    def warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message (str): The warning message.
        """
        self.log(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message (str): The error message.
        """
        self.log(f"[ERROR] {message}")
    
    def get_messages(self) -> list[str]:
        """
        Get all logged messages.
        
        Returns:
            list[str]: List of logged messages.
        """
        return self.messages
    
    def clear(self) -> None:
        """Clear all logged messages."""
        self.messages = []








