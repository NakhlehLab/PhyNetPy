#!/usr/bin/env python

import ast
import os
import sys

sys.path.insert(0, 'src/PhyNetPy')
import argparse
import PhyNetPy.Inference as I
 
# error messages
INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .txt file."
INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."
 
class MalformedArgsError(Exception):
    def __init__(self, message = "Args passed in are malformed"):
        self.message = message
        super().__init__(self.message)

class Help:
    
    def out(self):
        print("-----------------------")
        print("-----------------------")
        
        print("Welcome to PhyNetPy. This is a list of all possible commands:")
        
        print(" ")
        print("1) SNP_Likelihood")
        print("      ~For a description of flags and parameters for this command, use phynetpy.exe SNP_Likelihood -h")
        
        
        print("-----------------------")
        print("-----------------------")


 
def validate_file(file_name):
    '''
    validate file name and path.
    '''
    if not valid_path(file_name):
        print(INVALID_PATH_MSG%(file_name))
        quit()
    elif not valid_filetype(file_name):
        print(INVALID_FILETYPE_MSG%(file_name))
        quit()
    return
     
def valid_filetype(file_name):
    # validate file type
    return file_name.endswith('.nex') or file_name.endswith('.txt')
 
def valid_path(path):
    # validate file path
    return os.path.exists(path)
         
    
def SNP_Likelihood_Args(args):
    
    argmap = {}
    key : str
    flag_set = set(["-u", "-v", "-p", "-g", "-theta", "-a", "-f"])
    print(args)
    i = 0
    for sysarg in args:
        if sysarg in flag_set:
            if i%2 == 0:
                key = sysarg
            else:
                raise MalformedArgsError("Expected flag from ['-u', '-v', '-p', '-g', '-theta', '-a'] in this position")
        else:
            if i%2 == 1:
                argmap[key] = sysarg
            else:
                raise MalformedArgsError("Expected flag from ['-u', '-v', '-p', '-g', '-theta', '-a'] in this position")
        
        i+=1
    
    for flag in flag_set:
        if flag not in argmap.keys():
            argmap[flag] = None
        
    return argmap
       
    
    
     
if __name__ == "__main__":
    
    commands_list = ["SNP_Likelihood"]
    command_name = str(sys.argv[1])
    helpMenu = Help()
    
    if command_name in commands_list:
        if command_name == "SNP_Likelihood":
            args = SNP_Likelihood_Args(sys.argv[2:])
            validate_file(args["-f"])
            I.SNAPP_Likelihood(args["-f"], float(args["-u"]), float(args["-v"]), float(args["-theta"]), ast.literal_eval(args["-p"]), ast.literal_eval(args["-g"]), ast.literal_eval(args["-a"]))
                 
    elif command_name == "-h":
        helpMenu.out()
    else:
        print(f"Invalid command name <{command_name}>.")
        print(" ")
        helpMenu.out()