from distutils import cmd
import subprocess
from pathlib import Path

def runPhyloNet(*args):
        """
        given a list of command line args, run phylonet 
        and return the output of PhyloNet for that given command
        """

        script_dir = Path(__file__).parent
        jar_file = script_dir / "PhyloNet.jar"
        cmdLineArgs = ["java", "-jar", jar_file.absolute()]
        for arg in args:
                cmdLineArgs.append(arg)

        subprocess.call(cmdLineArgs)



#runPhyloNet("src/test/testNex.nexus")
