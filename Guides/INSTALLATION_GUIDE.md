# PhyNetPy – Installation & Development Setup Guide
This guide explains how to:
    Create and activate a Python virtual environment
    Set it up in VS Code and PyCharm
    Install PhyNetPy using pip install phynetpy

## 1. Create a Virtual Environment (Recommended)
#### macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
#### Windows (PowerShell)
    python -m venv .venv
    .\.venv\Scripts\Activate

You should now see (.venv) in your terminal prompt.

## 2. Installing PhyNetPy
Once the virtual environment is active:
    pip install --upgrade pip
    pip install phynetpy

PhyNetPy 0.6.0 supports Python 3.9 through 3.14 and requires compiled Cython
extensions. `pip` uses a platform wheel when one is published. Installing from
an sdist or source checkout requires a working C compiler (MSVC Build Tools on
Windows, Xcode Command Line Tools on macOS, or GCC/Clang on Linux).

Verify installation:
    python -c "import phynetpy; print(phynetpy.__version__)"
## VS Code Setup
1. Open the Project Folder
Open VS Code → File → Open Folder… → select your project folder.

2. Select the Virtual Environment Interpreter
Press Ctrl/Cmd + Shift + P
Type Python: Select Interpreter
Choose the interpreter located at:
#### macOS/Linux: <project>/.venv/bin/python
#### Windows: <project>\.venv\Scripts\python.exe
VS Code will now automatically use this environment for:
Running files
Jupyter notebooks
Integrated terminal (if set to activate)
3. (Optional) Auto-activate venv in VS Code terminal
Add this to .vscode/settings.json:
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.terminal.activateEnvironment": true
}
Windows path would be:
"python.defaultInterpreterPath": ".venv\\Scripts\\python.exe"
 

## PyCharm Setup
1. Open Your Project in PyCharm
2. Configure Interpreter
Go to PyCharm → Settings → Project → Python Interpreter
Click the gear icon → Add Interpreter
Choose Existing Environment
Select your .venv:
#### macOS/Linux: <project>/.venv/bin/python
#### Windows: <project>\.venv\Scripts\python.exe
PyCharm will detect dependencies and index the environment.
3. Install PhyNetPy via PyCharm (optional)
PyCharm → Python Packages → search phynetpy → Install
OR install via terminal inside PyCharm:
pip install phynetpy

## Example Usage
After installing:
    from phynetpy import read_newick
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.data import GeneTrees
    from phynetpy.infer import infer

    gene_trees = GeneTrees.from_newick(["((A,B),C);"])
    result = infer(gene_trees, criterion=PseudoLikelihood())
    print(result.best)
