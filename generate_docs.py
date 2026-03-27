#!/usr/bin/env python3
"""
PhyNetPy Documentation Generator

Parses all Python source files in src/ using the ast module and generates
static HTML documentation matching the existing Sphinx-like template.
"""

import ast
import os
import re
import textwrap
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
DOCS_DIR = SRC_DIR / "docs"

# Module metadata: descriptions, categories, version overrides
MODULE_META = {
    "Alphabet": {
        "desc": "Character-to-state mapping for biological sequence data (DNA, RNA, Protein, Codon, SNP).",
        "category": "Core Data Structures",
    },
    "BirthDeath": {
        "desc": "Birth-death process network simulators (Yule and CBDP models).",
        "category": "Simulation",
    },
    "BiMarkers": {
        "desc": "SNP (biallelic marker) likelihood computation for phylogenetic networks, with optional GPU acceleration.",
        "category": "Inference",
    },
    "Executor": {
        "desc": "Computation backend abstraction layer providing CPU (NumPy) and GPU (CuPy) array operations.",
        "category": "Infrastructure",
    },
    "GeneTrees": {
        "desc": "Gene tree container and analysis utilities including consensus tree construction and concordance factors.",
        "category": "Analysis",
    },
    "graph_core": {
        "desc": "Graph core data structures with automatic Cython acceleration for NodeSet and EdgeSet.",
        "category": "Infrastructure",
    },
    "GraphUtils": {
        "desc": "Graph and network utility functions for topology analysis, manipulation, and ASCII rendering.",
        "category": "Analysis",
    },
    "GTR": {
        "desc": "Time-reversible nucleotide substitution models (GTR, JC, K80, F81, HKY, K81, SYM, TN93).",
        "category": "Models",
    },
    "IO": {
        "desc": "Central I/O hub for reading and writing phylogenetic file formats (FASTA, VCF, Newick, Nexus).",
        "category": "I/O",
    },
    "Logger": {
        "desc": "Simple debug logger for internal model move compatibility.",
        "category": "Infrastructure",
    },
    "Matrix": {
        "desc": "Data matrix storage and reduction for sequence alignments with unique site pattern compression.",
        "category": "Core Data Structures",
    },
    "MetropolisHastings": {
        "desc": "Metropolis-Hastings MCMC and Hill Climbing search algorithms for phylogenetic inference.",
        "category": "Inference",
    },
    "ModelFactory": {
        "desc": "Component-based model building factory for constructing probabilistic phylogenetic models.",
        "category": "Infrastructure",
    },
    "ModelGraph": {
        "desc": "Probabilistic graphical model for phylogenetics with typed model nodes and visitor pattern support.",
        "category": "Infrastructure",
    },
    "ModelMove": {
        "desc": "Network topology move operations for MCMC search (add/remove/flip reticulation, SPR).",
        "category": "Inference",
    },
    "MSA": {
        "desc": "Multiple Sequence Alignment parsing, storage, grouping, and distance computation.",
        "category": "Core Data Structures",
    },
    "Network": {
        "desc": "Core phylogenetic network data structures: Node, Edge, and Network classes.",
        "category": "Core Data Structures",
    },
    "NetworkMoves": {
        "desc": "Network topology modification operations for MCMC search (add/remove hybrid, NNI, node height).",
        "category": "Inference",
    },
    "Newick": {
        "desc": "Newick format label extraction and Nexus file generation utilities.",
        "category": "I/O",
    },
    "Phylo": {
        "desc": "Core Branch class for storing phylogenetic edge attributes (length, inheritance probability).",
        "category": "Core Data Structures",
    },
    "PhyloNet": {
        "desc": "PhyloNet Java wrapper for running external phylogenetic analysis tools.",
        "category": "I/O",
    },
    "SNPSimulator": {
        "desc": "SNP data simulator for phylogenetic networks using a forward-in-time 2-state CTMC.",
        "category": "Simulation",
    },
    "State": {
        "desc": "State management for MCMC accept/reject decisions with model validation.",
        "category": "Inference",
    },
    "Strategy": {
        "desc": "Strategy pattern interface for node-level computations dispatched during bottom-up traversal.",
        "category": "Infrastructure",
    },
    "Traversal": {
        "desc": "Iterator-based graph traversal for model nodes supporting pre-order, post-order, and level-order.",
        "category": "Infrastructure",
    },
    "Validation": {
        "desc": "Comprehensive file format validation for phylogenetic data files (Newick, Nexus, FASTA, PHYLIP, etc.).",
        "category": "I/O",
    },
    "Visitor": {
        "desc": "Visitor pattern interface for ModelNode traversals with typed dispatch.",
        "category": "Infrastructure",
    },
}

CATEGORY_ORDER = [
    "Core Data Structures",
    "Models",
    "Simulation",
    "Inference",
    "Analysis",
    "I/O",
    "Infrastructure",
]

SKIP_MODULES = {"__init__"}


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_module_header(source: str) -> dict:
    """Extract Author, Last Edit, Version from module header comments/docstring."""
    info = {"author": "Mark Kessler", "last_edit": "", "version": ""}
    for line in source.split("\n")[:35]:
        if "Author" in line and ":" in line:
            info["author"] = line.split(":", 1)[1].strip()
        if "Last" in line and "Edit" in line and ":" in line:
            info["last_edit"] = line.split(":", 1)[1].strip()
        if "Version" in line and ":" in line and "First" not in line:
            info["version"] = line.split(":", 1)[1].strip()
        if "First Included in Version" in line and ":" in line:
            info["version"] = line.split(":", 1)[1].strip()
    return info


def parse_docstring(docstring: str | None) -> dict:
    """Parse a docstring into description, args, returns, raises sections."""
    if not docstring:
        return {"desc": "", "args": [], "returns": "", "raises": []}

    lines = textwrap.dedent(docstring).strip().split("\n")
    result = {"desc": "", "args": [], "returns": "", "raises": []}

    section = "desc"
    desc_lines = []
    current_arg = None
    current_raise = None

    for line in lines:
        stripped = line.strip()

        if stripped in ("Args:", "Arguments:", "Parameters:"):
            section = "args"
            continue
        elif stripped in ("Returns:", "Return:"):
            section = "returns"
            continue
        elif stripped in ("Raises:", "Raise:"):
            section = "raises"
            continue
        elif stripped == "N/A":
            continue

        if section == "desc":
            desc_lines.append(stripped)
        elif section == "args":
            arg_match = re.match(
                r"(\w+)\s*\(([^)]*)\)\s*[:,]\s*(.*)", stripped
            )
            arg_match2 = re.match(r"(\w+)\s*[:,]\s*(.*)", stripped)
            if arg_match:
                if current_arg:
                    result["args"].append(current_arg)
                current_arg = {
                    "name": arg_match.group(1),
                    "type": arg_match.group(2).strip(),
                    "desc": arg_match.group(3).strip(),
                }
            elif arg_match2 and not stripped.startswith(" "):
                if current_arg:
                    result["args"].append(current_arg)
                current_arg = {
                    "name": arg_match2.group(1),
                    "type": "",
                    "desc": arg_match2.group(2).strip(),
                }
            elif current_arg and stripped:
                current_arg["desc"] += " " + stripped
        elif section == "returns":
            ret_match = re.match(r"([^:]+):\s*(.*)", stripped)
            if ret_match and not result["returns"]:
                result["returns"] = f"{ret_match.group(1).strip()}: {ret_match.group(2).strip()}"
            elif stripped:
                if result["returns"]:
                    result["returns"] += " " + stripped
                else:
                    result["returns"] = stripped
        elif section == "raises":
            raise_match = re.match(r"(\w+Error|\w+Exception)\s*[:,]\s*(.*)", stripped)
            if raise_match:
                if current_raise:
                    result["raises"].append(current_raise)
                current_raise = {
                    "name": raise_match.group(1),
                    "desc": raise_match.group(2).strip(),
                }
            elif current_raise and stripped:
                current_raise["desc"] += " " + stripped

    if current_arg:
        result["args"].append(current_arg)
    if current_raise:
        result["raises"].append(current_raise)

    result["desc"] = " ".join(desc_lines).strip()
    # Clean up multi-space from wrapping
    result["desc"] = re.sub(r"\s+", " ", result["desc"])

    return result


def get_annotation_str(annotation) -> str:
    """Convert an AST annotation node to a readable string."""
    if annotation is None:
        return ""
    return ast.unparse(annotation)


def get_function_signature(node: ast.FunctionDef) -> str:
    """Build a human-readable function signature string."""
    args = node.args
    parts = []

    # Regular args
    all_args = args.args
    defaults = args.defaults
    num_no_default = len(all_args) - len(defaults)

    for i, arg in enumerate(all_args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        ann = get_annotation_str(arg.annotation)
        part = arg.arg
        if ann:
            part += f": {ann}"
        if i >= num_no_default:
            default = ast.unparse(defaults[i - num_no_default])
            part += f" = {default}"
        parts.append(part)

    # *args
    if args.vararg:
        ann = get_annotation_str(args.vararg.annotation)
        part = f"*{args.vararg.arg}"
        if ann:
            part += f": {ann}"
        parts.append(part)

    # keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        ann = get_annotation_str(arg.annotation)
        part = arg.arg
        if ann:
            part += f": {ann}"
        if args.kw_defaults[i] is not None:
            default = ast.unparse(args.kw_defaults[i])
            part += f" = {default}"
        parts.append(part)

    # **kwargs
    if args.kwarg:
        ann = get_annotation_str(args.kwarg.annotation)
        part = f"**{args.kwarg.arg}"
        if ann:
            part += f": {ann}"
        parts.append(part)

    return ", ".join(parts)


def get_return_annotation(node: ast.FunctionDef) -> str:
    if node.returns:
        return get_annotation_str(node.returns)
    return ""


def is_property(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "property":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "property":
            return True
    return False


def is_staticmethod(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "staticmethod":
            return True
    return False


def is_classmethod(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "classmethod":
            return True
    return False


def is_abstract(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
    return False


def extract_module_info(filepath: Path) -> dict:
    """Parse a Python file and extract all documentation-relevant information."""
    source = filepath.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    header = parse_module_header(source)
    module_doc = ast.get_docstring(tree) or ""

    classes = []
    functions = []
    constants = []
    exceptions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            cls_info = extract_class_info(node)
            if any(
                b
                for b in cls_info.get("bases", [])
                if "Exception" in b or "Error" in b
            ):
                exceptions.append(cls_info)
            else:
                classes.append(cls_info)
        elif isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            if not node.name.startswith("_"):
                functions.append(extract_function_info(node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                    val = ast.unparse(node.value) if node.value else ""
                    if len(val) > 120:
                        val = val[:120] + "..."
                    constants.append({"name": target.id, "value": val})
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper() and not node.target.id.startswith("_"):
                val = ast.unparse(node.value) if node.value else ""
                ann = get_annotation_str(node.annotation)
                if len(val) > 120:
                    val = val[:120] + "..."
                constants.append(
                    {"name": node.target.id, "value": val, "type": ann}
                )

    return {
        "header": header,
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
        "constants": constants,
        "exceptions": exceptions,
    }


def extract_class_info(node: ast.ClassDef) -> dict:
    bases = [ast.unparse(b) for b in node.bases]
    docstring = ast.get_docstring(node) or ""
    methods = []
    class_attrs = []
    properties = []

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("__") and item.name != "__init__":
                # Skip most dunder methods except __init__
                if item.name in ("__len__", "__iter__", "__contains__",
                                  "__getitem__", "__setitem__", "__eq__",
                                  "__hash__", "__str__", "__repr__"):
                    methods.append(extract_function_info(item))
                continue
            if item.name.startswith("_") and item.name != "__init__":
                continue
            if is_property(item):
                properties.append(extract_function_info(item))
            else:
                methods.append(extract_function_info(item))

    return {
        "name": node.name,
        "bases": bases,
        "docstring": docstring,
        "methods": methods,
        "properties": properties,
        "class_attrs": class_attrs,
    }


def extract_function_info(node: ast.FunctionDef) -> dict:
    docstring = ast.get_docstring(node) or ""
    sig = get_function_signature(node)
    ret = get_return_annotation(node)
    parsed = parse_docstring(docstring)

    decorators = []
    if is_property(node):
        decorators.append("property")
    if is_staticmethod(node):
        decorators.append("staticmethod")
    if is_classmethod(node):
        decorators.append("classmethod")
    if is_abstract(node):
        decorators.append("abstractmethod")

    return {
        "name": node.name,
        "signature": sig,
        "return_type": ret,
        "docstring": docstring,
        "parsed_doc": parsed,
        "decorators": decorators,
    }


# ──────────────────────────────────────────────────────────────
# HTML Generation
# ──────────────────────────────────────────────────────────────

def generate_sidebar(all_modules: list[str], current: str = "") -> str:
    items = []
    for mod in sorted(all_modules):
        cls = ' class="current"' if mod == current else ""
        items.append(f'                <li><a href="{mod}.html"{cls}>{mod}</a></li>')
    return "\n".join(items)


def render_param_table(args: list[dict]) -> str:
    if not args:
        return ""
    rows = []
    for a in args:
        name_esc = html_escape(a["name"])
        type_esc = html_escape(a.get("type", ""))
        desc_esc = html_escape(a.get("desc", ""))
        rows.append(f"""                                    <tr>
                                        <td><span class="param-name">{name_esc}</span></td>
                                        <td><span class="param-type">{type_esc}</span></td>
                                        <td>{desc_esc}</td>
                                    </tr>""")

    return f"""                                <table class="param-table">
                                    <tr>
                                        <th>Parameter</th>
                                        <th>Type</th>
                                        <th>Description</th>
                                    </tr>
{chr(10).join(rows)}
                                </table>"""


def render_function(func: dict, is_method: bool = False) -> str:
    name = html_escape(func["name"])
    sig = html_escape(func["signature"])
    ret = html_escape(func["return_type"])
    parsed = func["parsed_doc"]
    desc = html_escape(parsed["desc"])
    decs = func.get("decorators", [])

    dec_str = ""
    if "property" in decs:
        dec_str = ' <span class="keyword">property</span>'
    elif "staticmethod" in decs:
        dec_str = ' <span class="keyword">@staticmethod</span>'
    elif "classmethod" in decs:
        dec_str = ' <span class="keyword">@classmethod</span>'
    if "abstractmethod" in decs:
        dec_str += ' <span class="keyword">abstract</span>'

    ret_str = f' <span class="keyword">-&gt;</span> {ret}' if ret else ""

    has_sig = sig or func["name"] == "__init__"
    sig_display = f'(<span class="params">{sig}</span>)' if has_sig else ""

    html = f"""                        <div class="function-def">
                            <div class="function-header">
                                <span class="name">{name}</span>{sig_display}{ret_str}{dec_str}
                            </div>"""

    body_parts = []
    if desc:
        body_parts.append(f"<p>{desc}</p>")

    param_table = render_param_table(parsed["args"])
    if param_table:
        body_parts.append(param_table)

    if parsed["returns"]:
        ret_esc = html_escape(parsed["returns"])
        body_parts.append(
            f'<div class="returns"><span class="returns-label">Returns:</span> {ret_esc}</div>'
        )

    if parsed["raises"]:
        raises_items = ", ".join(
            f"<code>{html_escape(r['name'])}</code>: {html_escape(r['desc'])}"
            for r in parsed["raises"]
        )
        body_parts.append(
            f'<div class="raises"><span class="raises-label">Raises:</span> {raises_items}</div>'
        )

    if body_parts:
        body_content = "\n                                ".join(body_parts)
        html += f"""
                            <div class="function-body">
                                {body_content}
                            </div>"""

    html += """
                        </div>"""
    return html


def render_class(cls: dict) -> str:
    name = html_escape(cls["name"])
    bases = ", ".join(html_escape(b) for b in cls.get("bases", []))
    docstring = cls.get("docstring", "")
    parsed = parse_docstring(docstring)
    desc = html_escape(parsed["desc"])

    # Trim extremely long descriptions
    if len(desc) > 800:
        desc = desc[:800] + "..."

    bases_str = f'(<span class="params">{bases}</span>)' if bases else ""

    html = f"""                <div class="class-def">
                    <div class="class-header">
                        <span class="keyword">class</span> <span class="name">{name}</span>{bases_str}
                    </div>
                    <div class="class-body">
                        <p>{desc}</p>"""

    # Properties
    if cls.get("properties"):
        html += """
                        <h4>Properties</h4>"""
        for prop in cls["properties"]:
            html += "\n" + render_function(prop, is_method=True)

    # Constructor
    init_methods = [m for m in cls["methods"] if m["name"] == "__init__"]
    if init_methods:
        html += """
                        <h4>Constructor</h4>"""
        html += "\n" + render_function(init_methods[0], is_method=True)

    # Regular methods
    regular = [m for m in cls["methods"] if m["name"] != "__init__"]
    if regular:
        html += """
                        <h4>Methods</h4>"""
        for method in regular:
            html += "\n" + render_function(method, is_method=True)

    html += """
                    </div>
                </div>"""
    return html


def render_exception(exc: dict) -> str:
    name = html_escape(exc["name"])
    bases = ", ".join(html_escape(b) for b in exc.get("bases", []))
    docstring = exc.get("docstring", "")
    parsed = parse_docstring(docstring)
    desc = html_escape(parsed["desc"]) or f"Raised for {name}-related errors."

    return f"""                <div class="class-def exception-def">
                    <div class="class-header">
                        <span class="keyword">exception</span> <span class="name">{name}</span>(<span class="params">{bases}</span>)
                    </div>
                    <div class="class-body">
                        <p>{desc}</p>
                    </div>
                </div>"""


def generate_module_page(
    module_name: str, info: dict, all_modules: list[str]
) -> str:
    meta = MODULE_META.get(module_name, {})
    mod_desc = meta.get("desc", "")
    header = info["header"]

    # Build TOC
    toc_items = []
    if info["constants"]:
        toc_items.append(("constants", "Constants"))
    if info["exceptions"]:
        toc_items.append(("exceptions", "Exceptions"))
    for cls in info["classes"]:
        anchor = cls["name"].lower().replace(" ", "-")
        toc_items.append((anchor, cls["name"]))
    if info["functions"]:
        toc_items.append(("functions", "Module Functions"))

    toc_html = ""
    if toc_items:
        toc_links = "\n".join(
            f'                        <li><a href="#{a}">{t}</a></li>'
            for a, t in toc_items
        )
        toc_html = f"""
                <div class="toc">
                    <h4>Contents</h4>
                    <ul>
{toc_links}
                    </ul>
                </div>"""

    # Build main content sections
    content_sections = []

    # Constants
    if info["constants"]:
        const_html = '                <h2 id="constants">Constants</h2>\n'
        for c in info["constants"]:
            cname = html_escape(c["name"])
            cval = html_escape(c.get("value", ""))
            ctype = html_escape(c.get("type", ""))
            type_str = f' : <span class="param-type">{ctype}</span>' if ctype else ""
            const_html += f"""                <div class="constant-def">
                    <span class="constant-name">{cname}</span>{type_str} = {cval}
                </div>
"""
        content_sections.append(const_html)

    # Exceptions
    if info["exceptions"]:
        exc_html = '                <h2 id="exceptions">Exceptions</h2>\n'
        for exc in info["exceptions"]:
            exc_html += render_exception(exc) + "\n"
        content_sections.append(exc_html)

    # Classes
    for cls in info["classes"]:
        anchor = cls["name"].lower().replace(" ", "-")
        cls_html = f'                <h2 id="{anchor}">{html_escape(cls["name"])}</h2>\n'
        cls_html += render_class(cls) + "\n"
        content_sections.append(cls_html)

    # Module-level functions
    if info["functions"]:
        func_html = '                <h2 id="functions">Module Functions</h2>\n'
        for func in info["functions"]:
            func_html += render_function(func) + "\n"
        content_sections.append(func_html)

    body_content = "\n".join(content_sections)

    # Sidebar TOC for this page
    sidebar_this_page = ""
    if toc_items:
        this_links = "\n".join(
            f'                <li><a href="#{a}">{t}</a></li>'
            for a, t in toc_items
        )
        sidebar_this_page = f"""
            <h3>This Page</h3>
            <ul>
{this_links}
            </ul>"""

    sidebar_modules = generate_sidebar(all_modules, module_name)
    version = header.get("version", "") or "1.0.0"
    last_edit = header.get("last_edit", "")
    author = header.get("author", "Mark Kessler")

    module_info_html = f"""
                <div class="module-info">
                    <dl>
                        <dt>Author:</dt>
                        <dd>{html_escape(author)}</dd>"""
    if last_edit:
        module_info_html += f"""
                        <dt>Last Edit:</dt>
                        <dd>{html_escape(last_edit)}</dd>"""
    module_info_html += f"""
                        <dt>Source:</dt>
                        <dd>{module_name}.py</dd>
                    </dl>
                </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{module_name} - PhyNetPy Documentation</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="header">
        <h1>PhyNetPy Documentation</h1>
        <p>Library for the Development and Use of Phylogenetic Network Methods</p>
    </div>
    
    <div class="document">
        <div class="documentwrapper">
            <div class="body">
                <div class="breadcrumb">
                    <a href="index.html">Home</a> &raquo; {module_name}
                </div>
                
                <h1>{module_name} Module <span class="version-badge">v{html_escape(version)}</span></h1>
                
                <p>{html_escape(mod_desc)}</p>
                {module_info_html}
                {toc_html}

{body_content}
                
                <div class="footer">
                    <p>PhyNetPy Documentation - Copyright 2025 Mark Kessler, Luay Nakhleh</p>
                </div>
            </div>
        </div>
        
        <div class="sphinxsidebar">
            <h3>Navigation</h3>
            <ul>
                <li><a href="index.html">Home</a></li>
            </ul>
            
            <h3>Modules</h3>
            <ul>
{sidebar_modules}
            </ul>
            {sidebar_this_page}
        </div>
    </div>
</body>
</html>
"""


def generate_index_page(all_modules: list[str]) -> str:
    # Group modules by category
    categories: dict[str, list[tuple[str, str]]] = {}
    for cat in CATEGORY_ORDER:
        categories[cat] = []

    for mod in sorted(all_modules):
        meta = MODULE_META.get(mod, {})
        cat = meta.get("category", "Infrastructure")
        desc = meta.get("desc", "")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((mod, desc))

    # Module index table rows
    table_rows = []
    for mod in sorted(all_modules):
        meta = MODULE_META.get(mod, {})
        desc = html_escape(meta.get("desc", ""))
        table_rows.append(
            f"""                        <tr>
                            <td><a href="{mod}.html">{mod}</a></td>
                            <td>{desc}</td>
                        </tr>"""
        )

    # Category sections
    cat_sections = []
    for cat in CATEGORY_ORDER:
        mods = categories.get(cat, [])
        if not mods:
            continue
        items = "\n".join(
            f'                    <li><a href="{m}.html">{m}</a> - {html_escape(d)}</li>'
            for m, d in mods
        )
        cat_sections.append(
            f"""                <h3>{cat}</h3>
                <ul class="method-list">
{items}
                </ul>"""
        )

    sidebar_modules = generate_sidebar(all_modules)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhyNetPy Documentation</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="header">
        <h1>PhyNetPy Documentation</h1>
        <p>Library for the Development and Use of Phylogenetic Network Methods</p>
    </div>
    
    <div class="document">
        <div class="documentwrapper">
            <div class="body">
                <h1>PhyNetPy API Reference</h1>
                
                <p>Welcome to the PhyNetPy documentation. PhyNetPy is a comprehensive Python library 
                for phylogenetic network analysis, providing tools for network construction, manipulation, 
                simulation, and inference.</p>
                
                <div class="module-info">
                    <dl>
                        <dt>Version:</dt>
                        <dd>0.3.0</dd>
                        <dt>Authors:</dt>
                        <dd>Mark Kessler, Luay Nakhleh</dd>
                        <dt>Copyright:</dt>
                        <dd>2025</dd>
                    </dl>
                </div>
                
                <h2>Module Index</h2>
                
                <table class="module-index">
                    <thead>
                        <tr>
                            <th>Module</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
{chr(10).join(table_rows)}
                    </tbody>
                </table>
                
                <h2>Getting Started</h2>
                
                <h3>Installation</h3>
                <pre><code>pip install phynetpy</code></pre>
                
                <h3>Quick Example</h3>
                <pre><code>from PhyNetPy import Network, Node, Edge
from PhyNetPy import read_nexus, read_newick

# Parse a network from a Nexus file
networks = read_nexus("my_network.nex")
network = networks[0]

# Access network properties
print(f"Number of nodes: {{len(network.V())}}")
print(f"Number of edges: {{len(network.E())}}")
print(f"Leaves: {{[leaf.label for leaf in network.get_leaves()]}}")

# Parse from a Newick string
net = read_newick("((A:0.1,B:0.2):0.3,C:0.4);")

# Simulate a network
from PhyNetPy import CBDP
sim = CBDP(gamma=1.0, mu=0.5, n=10)
simulated_net = sim.generate_network()</code></pre>
                
                <h2>Module Categories</h2>
                
{chr(10).join(cat_sections)}
                
                <div class="footer">
                    <p>PhyNetPy Documentation - Copyright 2025 Mark Kessler, Luay Nakhleh</p>
                </div>
            </div>
        </div>
        
        <div class="sphinxsidebar">
            <h3>Navigation</h3>
            <ul>
                <li><a href="index.html" class="current">Home</a></li>
            </ul>
            
            <h3>Modules</h3>
            <ul>
{sidebar_modules}
            </ul>
        </div>
    </div>
</body>
</html>
"""


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    py_files = sorted(SRC_DIR.glob("*.py"))
    all_modules = []
    module_infos = {}

    for pyfile in py_files:
        mod_name = pyfile.stem
        if mod_name in SKIP_MODULES:
            continue

        print(f"  Parsing {mod_name}.py ...")
        info = extract_module_info(pyfile)
        if info is None:
            print(f"    SKIPPED (syntax error)")
            continue

        all_modules.append(mod_name)
        module_infos[mod_name] = info

    print(f"\nFound {len(all_modules)} modules. Generating HTML...\n")

    for mod_name, info in module_infos.items():
        html = generate_module_page(mod_name, info, all_modules)
        out_path = DOCS_DIR / f"{mod_name}.html"
        out_path.write_text(html, encoding="utf-8")
        n_classes = len(info["classes"])
        n_funcs = len(info["functions"])
        n_exc = len(info["exceptions"])
        print(f"  {mod_name}.html  ({n_classes} classes, {n_funcs} functions, {n_exc} exceptions)")

    # Generate index
    index_html = generate_index_page(all_modules)
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(f"\n  index.html (module index)")

    print(f"\nDone! Generated {len(all_modules) + 1} HTML files in {DOCS_DIR}")


if __name__ == "__main__":
    main()
