#!/usr/bin/env python3
"""
Script to restructure gene copy data by mapping genes to species.
This converts a Nexus file with gene copies to a Nexus file with species-level taxa.
"""

import os
import sys
import re
from collections import defaultdict
import tempfile

def parse_gene_to_species_mapping(taxon_names):
    """
    Parse gene names to extract species information.
    Based on the naming pattern in your data.
    
    Args:
        taxon_names (list): List of gene copy names
        
    Returns:
        dict: Mapping from gene name to species name
    """
    gene_to_species = {}
    
    for gene_name in taxon_names:
        # Extract species name from gene name
        # Pattern: RSC01_1, RSC02_3, RS244, etc.
        if gene_name.startswith('RSC'):
            # RSC01_1 -> RSC01
            species = gene_name.split('_')[0]
        elif gene_name.startswith('RS'):
            # RS244 -> RS244 (single gene per species)
            species = gene_name
        elif gene_name in ['Helia', 'Senec']:
            # These appear to be species names already
            species = gene_name
        else:
            # Default: use the gene name as species name
            species = gene_name
        
        gene_to_species[gene_name] = species
    
    return gene_to_species

def extract_leaf_labels_from_newick(newick_str):
    """
    Extract all unique leaf labels from a Newick string.
    Args:
        newick_str (str): The Newick string
    Returns:
        set: Set of leaf labels
    """
    # Remove branch lengths and comments
    newick_str = re.sub(r':[0-9.eE+-]+', '', newick_str)
    newick_str = re.sub(r'\[.*?\]', '', newick_str)
    # Find all words (taxon labels)
    tokens = re.findall(r'([A-Za-z0-9_]+)', newick_str)
    # Remove internal node labels (those after ')')
    # Only keep tokens that are not immediately after ')'
    leaves = set()
    prev = ''
    for match in re.finditer(r'([(),;])|([A-Za-z0-9_]+)', newick_str):
        if match.group(2):  # It's a label
            if prev != ')':
                leaves.add(match.group(2))
        if match.group(1):
            prev = match.group(1)
    return leaves

def restructure_nexus_file(input_file, output_file):
    """
    Restructure a Nexus file from gene copies to species-level taxa.
    
    Args:
        input_file (str): Path to input Nexus file with gene copies
        output_file (str): Path to output Nexus file with species-level taxa
    """
    print(f"Restructuring {input_file} -> {output_file}")
    
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Extract tree section
    trees_start = content.find('BEGIN TREES;')
    trees_end = content.find('END;', trees_start)
    
    if trees_start == -1 or trees_end == -1:
        print("ERROR: Could not find TREES block in Nexus file")
        return False
    
    trees_section = content[trees_start:trees_end]
    
    # Instead of extracting all words, extract only leaf labels from Newick strings
    all_genes = set()
    tree_lines = trees_section.split('\n')
    for line in tree_lines:
        if line.strip().startswith('Tree'):
            tree_start = line.find('(')
            tree_end = line.rfind(')') + 1
            if tree_start != -1 and tree_end != -1:
                tree_str = line[tree_start:tree_end]
                leaves = extract_leaf_labels_from_newick(tree_str)
                all_genes.update(leaves)
    taxon_names = sorted(list(all_genes))
    print(f"Found {len(taxon_names)} gene copies: {taxon_names}")
    
    # Create gene to species mapping
    gene_to_species = parse_gene_to_species_mapping(taxon_names)
    
    # Get unique species
    species_set = set(gene_to_species.values())
    species_list = sorted(list(species_set))
    print(f"Mapped to {len(species_list)} species: {species_list}")
    
    # Show the mapping
    print("\nGene to Species Mapping:")
    for gene, species in sorted(gene_to_species.items()):
        print(f"  {gene} -> {species}")
    
    # Create species to genes mapping
    species_to_genes = defaultdict(list)
    for gene, species in gene_to_species.items():
        species_to_genes[species].append(gene)
    
    print(f"\nSpecies to Genes Mapping:")
    for species, genes in sorted(species_to_genes.items()):
        print(f"  {species}: {genes}")
    
    # Restructure trees
    restructured_trees = []
    tree_lines = trees_section.split('\n')
    
    for line in tree_lines:
        if line.strip().startswith('Tree'):
            # Extract tree string
            tree_start = line.find('(')
            tree_end = line.rfind(')') + 1
            if tree_start != -1 and tree_end != -1:
                tree_str = line[tree_start:tree_end]
                
                # Replace gene names with species names
                restructured_tree_str = tree_str
                for gene, species in gene_to_species.items():
                    # Use word boundaries to avoid partial matches
                    restructured_tree_str = re.sub(r'\b' + re.escape(gene) + r'\b', species, restructured_tree_str)
                
                # Create new tree line
                tree_name = line.split('=')[0].strip()
                new_line = f"{tree_name} = {restructured_tree_str};"
                restructured_trees.append(new_line)
    
    # Write new Nexus file
    with open(output_file, 'w') as f:
        f.write('#NEXUS\n\n')
        f.write('BEGIN TAXA;\n')
        f.write(f'\tDIMENSIONS NTAX={len(species_list)};\n')
        f.write('\tTAXLABELS\n')
        for species in species_list:
            f.write(f'\t\t{species}\n')
        f.write('\t;\n')
        f.write('END;\n\n')
        
        f.write('BEGIN TREES;\n')
        for tree_line in restructured_trees:
            f.write(f'{tree_line}\n')
        f.write('END;\n')
    
    print(f"\nRestructured Nexus file saved to: {output_file}")
    print(f"Original: {len(taxon_names)} gene copies")
    print(f"Restructured: {len(species_list)} species")
    
    return True

def test_restructured_data(restructured_file):
    """
    Test the restructured data with a simple analysis.
    """
    print(f"\n=== TESTING RESTRUCTURED DATA ===")
    
    try:
        # Import the necessary modules
        from NetworkParser import NetworkParser
        
        # Load the restructured data
        parser = NetworkParser(restructured_file)
        trees = parser.get_all_networks()
        
        print(f"Loaded {len(trees)} trees from restructured file")
        
        # Get species
        all_species = set()
        for tree in trees:
            all_species.update(node.label for node in tree.get_leaves())
        
        species_names = sorted(list(all_species))
        print(f"Found {len(species_names)} species: {species_names}")
        
        # Simple analysis: check tree structure
        for i, tree in enumerate(trees[:3]):  # Show first 3 trees
            leaves = tree.get_leaves()
            print(f"Tree {i+1}: {len(leaves)} leaves")
            if len(leaves) <= 10:
                print(f"  Leaves: {[leaf.label for leaf in leaves]}")
        
        return True
        
    except Exception as e:
        print(f"Error testing restructured data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to restructure gene data.
    """
    print("=== GENE DATA RESTRUCTURING ===")
    
    # Input and output files
    input_file = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/5_MPAllop.nex"
    output_file = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/5_MPAllop_species.nex"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False
    
    # Restructure the data
    success = restructure_nexus_file(input_file, output_file)
    
    if success:
        # Test the restructured data
        test_restructured_data(output_file)
        
        print(f"\n=== NEXT STEPS ===")
        print(f"1. Restructured file created: {output_file}")
        print(f"2. You can now test your taxon set partitioning method on this species-level data")
        print(f"3. Run: test_real_biological_data('{output_file}')")
        
        return True
    else:
        print("Restructuring failed!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nRestructuring completed successfully!")
    else:
        print("\nRestructuring failed!") 