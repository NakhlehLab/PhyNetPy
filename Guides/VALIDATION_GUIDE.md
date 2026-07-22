# PhyNetPy Validation Module Guide

## Overview

The PhyNetPy Validation module provides comprehensive validation and summary reporting for common phylogenetic data formats. It helps users identify issues with their input files before processing, preventing unexpected errors and providing detailed summaries of file contents.

## Supported Formats

The validation module supports the following phylogenetic file formats:

| Format | Extensions | Description |
|--------|------------|-------------|
| **Newick** | `.nwk`, `.newick`, `.tre`, `.tree` | Tree format in Newick notation |
| **Nexus** | `.nex`, `.nexus` | Multi-purpose format for trees, networks, and data |
| **FASTA** | `.fasta`, `.fas`, `.fa`, `.fna`, `.ffn`, `.faa` | Sequence data format |
| **PHYLIP** | `.phy`, `.phylip` | Sequence alignment format |
| **Clustal** | `.aln`, `.clustal` | Multiple sequence alignment format |
| **XML** | `.xml` | General XML format (with phylogenetic element detection) |
| **GenBank** | `.gb`, `.gbk`, `.genbank` | NCBI GenBank sequence format |

## Quick Start

### Basic File Validation

```python
from Validation import validate_file

# Validate a single file and print summary
summary = validate_file("my_tree.nex")

# Validate without printing (for programmatic use)
summary = validate_file("my_tree.nex", print_summary=False)

# Check if validation passed
if summary.is_valid:
    print("File is valid!")
else:
    print("Validation failed:", summary.errors)
```

### Directory Validation

```python
from Validation import validate_directory

# Validate all supported files in a directory
summaries = validate_directory("my_data_folder/")

# Validate recursively
summaries = validate_directory("my_data_folder/", recursive=True)

# Process results programmatically
valid_files = [s for s in summaries if s.is_valid]
invalid_files = [s for s in summaries if not s.is_valid]
```

### Using the Main Validator Class

```python
from Validation import PhylogeneticValidator

validator = PhylogeneticValidator()

# Get supported formats
formats = validator.get_supported_formats()
print(formats)

# Validate with format hint
summary = validator.validate_file("data.txt", format_hint="fasta")
```

## Integration with NetworkParser

The validation module is integrated with the existing `NetworkParser` class to provide pre-parsing validation:

```python
from NetworkParser import NetworkParser

# Default behavior: validation enabled with summary printed
parser = NetworkParser("my_network.nex")

# Disable validation
parser = NetworkParser("my_network.nex", validate_input=False)

# Enable validation but don't print summary
parser = NetworkParser("my_network.nex", 
                      validate_input=True, 
                      print_validation_summary=False)

# Access validation results
summary = parser.get_validation_summary()
if summary:
    print(f"Taxa count: {summary.summary_stats.get('Number of Taxa', 'Unknown')}")
```

## Validation Summary Structure

Each validation produces a `ValidationSummary` object with the following attributes:

- **`file_path`**: Path to the validated file
- **`file_format`**: Detected file format
- **`is_valid`**: Boolean indicating if validation passed
- **`errors`**: List of error messages (validation fails if any errors)
- **`warnings`**: List of warning messages (validation passes but issues noted)
- **`summary_stats`**: Dictionary of file statistics and metadata

### Example Summary Statistics by Format

#### Nexus Files
- File size information
- Taxa block analysis (taxa count, names)
- Trees block analysis (tree/network count, names, reticulation detection)
- **Taxa consistency validation**: Ensures trees don't contain undefined taxa
- **Gene loss detection**: Identifies taxa missing from trees (normal biological phenomenon)
- Per-tree taxa coverage statistics
- Data block analysis (sequence count, length, character composition)

#### Newick Files
- Tree count and consistency across multiple trees
- Taxa names and count
- Branch length information
- Tree structure statistics

#### Sequence Files (FASTA, PHYLIP, Clustal)
- Sequence count and IDs
- Sequence length statistics
- Alignment status (aligned vs. unaligned)
- Character composition and sequence type detection (DNA/RNA/Protein)

#### GenBank Files
- Record count and metadata
- Feature analysis
- Sequence composition

## Error Handling

The validation module uses a hierarchy of exception classes:

```python
ValidationError          # Base validation exception
├── FileFormatError     # Invalid or corrupted file format
└── DataIntegrityError  # Data integrity issues
```

Common error scenarios:
- File not found or not readable
- Invalid file format or corruption
- Missing required dependencies (BioPython, python-nexus)
- Inconsistent data (e.g., mismatched gamma values in networks)

## Dependencies

The validation module has optional dependencies that enable different features:

- **BioPython**: Required for most sequence and tree format validation
- **python-nexus**: Required for comprehensive Nexus file validation
- **xml.etree.ElementTree**: Required for XML validation (usually built-in)

If dependencies are missing, the validator will issue warnings and provide limited functionality.

## Command Line Usage

You can run validation from the command line:

```bash
# Validate a single file
python Validation.py my_file.nex

# Run the demo script
python validation_demo.py

# Validate a specific file with the demo
python validation_demo.py my_file.nex
```

## Biological Interpretation of Validation Results

### Understanding Gene Loss vs. Data Errors

The validation module makes an important distinction between two scenarios:

#### ✅ **Gene Loss (Normal - Warning Only)**
```
Taxa defined but not present in any tree (possible gene loss): ['TaxonX', 'TaxonY']
```
This is **biologically normal**. Genes can be lost during evolution, so it's expected that some taxa defined in the taxa block might not appear in all (or any) gene trees.

**Example**: You have 20 species in your study, but a particular gene was lost in 3 species, so those gene trees only have 17 taxa.

#### ❌ **Undefined Taxa (Error - Validation Fails)**
```
Trees contain taxa not defined in taxa block: ['UnknownTaxon1', 'UnknownTaxon2']
```
This indicates **data corruption or formatting errors**. Trees should never contain taxa that aren't defined in the taxa block.

**Example**: Typos in taxon names, data from different studies mixed together, or corrupted file formatting.

### Taxa Coverage Statistics

The validator provides detailed coverage information:
- **Taxa Coverage**: `16/16 defined taxa present in trees` - How many defined taxa appear somewhere in the trees
- **Per-Tree Taxa Coverage**: `['Tree 1: 100.0%', 'Tree 2: 93.8%', ...]` - Percentage of defined taxa in each tree

This helps you understand:
1. **Completeness**: How much of your taxon sampling is represented
2. **Gene loss patterns**: Which trees have missing taxa (normal variation)
3. **Data quality**: Consistent vs. inconsistent taxon sampling

## Advanced Usage

### Custom Validation Logic

You can extend the validation system by creating custom validators:

```python
from Validation import BaseValidator, ValidationSummary

class MyCustomValidator(BaseValidator):
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.myformat'}
    
    def validate(self, file_path: str) -> ValidationSummary:
        summary = ValidationSummary(file_path, "MyFormat")
        # Add custom validation logic here
        return summary
```

### Programmatic Analysis

Use validation results for automated analysis:

```python
from Validation import validate_directory

# Analyze a dataset
summaries = validate_directory("phylogenetic_dataset/", recursive=True)

# Generate report
total_files = len(summaries)
valid_files = sum(1 for s in summaries if s.is_valid)
formats = {}

for summary in summaries:
    fmt = summary.file_format
    formats[fmt] = formats.get(fmt, 0) + 1

print(f"Dataset Analysis:")
print(f"  Total files: {total_files}")
print(f"  Valid files: {valid_files}")
print(f"  Format distribution: {formats}")
```

## Best Practices

1. **Always validate input files** before processing, especially in automated pipelines
2. **Check validation summaries** even for valid files to understand your data
3. **Use format hints** when file extensions might be misleading
4. **Handle missing dependencies** gracefully in production code
5. **Review warnings** as they may indicate data quality issues
6. **Understand gene loss vs. data errors**: Missing taxa in some trees is normal (gene loss), but undefined taxa in trees indicates corrupted data
7. **Check taxa coverage statistics** to understand the completeness of your gene trees

## Troubleshooting

### Common Issues

**"BioPython required" errors**: Install BioPython with `pip install biopython`

**"python-nexus required" errors**: Install with `pip install python-nexus`

**"No valid sequences found"**: Check file encoding and format consistency

**"Trees contain taxa not defined in taxa block"**: Critical error - trees reference undefined taxa

**"Taxa defined but not present in any tree"**: Warning about possible gene loss (biologically normal)

**Validation passes but parsing fails**: The validator checks format validity but may not catch all semantic issues specific to PhyNetPy's requirements

### Performance Considerations

- Validation adds overhead but is typically fast for reasonable file sizes
- For very large files, consider validating a subset or using format hints
- Directory validation can be parallelized if needed for large datasets

## Examples

See the included example scripts:
- `validation_demo.py`: Demonstrates basic validation functionality
- `test_validation_integration.py`: Shows NetworkParser integration

## Contributing

To add support for new file formats:

1. Create a new validator class inheriting from `BaseValidator`
2. Implement the `validate()` method
3. Add the validator to the `PhylogeneticValidator` class
4. Update the supported extensions mapping
5. Add tests and documentation

The validation module is designed to be extensible and maintainable, following the existing PhyNetPy code style and patterns.
