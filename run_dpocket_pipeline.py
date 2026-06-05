#!/usr/bin/env python3
"""
Robust dpocket pipeline - processes one complex at a time with minimal disk usage.

Uses gemmi for structure handling (same as arpeggio pipeline) for maximum robustness
across different PDB sources and formats.

Usage:
    cd /Users/ivanshatrov/AEV-PLIG/data
    python ~/Desktop/run_dpocket_pipeline.py

    # Limit for testing
    python ~/Desktop/run_dpocket_pipeline.py --limit 10 -v

    # Specific datasets
    python ~/Desktop/run_dpocket_pipeline.py --datasets pdbbind bindingnet
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ImportError:
    print("ERROR: openbabel not found. Install with: pip install openbabel-wheel")
    sys.exit(1)

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi not found. Install with: pip install gemmi")
    sys.exit(1)


# ============================================================================
# DPOCKET OUTPUT - parsed dynamically from header
# ============================================================================

# We'll read the actual column names from dpocket's output header
# instead of hardcoding them, since different versions may have different columns


# ============================================================================
# PDB FIXING FUNCTIONS (from arpeggio pipeline)
# ============================================================================

def fix_pdb_atom_names(input_pdb: str, output_pdb: str) -> dict:
    """Fix duplicate atom names in a PDB file."""
    stats = {'renamed_atoms': 0, 'residues_fixed': set()}

    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    residue_atoms = defaultdict(list)

    for i, line in enumerate(lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21] if len(line) > 21 else ' '
            res_num = line[22:26].strip()
            ins_code = line[26] if len(line) > 26 else ' '
            res_key = (chain, res_num, ins_code, res_name)
            residue_atoms[res_key].append((i, atom_name))

    renames = {}
    for res_key, atoms in residue_atoms.items():
        name_counts = defaultdict(list)
        for line_idx, atom_name in atoms:
            name_counts[atom_name].append(line_idx)

        for atom_name, line_indices in name_counts.items():
            if len(line_indices) > 1:
                stats['residues_fixed'].add(res_key)
                for j, line_idx in enumerate(line_indices, 1):
                    base = atom_name.lstrip()
                    new_name = f"{base}{j}"
                    if len(new_name) > 4:
                        new_name = new_name[:4]
                    if len(base) == 1 and base.isalpha():
                        new_name = f"{new_name:>4}"
                    else:
                        new_name = f"{new_name:<4}"
                    renames[line_idx] = new_name
                    stats['renamed_atoms'] += 1

    output_lines = []
    for i, line in enumerate(lines):
        if i in renames:
            new_line = line[:12] + renames[i] + line[16:]
            output_lines.append(new_line)
        else:
            output_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

    stats['residues_fixed'] = len(stats['residues_fixed'])
    return stats


def fix_duplicate_residues(input_pdb: str, output_pdb: str) -> dict:
    """Fix duplicate residues in a PDB file by renumbering them."""
    stats = {'renumbered_residues': 0}

    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    seen_residues = {}
    residue_renumber_map = {}
    current_occurrence = {}

    max_resnum = {}
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21] if len(line) > 21 else ' '
            try:
                resnum = int(line[22:26].strip())
                if chain not in max_resnum:
                    max_resnum[chain] = resnum
                else:
                    max_resnum[chain] = max(max_resnum[chain], resnum)
            except ValueError:
                pass

    output_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21] if len(line) > 21 else ' '
            res_num = line[22:26].strip()
            ins_code = line[26] if len(line) > 26 else ' '
            res_name = line[17:20].strip()
            atom_name = line[12:16].strip()

            res_key = (chain, res_num, ins_code)

            if res_key not in seen_residues:
                seen_residues[res_key] = (res_name, set())
                current_occurrence[res_key] = 0

            stored_resname, stored_atoms = seen_residues[res_key]
            is_new_residue = (res_name != stored_resname) or (atom_name in stored_atoms)

            if is_new_residue:
                seen_residues[res_key] = (res_name, set())
                current_occurrence[res_key] += 1

                if chain not in max_resnum:
                    max_resnum[chain] = 1000
                max_resnum[chain] += 1
                new_resnum = max_resnum[chain]
                residue_renumber_map[(chain, res_num, ins_code, current_occurrence[res_key])] = new_resnum
                stats['renumbered_residues'] += 1

            seen_residues[res_key][1].add(atom_name)

            occ = current_occurrence[res_key]
            if (chain, res_num, ins_code, occ) in residue_renumber_map:
                new_resnum = residue_renumber_map[(chain, res_num, ins_code, occ)]
                new_resnum_str = f"{new_resnum:4d}"
                line = line[:22] + new_resnum_str + line[26:]

            output_lines.append(line)
        else:
            output_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

    return stats


# ============================================================================
# LIGAND CONVERSION (using openbabel, output to PDB string)
# ============================================================================

def ligand_to_pdb(ligand_path: str, ligand_format: str,
                  resname: str = "LIG", chain: str = "X", resnum: int = 1) -> str:
    """
    Convert ligand (SDF or MOL2) to PDB format with proper naming.
    Returns PDB content as a string.
    
    This matches the approach in the arpeggio pipeline.
    """
    try:
        mol = next(pybel.readfile(ligand_format, ligand_path))
    except StopIteration:
        raise ValueError(f"No molecules found in {ligand_path}")
    except Exception as e:
        raise ValueError(f"Failed to read ligand file {ligand_path}: {e}")
    
    obmol = mol.OBMol
    
    if len(mol.atoms) == 0:
        raise ValueError(f"Ligand has no atoms: {ligand_path}")
    
    atom_counts = {}
    pdb_lines = []
    pdb_lines.append(f"COMPND    {resname}")
    pdb_lines.append("AUTHOR    GENERATED BY DPOCKET PIPELINE")
    
    for i, atom in enumerate(mol.atoms):
        idx = i + 1
        element = ob.GetSymbol(atom.atomicnum)
        
        if element not in atom_counts:
            atom_counts[element] = 0
        atom_counts[element] += 1
        
        # Format atom name according to PDB conventions
        if len(element) == 1:
            atom_name = f"{element}{atom_counts[element]}"
            if len(atom_name) <= 3:
                atom_name = f" {atom_name}"
        else:
            atom_name = f"{element}{atom_counts[element]}"
        atom_name = atom_name[:4].ljust(4)
        
        x, y, z = atom.coords
        
        # Check for invalid coordinates
        if any(abs(c) > 9999 for c in (x, y, z)):
            raise ValueError(f"Ligand has coordinates out of PDB range: {x}, {y}, {z}")
        
        line = (f"HETATM{idx:5d} {atom_name} {resname:3s} {chain:1s}"
                f"{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2}")
        pdb_lines.append(line)
    
    # Add CONECT records for bond information
    for bond in ob.OBMolBondIter(obmol):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        pdb_lines.append(f"CONECT{begin_idx:5d}{end_idx:5d}")
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


# ============================================================================
# STRUCTURE COMBINATION (using gemmi, same as arpeggio pipeline)
# ============================================================================

def combine_structures(protein_pdb: str, ligand_pdb_content: str,
                       output_pdb: str, ligand_chain: str = "X") -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Combine protein and ligand into a single PDB file using gemmi.
    
    Returns: (success, actual_ligand_chain, error_message)
    """
    try:
        # Read protein structure with gemmi
        protein = gemmi.read_structure(protein_pdb)
        
        # Write ligand content to temp file for gemmi to read
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(ligand_pdb_content)
            temp_lig = f.name
        
        try:
            # Read ligand structure
            ligand = gemmi.read_structure(temp_lig)
            
            # Get existing chain names in protein
            existing_chains = [c.name for c in protein[0]]
            
            # Add ligand chains to protein, avoiding name conflicts
            actual_chain = None
            for chain in ligand[0]:
                # Ensure unique chain name
                if chain.name in existing_chains or not chain.name:
                    chain.name = ligand_chain
                    while chain.name in existing_chains:
                        # Try next letter
                        chain.name = chr((ord(chain.name) + 1 - ord('A')) % 26 + ord('A'))
                
                actual_chain = chain.name
                protein[0].add_chain(chain)
            
            # Write combined structure as PDB
            protein.write_pdb(output_pdb)
            
            return True, actual_chain, None
            
        finally:
            os.remove(temp_lig)
            
    except Exception as e:
        return False, None, str(e)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def create_complex_and_run_dpocket(
    protein_pdb: str,
    ligand_path: str,
    ligand_format: str,
    temp_dir: str,
    ligand_resname: str = "LIG",
    fix_protein: bool = True
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Create a combined PDB, run dpocket, parse results, all in one temp directory.

    Returns:
        (success, descriptors_dict, error_message)
    """
    try:
        # Step 1: Fix protein issues if needed
        if fix_protein:
            temp_pdb1 = os.path.join(temp_dir, "protein_dedup.pdb")
            temp_pdb2 = os.path.join(temp_dir, "protein_fixed.pdb")
            fix_duplicate_residues(protein_pdb, temp_pdb1)
            fix_pdb_atom_names(temp_pdb1, temp_pdb2)
            protein_path = temp_pdb2
        else:
            protein_path = protein_pdb

        # Step 2: Convert ligand to PDB format
        try:
            ligand_pdb_content = ligand_to_pdb(
                ligand_path, ligand_format,
                resname=ligand_resname,
                chain="X",
                resnum=1
            )
        except ValueError as e:
            return False, None, f"Ligand conversion failed: {e}"

        # Step 3: Combine structures using gemmi
        complex_pdb = os.path.join(temp_dir, "complex.pdb")
        success, actual_chain, error = combine_structures(
            protein_path, ligand_pdb_content, complex_pdb, ligand_chain="X"
        )
        
        if not success:
            return False, None, f"Structure combination failed: {error}"

        # Step 4: Verify the combined PDB has the ligand
        with open(complex_pdb, 'r') as f:
            content = f.read()
        
        if ligand_resname not in content:
            return False, None, f"Combined PDB does not contain ligand residue {ligand_resname}"

        # Step 5: Create dpocket input file
        dpocket_input = os.path.join(temp_dir, "dpocket_input.txt")
        with open(dpocket_input, 'w') as f:
            f.write(f"{complex_pdb}\t{ligand_resname}\n")

        # Step 6: Run dpocket
        dpocket_output_prefix = os.path.join(temp_dir, "dpocket_out")
        cmd = ["dpocket", "-f", dpocket_input, "-o", dpocket_output_prefix]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return False, None, "dpocket timed out after 300 seconds"
        except FileNotFoundError:
            return False, None, "dpocket not found - please install fpocket"

        # Step 7: Check for output files
        # dpocket output files use short suffixes: _exp.txt, _fp.txt, _fpn.txt
        explicit_file = f"{dpocket_output_prefix}_exp.txt"
        
        if not os.path.exists(explicit_file):
            # Collect debug info
            output_files = [f for f in os.listdir(temp_dir) if 'dpocket' in f.lower()]
            error_msg = f"dpocket produced no explicit output file."
            error_msg += f" Return code: {proc.returncode}."
            if proc.stderr:
                error_msg += f" Stderr: {proc.stderr[:300]}"
            if proc.stdout:
                error_msg += f" Stdout: {proc.stdout[:300]}"
            if output_files:
                error_msg += f" Files found: {output_files}"
            return False, None, error_msg

        # Step 8: Parse dpocket output - read header to get column names
        with open(explicit_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return False, None, f"dpocket output has only {len(lines)} lines (need header + data)"

        # Parse header to get column names
        header_line = lines[0].strip()
        header_cols = header_line.split()
        
        # Parse the data line
        data_line = lines[1].strip()
        if not data_line:
            return False, None, "dpocket output data line is empty"
        
        # Split data - handle multiple spaces
        data_values = data_line.split()

        if len(data_values) < len(header_cols):
            return False, None, f"Data has {len(data_values)} values but header has {len(header_cols)} columns"

        # Build descriptors dict from header and data
        # Skip first two columns (pdb path and ligand name)
        descriptors = {}
        for i, col_name in enumerate(header_cols):
            if i < 2:  # Skip 'pdb' and 'lig' columns
                continue
            if i < len(data_values):
                # Normalize column name (replace - with _)
                col_name_normalized = col_name.replace('-', '_')
                descriptors[col_name_normalized] = data_values[i]

        return True, descriptors, None

    except Exception as e:
        import traceback
        return False, None, f"{str(e)}\n{traceback.format_exc()}"


# ============================================================================
# DATASET DISCOVERY (from arpeggio pipeline)
# ============================================================================

def discover_pdbbind(base_dir: str) -> List[Dict]:
    """Discover PDBBind protein-ligand pairs."""
    pairs = []
    pdbbind_dir = Path(base_dir) / "pdbbind"

    if not pdbbind_dir.exists():
        return pairs

    for subset in ["refined-set", "v2020-other-PL"]:
        subset_dir = pdbbind_dir / subset
        if not subset_dir.exists():
            continue

        for complex_dir in subset_dir.iterdir():
            if not complex_dir.is_dir():
                continue

            pdb_id = complex_dir.name
            protein_file = complex_dir / f"{pdb_id}_protein.pdb"
            ligand_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
            ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"

            if protein_file.exists():
                if ligand_sdf.exists():
                    pairs.append({
                        'dataset': 'pdbbind',
                        'subset': subset,
                        'id': pdb_id,
                        'protein': str(protein_file),
                        'ligand': str(ligand_sdf),
                        'ligand_format': 'sdf'
                    })
                elif ligand_mol2.exists():
                    pairs.append({
                        'dataset': 'pdbbind',
                        'subset': subset,
                        'id': pdb_id,
                        'protein': str(protein_file),
                        'ligand': str(ligand_mol2),
                        'ligand_format': 'mol2'
                    })

    return pairs


def discover_bindingnet(base_dir: str) -> List[Dict]:
    """Discover BindingNet protein-ligand pairs."""
    pairs = []
    bindingnet_dir = Path(base_dir) / "bindingnet" / "from_chembl_client"

    if not bindingnet_dir.exists():
        return pairs

    for pdb_dir in bindingnet_dir.iterdir():
        if not pdb_dir.is_dir():
            continue

        pdb_id = pdb_dir.name
        protein_file = pdb_dir / "rec_h_opt.pdb"

        if not protein_file.exists():
            continue

        for target_dir in pdb_dir.iterdir():
            if not target_dir.is_dir() or not target_dir.name.startswith("target_"):
                continue

            target_id = target_dir.name

            for ligand_dir in target_dir.iterdir():
                if not ligand_dir.is_dir():
                    continue

                ligand_id = ligand_dir.name
                sdf_pattern = f"{pdb_id}_{target_id.replace('target_', '')}_{ligand_id}.sdf"
                ligand_file = ligand_dir / sdf_pattern

                if ligand_file.exists():
                    pairs.append({
                        'dataset': 'bindingnet',
                        'subset': target_id,
                        'id': f"{pdb_id}_{ligand_id}",
                        'protein': str(protein_file),
                        'ligand': str(ligand_file),
                        'ligand_format': 'sdf'
                    })

    return pairs


def discover_bindingdb(base_dir: str) -> List[Dict]:
    """Discover BindingDB/Surflex protein-ligand pairs."""
    pairs = []
    surflex_dir = Path(base_dir) / "bindingdb" / "surflex"

    if not surflex_dir.exists():
        return pairs

    ligand_pattern = re.compile(r'^([A-Za-z0-9]{4})-results_(\d{3,})\.mol2$')

    for complex_dir in surflex_dir.iterdir():
        if not complex_dir.is_dir():
            continue

        folder_name = complex_dir.name
        pdb_id = folder_name[:4]

        protein_file = complex_dir / f"{pdb_id}.pdb"

        if not protein_file.exists():
            continue

        for f in complex_dir.iterdir():
            if not f.is_file():
                continue

            match = ligand_pattern.match(f.name)
            if match and match.group(1) == pdb_id:
                ligand_num = match.group(2)
                pairs.append({
                    'dataset': 'bindingdb',
                    'subset': folder_name,
                    'id': f"{pdb_id}_{ligand_num}",
                    'protein': str(protein_file),
                    'ligand': str(f),
                    'ligand_format': 'mol2'
                })

    return pairs


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run dpocket pipeline with minimal disk usage (one complex at a time)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on all datasets
    cd /Users/ivanshatrov/AEV-PLIG/data
    python ~/Desktop/run_dpocket_pipeline.py

    # Test with a small subset
    python ~/Desktop/run_dpocket_pipeline.py --limit 10 -v

    # Run on specific datasets
    python ~/Desktop/run_dpocket_pipeline.py --datasets pdbbind bindingnet
        """
    )

    parser.add_argument('--output', '-o', default='~/Desktop/dpocket_results.csv',
                        help='Output CSV file path (default: ~/Desktop/dpocket_results.csv)')
    parser.add_argument('--datasets', nargs='+',
                        choices=['pdbbind', 'bindingnet', 'bindingdb'],
                        default=['pdbbind', 'bindingnet', 'bindingdb'],
                        help='Datasets to process (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of complexes to process (for testing)')
    parser.add_argument('--ligand-resname', default='LIG',
                        help='Residue name for ligands (default: LIG)')
    parser.add_argument('--no-fix', action='store_true',
                        help='Skip fixing duplicate atom names in proteins')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Check if dpocket is available
    print("Checking dependencies...")
    try:
        proc = subprocess.run(["dpocket"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("ERROR: dpocket not found. Please install fpocket first.")
        print("  macOS: brew install fpocket")
        print("  conda: conda install -c conda-forge fpocket")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        pass  # dpocket is available but waiting for input
    
    print("  ✓ dpocket found")
    print("  ✓ openbabel found")
    print("  ✓ gemmi found")
    print()

    output_file = Path(args.output).expanduser().resolve()
    base_dir = os.getcwd()

    print(f"Base directory: {base_dir}")
    print(f"Output file: {output_file}")
    print(f"Datasets: {args.datasets}")
    print()

    # Discover all pairs
    all_pairs = []

    if 'pdbbind' in args.datasets:
        pairs = discover_pdbbind(base_dir)
        print(f"Found {len(pairs)} PDBBind complexes")
        all_pairs.extend(pairs)

    if 'bindingnet' in args.datasets:
        pairs = discover_bindingnet(base_dir)
        print(f"Found {len(pairs)} BindingNet complexes")
        all_pairs.extend(pairs)

    if 'bindingdb' in args.datasets:
        pairs = discover_bindingdb(base_dir)
        print(f"Found {len(pairs)} BindingDB complexes")
        all_pairs.extend(pairs)

    print(f"\nTotal complexes to process: {len(all_pairs)}")

    if args.limit:
        all_pairs = all_pairs[:args.limit]
        print(f"Limited to {len(all_pairs)} complexes")

    if not all_pairs:
        print("No complexes found. Check that you're running from the correct directory.")
        sys.exit(1)

    # Base columns (descriptors will be added dynamically)
    base_columns = ['dataset', 'id', 'protein_path', 'ligand_path', 'success', 'error']

    # Process all pairs and collect results
    results = []
    successful = 0
    failed = 0
    all_descriptor_cols = set()  # Collect all descriptor column names

    print("\nProcessing (one complex at a time)...")

    for i, pair in enumerate(all_pairs):
        if args.verbose:
            print(f"  [{i+1}/{len(all_pairs)}] {pair['dataset']}/{pair['id']}")
        else:
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing {i+1}/{len(all_pairs)}...", end='\r')

        # Create temporary directory for this single complex
        with tempfile.TemporaryDirectory() as temp_dir:
            success, descriptors, error = create_complex_and_run_dpocket(
                protein_pdb=pair['protein'],
                ligand_path=pair['ligand'],
                ligand_format=pair['ligand_format'],
                temp_dir=temp_dir,
                ligand_resname=args.ligand_resname,
                fix_protein=not args.no_fix
            )

        # Build result row
        row = {
            'dataset': pair['dataset'],
            'id': pair['id'],
            'protein_path': pair['protein'],
            'ligand_path': pair['ligand'],
            'success': str(success),
            'error': error or ''
        }

        # Add descriptors
        if success and descriptors:
            for col_name, value in descriptors.items():
                prefixed_col = f'dpk_{col_name}'
                row[prefixed_col] = value
                all_descriptor_cols.add(prefixed_col)
            successful += 1
            if args.verbose:
                print(f"    ✓ drug_score={descriptors.get('drug_score', 'N/A')}")
        else:
            failed += 1
            if args.verbose:
                # Truncate long error messages
                error_short = (error[:100] + '...') if error and len(error) > 100 else error
                print(f"    ✗ {error_short}")

        results.append(row)

    print(f"\n\nCompleted: {successful} successful, {failed} failed")

    # Sort descriptor columns for consistent ordering
    sorted_descriptor_cols = sorted(all_descriptor_cols)
    all_columns = base_columns + sorted_descriptor_cols

    # Fill in missing columns for failed results
    for row in results:
        for col in sorted_descriptor_cols:
            if col not in row:
                row[col] = ''

    # Write CSV
    print(f"\nWriting results to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Results saved to {output_file}")
    print(f"  Total rows: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Descriptor columns: {len(sorted_descriptor_cols)}")


if __name__ == '__main__':
    main()
