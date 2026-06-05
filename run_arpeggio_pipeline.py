#!/usr/bin/env python3
"""
Batch processing script for protein-ligand complexes from multiple dataset formats.

Handles three dataset types:
1. PDBBind (refined-set, v2020-other-PL): protein + ligand SDF in same folder
2. BindingNet: protein in parent folder, multiple ligand SDFs nested in subfolders  
3. BindingDB/Surflex: protein + multiple ligand MOL2 files, with filename filtering

Usage:
    Run from the parent data directory:
    cd /Users/ivanshatrov/AEV-PLIG/data
    python run_arpeggio_pipeline.py --output ~/Desktop/arpeggio_results.csv

    Or run on a subset:
    python run_arpeggio_pipeline.py --datasets pdbbind --output ~/Desktop/pdbbind_results.csv
"""

import argparse
import os
import sys
import re
import json
import csv
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

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
# PDB ATOM NAME FIXING
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
            chain = line[21]
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
    """
    Fix duplicate residues in a PDB file by renumbering them.
    
    This handles the error: "Blank altlocs in duplicate residue"
    When multiple different residues share the same residue number,
    we renumber them sequentially.
    """
    stats = {'removed_lines': 0, 'renumbered_residues': 0}
    
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    # Track (chain, resnum, icode) -> (resname, set of atom names)
    # If we see a different resname OR an atom we've already seen, it's a duplicate
    seen_residues = {}  # (chain, resnum, icode) -> (resname, set of atom names)
    residue_renumber_map = {}  # (chain, original_resnum, icode, occurrence) -> new_resnum
    current_occurrence = {}  # (chain, resnum, icode) -> current occurrence number
    
    # Find max residue number per chain
    max_resnum = {}
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21]
            try:
                resnum = int(line[22:26].strip())
                if chain not in max_resnum:
                    max_resnum[chain] = resnum
                else:
                    max_resnum[chain] = max(max_resnum[chain], resnum)
            except ValueError:
                pass
    
    # Process lines
    output_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21]
            res_num_str = line[22:26]
            res_num = res_num_str.strip()
            ins_code = line[26] if len(line) > 26 else ' '
            res_name = line[17:20].strip()
            atom_name = line[12:16].strip()
            
            res_key = (chain, res_num, ins_code)
            
            if res_key not in seen_residues:
                seen_residues[res_key] = (res_name, set())
                current_occurrence[res_key] = 0
            
            stored_resname, stored_atoms = seen_residues[res_key]
            
            # Check if this is a new residue (different name OR repeated atom)
            is_new_residue = (res_name != stored_resname) or (atom_name in stored_atoms)
            
            if is_new_residue:
                # This is a new residue with the same number - start fresh
                seen_residues[res_key] = (res_name, set())
                current_occurrence[res_key] += 1
                
                # Assign a new residue number
                if chain not in max_resnum:
                    max_resnum[chain] = 1000
                max_resnum[chain] += 1
                new_resnum = max_resnum[chain]
                residue_renumber_map[(chain, res_num, ins_code, current_occurrence[res_key])] = new_resnum
                stats['renumbered_residues'] += 1
            
            seen_residues[res_key][1].add(atom_name)
            
            # Apply renumbering if needed
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
    
    # For compatibility
    stats['removed_lines'] = stats['renumbered_residues']
    return stats


# ============================================================================
# LIGAND CONVERSION (SDF and MOL2)
# ============================================================================

def ligand_to_pdb(ligand_path: str, ligand_format: str, 
                  resname: str = "LIG", chain: str = "X", resnum: int = 1) -> str:
    """Convert ligand (SDF or MOL2) to PDB format with proper naming."""
    mol = next(pybel.readfile(ligand_format, ligand_path))
    obmol = mol.OBMol
    
    atom_counts = {}
    pdb_lines = []
    pdb_lines.append(f"COMPND    {resname}")
    pdb_lines.append("AUTHOR    GENERATED BY ARPEGGIO PIPELINE")
    
    for i, atom in enumerate(mol.atoms):
        idx = i + 1
        element = ob.GetSymbol(atom.atomicnum)
        
        if element not in atom_counts:
            atom_counts[element] = 0
        atom_counts[element] += 1
        
        if len(element) == 1:
            atom_name = f"{element}{atom_counts[element]}"
            if len(atom_name) <= 3:
                atom_name = f" {atom_name}"
        else:
            atom_name = f"{element}{atom_counts[element]}"
        atom_name = atom_name[:4].ljust(4)
        
        x, y, z = atom.coords
        line = (f"HETATM{idx:5d} {atom_name} {resname:3s} {chain:1s}"
                f"{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2}")
        pdb_lines.append(line)
    
    for bond in ob.OBMolBondIter(obmol):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        pdb_lines.append(f"CONECT{begin_idx:5d}{end_idx:5d}")
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


# ============================================================================
# STRUCTURE COMBINATION
# ============================================================================

def combine_structures(protein_pdb: str, ligand_pdb_content: str, 
                       output_cif: str, ligand_chain: str = "X") -> Tuple[bool, Optional[str]]:
    """Combine protein and ligand into mmCIF. Returns (success, actual_ligand_chain)."""
    try:
        protein = gemmi.read_structure(protein_pdb)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(ligand_pdb_content)
            temp_lig = f.name
        
        try:
            ligand = gemmi.read_structure(temp_lig)
            existing_chains = [c.name for c in protein[0]]
            
            actual_chain = None
            for chain in ligand[0]:
                if chain.name in existing_chains or not chain.name:
                    chain.name = ligand_chain
                    while chain.name in existing_chains:
                        chain.name = chr((ord(chain.name) + 1 - ord('A')) % 26 + ord('A'))
                actual_chain = chain.name
                protein[0].add_chain(chain)
            
            doc = protein.make_mmcif_document()
            doc.write_file(output_cif)
            return True, actual_chain
        finally:
            os.remove(temp_lig)
    except Exception as e:
        print(f"    ERROR combining structures: {e}")
        return False, None


# ============================================================================
# ARPEGGIO EXECUTION AND PARSING
# ============================================================================

def run_arpeggio(cif_file: str, output_dir: str, 
                 chain: str = "X", resnum: int = 1) -> Tuple[Optional[str], Optional[str]]:
    """Run Arpeggio and return path to JSON output and any error message."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cmd = ["pdbe-arpeggio", "-s", f"/{chain}/{resnum}/", "-o", output_dir, cif_file]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if proc.returncode == 0:
            json_files = list(Path(output_dir).glob("*.json"))
            if json_files:
                return str(json_files[0]), None
            return None, "Arpeggio produced no output JSON"
        else:
            # Return last 500 chars of stderr for debugging
            error_msg = proc.stderr[-500:] if proc.stderr else "Unknown error"
            return None, error_msg
    except Exception as e:
        return None, str(e)


def parse_arpeggio_json(json_path: str, ligand_resname: str = "LIG") -> dict:
    """Parse Arpeggio JSON and count interactions."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    counts = defaultdict(int)
    
    for interaction in data:
        bgn_res = interaction.get('bgn', {}).get('label_comp_id', '')
        end_res = interaction.get('end', {}).get('label_comp_id', '')
        
        if bgn_res != ligand_resname and end_res != ligand_resname:
            continue
        
        contacts = interaction.get('contact', [])
        
        # Count every ligand-involved interaction toward total
        counts['total_contacts'] += 1
        
        for contact in contacts:
            counts[contact] += 1
    
    return dict(counts)


# ============================================================================
# DATASET DISCOVERY
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
        
        # Find target directories
        for target_dir in pdb_dir.iterdir():
            if not target_dir.is_dir() or not target_dir.name.startswith("target_"):
                continue
            
            target_id = target_dir.name
            
            # Find ligand directories
            for ligand_dir in target_dir.iterdir():
                if not ligand_dir.is_dir():
                    continue
                
                ligand_id = ligand_dir.name
                
                # Look for SDF file
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
    
    # Pattern for ligand files: {4chars}-results_{3+digits}.mol2
    ligand_pattern = re.compile(r'^([A-Za-z0-9]{4})-results_(\d{3,})\.mol2$')
    
    for complex_dir in surflex_dir.iterdir():
        if not complex_dir.is_dir():
            continue
        
        folder_name = complex_dir.name
        # Extract first 4 characters for PDB ID
        pdb_id = folder_name[:4]
        
        protein_file = complex_dir / f"{pdb_id}.pdb"
        
        if not protein_file.exists():
            continue
        
        # Find all valid ligand files
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
# MAIN PROCESSING
# ============================================================================

def process_pair(pair: Dict, temp_dir: str, fix_atoms: bool = True) -> Dict:
    """Process a single protein-ligand pair."""
    result = {
        'protein_path': pair['protein'],
        'ligand_path': pair['ligand'],
        'dataset': pair['dataset'],
        'id': pair['id'],
        'success': False,
        'error': None,
        'counts': {}
    }
    
    try:
        protein_path = pair['protein']
        
        # Fix protein issues if needed
        if fix_atoms:
            # First fix duplicate residues
            fixed_protein_1 = os.path.join(temp_dir, "protein_dedup.pdb")
            dedup_stats = fix_duplicate_residues(protein_path, fixed_protein_1)
            protein_path = fixed_protein_1  # Always use the output file
            
            # Then fix duplicate atom names
            fixed_protein_2 = os.path.join(temp_dir, "protein_fixed.pdb")
            fix_stats = fix_pdb_atom_names(protein_path, fixed_protein_2)
            protein_path = fixed_protein_2  # Always use the output file
        
        # Convert ligand
        ligand_pdb = ligand_to_pdb(
            pair['ligand'], 
            pair['ligand_format'],
            resname="LIG", 
            chain="X", 
            resnum=1
        )
        
        # Combine structures
        cif_path = os.path.join(temp_dir, "complex.cif")
        success, actual_chain = combine_structures(protein_path, ligand_pdb, cif_path)
        if not success:
            result['error'] = "Failed to combine structures"
            return result
        
        # Run Arpeggio with the actual ligand chain
        arpeggio_dir = os.path.join(temp_dir, "arpeggio")
        json_path, arpeggio_error = run_arpeggio(cif_path, arpeggio_dir, chain=actual_chain, resnum=1)
        
        if not json_path:
            result['error'] = f"Arpeggio failed: {arpeggio_error}"
            return result
        
        # Parse results
        result['counts'] = parse_arpeggio_json(json_path)
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch process protein-ligand complexes with Arpeggio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets
    cd /Users/ivanshatrov/AEV-PLIG/data
    python run_arpeggio_pipeline.py --output ~/Desktop/results.csv

    # Process specific datasets
    python run_arpeggio_pipeline.py --datasets pdbbind bindingnet --output ~/Desktop/results.csv
    
    # Limit number of complexes (for testing)
    python run_arpeggio_pipeline.py --limit 10 --output ~/Desktop/test_results.csv
        """
    )
    
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file path')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['pdbbind', 'bindingnet', 'bindingdb'],
                        default=['pdbbind', 'bindingnet', 'bindingdb'],
                        help='Datasets to process (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of complexes to process (for testing)')
    parser.add_argument('--no-fix-atoms', action='store_true',
                        help='Skip fixing duplicate atom names in proteins')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Get current directory as base
    base_dir = os.getcwd()
    
    print(f"Base directory: {base_dir}")
    print(f"Datasets to process: {args.datasets}")
    print(f"Output file: {args.output}")
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
    
    # CSV columns
    interaction_columns = [
        'total_contacts', 'vdw', 'vdw_clash', 'covalent', 'clash', 'proximal',
        'polar', 'weak_polar', 'hbond', 'weak_hbond', 'xbond',
        'ionic', 'metal_complex', 'aromatic', 'hydrophobic', 'carbonyl',
        'CARBONPI', 'CATIONPI', 'DONORPI', 'HALOGENPI', 'METSULPHURPI',
        'FF', 'FE', 'EF', 'EE', 'OF', 'OE', 'FT', 'ET',
        'AMIDEAMIDE', 'AMIDERING'
    ]
    
    all_columns = ['dataset', 'id', 'protein_path', 'ligand_path', 'success', 'error'] + interaction_columns
    
    # Process all pairs
    results = []
    successful = 0
    failed = 0
    
    print("\nProcessing...")
    
    for i, pair in enumerate(all_pairs):
        if args.verbose:
            print(f"  [{i+1}/{len(all_pairs)}] {pair['dataset']}/{pair['id']}")
        else:
            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing {i+1}/{len(all_pairs)}...", end='\r')
        
        # Create temp directory for this pair
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_pair(pair, temp_dir, fix_atoms=not args.no_fix_atoms)
        
        if result['success']:
            successful += 1
            if args.verbose:
                print(f"    ✓ {result['counts'].get('total_contacts', 0)} contacts")
        else:
            failed += 1
            if args.verbose:
                print(f"    ✗ {result['error']}")
        
        results.append(result)
    
    print(f"\n\nCompleted: {successful} successful, {failed} failed")
    
    # Write CSV
    print(f"\nWriting results to {args.output}")
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        
        for result in results:
            row = {
                'dataset': result['dataset'],
                'id': result['id'],
                'protein_path': result['protein_path'],
                'ligand_path': result['ligand_path'],
                'success': result['success'],
                'error': result['error'] or ''
            }
            
            # Add interaction counts
            for col in interaction_columns:
                row[col] = result['counts'].get(col, 0)
            
            writer.writerow(row)
    
    print(f"Done! Results saved to {args.output}")


if __name__ == '__main__':
    main()
