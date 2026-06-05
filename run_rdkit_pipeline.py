#!/usr/bin/env python3
"""
Batch processing script for calculating RDKit molecular descriptors.

Handles three dataset types:
1. PDBBind (refined-set, v2020-other-PL): ligand SDF in complex folder
2. BindingNet: ligand SDFs nested in subfolders  
3. BindingDB/Surflex: ligand MOL2 files

Calculates:
- All 217 standard RDKit 2D descriptors
- 10 3D shape descriptors (Asphericity, Eccentricity, PMI, NPR, etc.)
- Ligand strain energy (MMFF94: E_bound - E_minimized)

Usage:
    cd /Users/ivanshatrov/AEV-PLIG/data
    python run_rdkit_pipeline.py --output ~/Desktop/rdkit_results.csv
"""

import argparse
import os
import sys
import re
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Descriptors3D
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import rdMolDescriptors
except ImportError:
    print("ERROR: rdkit not found. Install with: pip install rdkit")
    sys.exit(1)


# ============================================================================
# DESCRIPTOR SETUP
# ============================================================================

# Get all 2D descriptor names
DESCRIPTOR_2D_NAMES = [name for name, _ in Descriptors.descList]

# 3D shape descriptor names
DESCRIPTOR_3D_NAMES = [
    'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'SpherocityIndex',
    'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration'
]

# All descriptor columns (2D + 3D + strain)
ALL_DESCRIPTOR_NAMES = DESCRIPTOR_2D_NAMES + DESCRIPTOR_3D_NAMES + ['StrainEnergy']

# Create calculator for 2D descriptors
DESCRIPTOR_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_2D_NAMES)


# ============================================================================
# DESCRIPTOR CALCULATION
# ============================================================================

def calculate_2d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate all 217 standard RDKit 2D descriptors."""
    result = {}
    try:
        values = DESCRIPTOR_CALC.CalcDescriptors(mol)
        result = dict(zip(DESCRIPTOR_2D_NAMES, values))
    except Exception:
        # If batch calculation fails, try individual descriptors
        for name, func in Descriptors.descList:
            try:
                result[name] = func(mol)
            except Exception:
                result[name] = None
    return result


def calculate_3d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate 3D shape descriptors. Requires 3D coordinates."""
    result = {}
    
    if mol.GetNumConformers() == 0:
        return result
    
    # Calculate each descriptor individually to handle partial failures
    desc_funcs = [
        ('Asphericity', Descriptors3D.Asphericity),
        ('Eccentricity', Descriptors3D.Eccentricity),
        ('InertialShapeFactor', Descriptors3D.InertialShapeFactor),
        ('SpherocityIndex', Descriptors3D.SpherocityIndex),
        ('NPR1', Descriptors3D.NPR1),
        ('NPR2', Descriptors3D.NPR2),
        ('PMI1', Descriptors3D.PMI1),
        ('PMI2', Descriptors3D.PMI2),
        ('PMI3', Descriptors3D.PMI3),
        ('RadiusOfGyration', Descriptors3D.RadiusOfGyration),
    ]
    
    for name, func in desc_funcs:
        try:
            result[name] = func(mol)
        except Exception:
            result[name] = None
    
    return result


def calculate_strain_energy(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate ligand strain energy using MMFF94.
    
    Strain = E_bound - E_minimized
    
    Where:
    - E_bound is the energy of the input conformer (bound pose)
    - E_minimized is the energy after geometry optimization
    
    Returns None if calculation fails.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    
    try:
        # Work on a copy to preserve original
        mol_copy = Chem.Mol(mol)
        
        # Add hydrogens if not present (needed for MMFF)
        mol_with_h = Chem.AddHs(mol_copy, addCoords=True)
        
        # Get MMFF properties
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
        if mmff_props is None:
            # Try UFF as fallback
            return calculate_strain_energy_uff(mol)
        
        # Calculate energy of bound conformer
        ff_bound = AllChem.MMFFGetMoleculeForceField(mol_with_h, mmff_props)
        if ff_bound is None:
            return calculate_strain_energy_uff(mol)
        
        e_bound = ff_bound.CalcEnergy()
        
        # Create a copy for minimization
        mol_minimized = Chem.Mol(mol_with_h)
        
        # Minimize the conformer
        ff_min = AllChem.MMFFGetMoleculeForceField(mol_minimized, mmff_props)
        if ff_min is None:
            return calculate_strain_energy_uff(mol)
        
        ff_min.Minimize(maxIts=500)
        e_minimized = ff_min.CalcEnergy()
        
        strain = e_bound - e_minimized
        
        # Sanity check - strain should be non-negative (or slightly negative due to numerical issues)
        if strain < -1.0:
            # Something went wrong, try generating a new conformer
            return calculate_strain_with_new_conformer(mol)
        
        return max(0.0, strain)  # Clamp small negative values to 0
        
    except Exception as e:
        print(f"    Warning: MMFF strain calculation error: {e}")
        return calculate_strain_energy_uff(mol)


def calculate_strain_energy_uff(mol: Chem.Mol) -> Optional[float]:
    """Fallback strain calculation using UFF force field."""
    try:
        mol_copy = Chem.Mol(mol)
        mol_with_h = Chem.AddHs(mol_copy, addCoords=True)
        
        # Calculate energy of bound conformer
        ff_bound = AllChem.UFFGetMoleculeForceField(mol_with_h)
        if ff_bound is None:
            return None
        
        e_bound = ff_bound.CalcEnergy()
        
        # Minimize
        mol_minimized = Chem.Mol(mol_with_h)
        ff_min = AllChem.UFFGetMoleculeForceField(mol_minimized)
        if ff_min is None:
            return None
        
        ff_min.Minimize(maxIts=500)
        e_minimized = ff_min.CalcEnergy()
        
        strain = e_bound - e_minimized
        return max(0.0, strain)
        
    except Exception as e:
        print(f"    Warning: UFF strain calculation error: {e}")
        return None


def calculate_strain_with_new_conformer(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate strain by generating a new low-energy conformer.
    Used as fallback when direct minimization gives unexpected results.
    """
    try:
        mol_copy = Chem.Mol(mol)
        mol_with_h = Chem.AddHs(mol_copy, addCoords=True)
        
        # Get energy of bound conformer
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
        if mmff_props is None:
            return None
        
        ff_bound = AllChem.MMFFGetMoleculeForceField(mol_with_h, mmff_props)
        if ff_bound is None:
            return None
        
        e_bound = ff_bound.CalcEnergy()
        
        # Generate new conformers and find lowest energy
        mol_new = Chem.Mol(mol)
        mol_new = Chem.AddHs(mol_new)
        
        # Generate multiple conformers
        conf_ids = AllChem.EmbedMultipleConfs(mol_new, numConfs=50, 
                                               randomSeed=42,
                                               pruneRmsThresh=0.5)
        
        if len(conf_ids) == 0:
            return None
        
        # Minimize each and find lowest energy
        min_energy = float('inf')
        for conf_id in conf_ids:
            try:
                mmff_props_new = AllChem.MMFFGetMoleculeProperties(mol_new)
                if mmff_props_new is None:
                    continue
                ff = AllChem.MMFFGetMoleculeForceField(mol_new, mmff_props_new, confId=conf_id)
                if ff is None:
                    continue
                ff.Minimize(maxIts=500)
                energy = ff.CalcEnergy()
                min_energy = min(min_energy, energy)
            except:
                continue
        
        if min_energy == float('inf'):
            return None
        
        strain = e_bound - min_energy
        return max(0.0, strain)
        
    except Exception as e:
        print(f"    Warning: Conformer generation error: {e}")
        return None


def load_molecule(filepath: str, file_format: str) -> Optional[Chem.Mol]:
    """Load molecule from SDF or MOL2 file with fallbacks for problematic molecules."""
    mol = None
    
    try:
        # First attempt: standard loading
        if file_format == 'sdf':
            supplier = Chem.SDMolSupplier(filepath, removeHs=False)
            for m in supplier:
                if m is not None:
                    mol = m
                    break
        elif file_format == 'mol2':
            mol = Chem.MolFromMol2File(filepath, removeHs=False)
        
        if mol is not None:
            return mol
            
    except Exception:
        pass
    
    # Second attempt: load without sanitization, then do partial sanitization
    try:
        if file_format == 'sdf':
            supplier = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=False)
            for m in supplier:
                if m is not None:
                    mol = m
                    break
        elif file_format == 'mol2':
            mol = Chem.MolFromMol2File(filepath, removeHs=False, sanitize=False)
        
        if mol is not None:
            # Try partial sanitization (skip kekulization)
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                return mol
            except Exception:
                # If that fails too, just return unsanitized mol
                # Some descriptors may fail but most will work
                return mol
                
    except Exception:
        pass
    
    # Third attempt: use OpenBabel as fallback
    try:
        from openbabel import pybel
        
        obmol = next(pybel.readfile(file_format, filepath))
        # Convert to RDKit via SDF string
        sdf_string = obmol.write("sdf")
        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_string, removeHs=False)
        for m in supplier:
            if m is not None:
                return m
        
        # Try without sanitization
        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_string, removeHs=False, sanitize=False)
        for m in supplier:
            if m is not None:
                try:
                    Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    pass
                return m
                
    except Exception:
        pass
    
    return None


# ============================================================================
# DATASET DISCOVERY (from arpeggio/dpocket pipelines)
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
# MAIN PROCESSING
# ============================================================================

def process_ligand(pair: Dict) -> Dict:
    """Process a single ligand and calculate all descriptors."""
    result = {
        'protein_path': pair['protein'],
        'ligand_path': pair['ligand'],
        'dataset': pair['dataset'],
        'id': pair['id'],
        'success': False,
        'error': None,
        'descriptors': {}
    }
    
    try:
        # Load molecule
        mol = load_molecule(pair['ligand'], pair['ligand_format'])
        
        if mol is None:
            result['error'] = "Failed to load molecule"
            return result
        
        # Calculate 2D descriptors
        desc_2d = calculate_2d_descriptors(mol)
        result['descriptors'].update(desc_2d)
        
        # Calculate 3D descriptors
        desc_3d = calculate_3d_descriptors(mol)
        result['descriptors'].update(desc_3d)
        
        # Calculate strain energy
        strain = calculate_strain_energy(mol)
        if strain is not None:
            result['descriptors']['StrainEnergy'] = strain
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Calculate RDKit molecular descriptors for ligands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets
    cd /Users/ivanshatrov/AEV-PLIG/data
    python run_rdkit_pipeline.py --output ~/Desktop/rdkit_results.csv

    # Process specific datasets
    python run_rdkit_pipeline.py --datasets pdbbind bindingnet --output ~/Desktop/results.csv
    
    # Limit number of ligands (for testing)
    python run_rdkit_pipeline.py --limit 10 --output ~/Desktop/test_results.csv
        """
    )
    
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file path')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['pdbbind', 'bindingnet', 'bindingdb'],
                        default=['pdbbind', 'bindingnet', 'bindingdb'],
                        help='Datasets to process (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of ligands to process (for testing)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Get current directory as base
    base_dir = os.getcwd()
    
    print(f"Base directory: {base_dir}")
    print(f"Datasets to process: {args.datasets}")
    print(f"Output file: {args.output}")
    print(f"Descriptors: {len(DESCRIPTOR_2D_NAMES)} 2D + {len(DESCRIPTOR_3D_NAMES)} 3D + 1 strain = {len(ALL_DESCRIPTOR_NAMES)} total")
    print()
    
    # Discover all pairs
    all_pairs = []
    
    if 'pdbbind' in args.datasets:
        pairs = discover_pdbbind(base_dir)
        print(f"Found {len(pairs)} PDBBind ligands")
        all_pairs.extend(pairs)
    
    if 'bindingnet' in args.datasets:
        pairs = discover_bindingnet(base_dir)
        print(f"Found {len(pairs)} BindingNet ligands")
        all_pairs.extend(pairs)
    
    if 'bindingdb' in args.datasets:
        pairs = discover_bindingdb(base_dir)
        print(f"Found {len(pairs)} BindingDB ligands")
        all_pairs.extend(pairs)
    
    print(f"\nTotal ligands to process: {len(all_pairs)}")
    
    if args.limit:
        all_pairs = all_pairs[:args.limit]
        print(f"Limited to {len(all_pairs)} ligands")
    
    if not all_pairs:
        print("No ligands found. Check that you're running from the correct directory.")
        sys.exit(1)
    
    # CSV columns
    all_columns = ['dataset', 'id', 'protein_path', 'ligand_path', 'success', 'error'] + ALL_DESCRIPTOR_NAMES
    
    # Process all ligands
    results = []
    successful = 0
    failed = 0
    total_time = 0
    
    print("\nProcessing...")
    
    for i, pair in enumerate(all_pairs):
        start_time = time.time()
        
        if args.verbose:
            print(f"  [{i+1}/{len(all_pairs)}] {pair['dataset']}/{pair['id']}")
        else:
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing {i+1}/{len(all_pairs)}...", end='\r')
        
        result = process_ligand(pair)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if result['success']:
            successful += 1
            if args.verbose:
                strain = result['descriptors'].get('StrainEnergy', 'N/A')
                if isinstance(strain, float):
                    strain = f"{strain:.2f}"
                print(f"    ✓ Strain: {strain} kcal/mol ({elapsed:.2f}s)")
        else:
            failed += 1
            if args.verbose:
                print(f"    ✗ {result['error']} ({elapsed:.2f}s)")
        
        results.append(result)
    
    avg_time = total_time / len(all_pairs) if all_pairs else 0
    print(f"\n\nCompleted: {successful} successful, {failed} failed")
    print(f"Average time per ligand: {avg_time:.3f}s")
    
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
            
            # Add descriptor values
            for col in ALL_DESCRIPTOR_NAMES:
                val = result['descriptors'].get(col, '')
                # Handle NaN, Inf, and None values
                if val is None:
                    val = ''
                elif isinstance(val, float):
                    if val != val:  # NaN check
                        val = ''
                    elif val == float('inf') or val == float('-inf'):
                        val = ''
                row[col] = val
            
            writer.writerow(row)
    
    print(f"Done! Results saved to {args.output}")


if __name__ == '__main__':
    main()
