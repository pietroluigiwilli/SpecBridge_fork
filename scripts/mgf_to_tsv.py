#!/usr/bin/env python3
"""
Convert MGF file to TSV format matching MassSpecGym dataset format.

Based on: https://huggingface.co/datasets/roman-bushuiev/MassSpecGym

TSV columns:
- identifier
- mzs (comma-separated)
- intensities (comma-separated)
- smiles
- inchikey
- formula
- precursor_formula
- parent_mass
- precursor_mz
- adduct
- instrument_type
- collision_energy
- fold
- simulation_challenge
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_mgf_block(block: str) -> Optional[Dict]:
    """Parse a single MGF spectrum block."""
    lines = block.strip().split('\n')
    if not lines:
        return None
    
    metadata = {}
    mzs = []
    intensities = []
    
    in_peaks = False
    for line in lines:
        line = line.strip()
        if not line or line.upper() == 'END IONS':
            continue
        
        # Parse metadata (key=value format)
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().upper()
            value = value.strip()
            metadata[key] = value
        else:
            # Parse peak data (m/z intensity)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    mzs.append(mz)
                    intensities.append(intensity)
                except ValueError:
                    continue
    
    if not mzs:
        return None
    
    # Extract required fields with defaults
    identifier = metadata.get('TITLE') or metadata.get('NAME') or f"SPECTRAVERSE{len(mzs)}"
    
    # Get SMILES
    smiles = metadata.get('SMILES', '')
    
    # Get InChIKey
    inchikey = metadata.get('INCHIKEY', '')
    if not inchikey and 'INCHI' in metadata:
        # Try to extract InChIKey from InChI if available
        inchi = metadata['INCHI']
        if inchi.startswith('InChI='):
            # InChIKey is typically 14 characters after the last /
            parts = inchi.split('/')
            if len(parts) > 1:
                inchikey = parts[-1][:14] if len(parts[-1]) >= 14 else ''
    
    # Get formula
    formula = metadata.get('FORMULA', '')
    
    # Get precursor formula (may need to derive from formula + adduct)
    precursor_formula = metadata.get('PRECURSOR_FORMULA', '')
    if not precursor_formula:
        precursor_formula = formula  # Fallback to formula
    
    # Get parent mass
    parent_mass_str = metadata.get('PARENT_MASS', '')
    try:
        parent_mass = float(parent_mass_str) if parent_mass_str else None
    except (ValueError, TypeError):
        parent_mass = None
    
    # Get precursor m/z
    precursor_mz_str = metadata.get('PRECURSOR_MZ') or metadata.get('PEPMASS', '')
    try:
        precursor_mz = float(precursor_mz_str) if precursor_mz_str else None
    except (ValueError, TypeError):
        precursor_mz = None
    
    # Get adduct
    adduct = metadata.get('ADDUCT', '')
    
    # Get instrument type
    instrument_type = metadata.get('INSTRUMENT_TYPE', '')
    if instrument_type:
        instrument_type = instrument_type.lower()
        # Normalize to match MassSpecGym format
        if 'orbitrap' in instrument_type.lower():
            instrument_type = 'Orbitrap'
        elif 'qtof' in instrument_type.lower() or 'tof' in instrument_type.lower():
            instrument_type = 'QTOF'
        else:
            instrument_type = instrument_type.capitalize()
    
    # Get collision energy
    collision_energy_str = (
        metadata.get('COLLISION_ENERGY') or 
        metadata.get('COLLISION_ENERGY_1') or
        metadata.get('NORMALIZED_COLLISION_ENERGY_1') or
        ''
    )
    try:
        collision_energy = float(collision_energy_str) if collision_energy_str and collision_energy_str.lower() != 'nan' else None
    except (ValueError, TypeError):
        collision_energy = None
    
    # Get fold
    fold = metadata.get('FOLD', 'train').lower()
    if fold not in ['train', 'val', 'test']:
        fold = 'train'  # Default to train
    
    # Simulation challenge (default False for real data)
    simulation_challenge = metadata.get('SIMULATION_CHALLENGE', 'False').lower() == 'true'
    
    return {
        'identifier': identifier,
        'mzs': ','.join(f'{mz:.6f}' for mz in mzs),
        'intensities': ','.join(f'{intensity:.15f}' for intensity in intensities),
        'smiles': smiles,
        'inchikey': inchikey,
        'formula': formula,
        'precursor_formula': precursor_formula,
        'parent_mass': parent_mass if parent_mass is not None else '',
        'precursor_mz': precursor_mz if precursor_mz is not None else '',
        'adduct': adduct,
        'instrument_type': instrument_type,
        'collision_energy': collision_energy if collision_energy is not None else '',
        'fold': fold,
        'simulation_challenge': simulation_challenge
    }


def convert_mgf_to_tsv(mgf_path: str, tsv_path: str, max_spectra: Optional[int] = None):
    """Convert MGF file to TSV format."""
    mgf_path = Path(mgf_path)
    tsv_path = Path(tsv_path)
    
    # Column names matching MassSpecGym format
    columns = [
        'identifier',
        'mzs',
        'intensities',
        'smiles',
        'inchikey',
        'formula',
        'precursor_formula',
        'parent_mass',
        'precursor_mz',
        'adduct',
        'instrument_type',
        'collision_energy',
        'fold',
        'simulation_challenge'
    ]
    
    print(f"Reading MGF file: {mgf_path}")
    print(f"Output TSV file: {tsv_path}")
    
    spectra_count = 0
    skipped_count = 0
    
    with open(mgf_path, 'r', encoding='utf-8', errors='ignore') as mgf_file, \
         open(tsv_path, 'w', newline='', encoding='utf-8') as tsv_file:
        
        writer = csv.DictWriter(tsv_file, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        
        current_block = []
        in_block = False
        
        for line_num, line in enumerate(mgf_file, 1):
            if line_num % 100000 == 0:
                print(f"Processed {line_num} lines, {spectra_count} spectra written, {skipped_count} skipped")
            
            line_stripped = line.strip()
            
            if line_stripped.upper().startswith('BEGIN IONS'):
                in_block = True
                current_block = [line]
                continue
            
            if in_block:
                current_block.append(line)
                
                if line_stripped.upper().startswith('END IONS'):
                    # Process complete block
                    block_text = ''.join(current_block)
                    spectrum_data = parse_mgf_block(block_text)
                    
                    if spectrum_data:
                        writer.writerow(spectrum_data)
                        spectra_count += 1
                        
                        if max_spectra and spectra_count >= max_spectra:
                            print(f"Reached max_spectra limit: {max_spectra}")
                            break
                    else:
                        skipped_count += 1
                    
                    in_block = False
                    current_block = []
                    continue
    
    print(f"\nConversion complete!")
    print(f"Total spectra written: {spectra_count}")
    print(f"Total spectra skipped: {skipped_count}")
    print(f"Output file: {tsv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MGF file to TSV format matching MassSpecGym dataset'
    )
    parser.add_argument(
        'mgf_path',
        type=str,
        help='Path to input MGF file'
    )
    parser.add_argument(
        'tsv_path',
        type=str,
        help='Path to output TSV file'
    )
    parser.add_argument(
        '--max-spectra',
        type=int,
        default=None,
        help='Maximum number of spectra to process (for testing)'
    )
    
    args = parser.parse_args()
    
    convert_mgf_to_tsv(args.mgf_path, args.tsv_path, args.max_spectra)


if __name__ == '__main__':
    main()

