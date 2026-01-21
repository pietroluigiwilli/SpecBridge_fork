#!/usr/bin/env python3
"""
Convert MGF file to TSV format matching spectraverse_clean.tsv structure
"""

import sys
import csv
import re
from typing import List, Tuple, Dict, Optional

def parse_mgf_file(mgf_path: str) -> List[Dict]:
    """Parse MGF file and extract spectrum data"""
    spectra = []
    current_spectrum = {}
    peaks = []
    
    with open(mgf_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == "BEGIN IONS":
                current_spectrum = {}
                peaks = []
                continue
            elif line == "END IONS":
                if peaks:
                    current_spectrum['peaks'] = peaks
                    spectra.append(current_spectrum)
                current_spectrum = {}
                peaks = []
                continue
            
            # Parse metadata fields
            if '=' in line:
                key, value = line.split('=', 1)
                current_spectrum[key] = value
            # Parse peak data (mz intensity)
            elif line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        peaks.append((mz, intensity))
                    except ValueError:
                        continue
    
    return spectra

def normalize_intensities(intensities: List[float]) -> List[float]:
    """Normalize intensities to have max = 1.0"""
    if not intensities:
        return []
    max_intensity = max(intensities)
    if max_intensity == 0:
        return intensities
    return [i / max_intensity for i in intensities]

def extract_collision_energy(collision_energy_str: str) -> str:
    """Extract collision energy from string, handle array format"""
    if not collision_energy_str:
        return ""
    
    # Handle array format like "[20.0, 45.0, 60.0]"
    if collision_energy_str.startswith('['):
        # Extract numbers and take average or first value
        numbers = re.findall(r'\d+\.?\d*', collision_energy_str)
        if numbers:
            # Return the first value as string
            return numbers[0]
    
    # Return as is if it's a single value
    return collision_energy_str

def convert_to_tsv(spectra: List[Dict], output_path: str):
    """Convert spectra to TSV format"""
    
    # Define column headers matching spectraverse_clean.tsv
    headers = [
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
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        
        for idx, spectrum in enumerate(spectra, start=1):
            # Extract peaks
            peaks = spectrum.get('peaks', [])
            if not peaks:
                continue
            
            # Sort peaks by m/z
            peaks.sort(key=lambda x: x[0])
            
            # Extract m/z and intensities
            mzs = [str(p[0]) for p in peaks]
            intensities = [p[1] for p in peaks]
            
            # Normalize intensities
            intensities_normalized = normalize_intensities(intensities)
            
            # Generate identifier
            identifier = f"MSNLIB{idx:08d}"
            
            # Extract fields
            smiles = spectrum.get('SMILES', '')
            inchikey = spectrum.get('INCHIAUX', '')
            formula = spectrum.get('FORMULA', '')
            precursor_formula = formula  # Use same as formula
            parent_mass = spectrum.get('EXACTMASS', '')
            precursor_mz = spectrum.get('PEPMASS', '')
            adduct = spectrum.get('ADDUCT', '')
            instrument_type = spectrum.get('INSTRUMENT_TYPE', '')
            collision_energy = extract_collision_energy(spectrum.get('COLLISION_ENERGY', ''))
            fold = spectrum.get('FOLD', '')
            simulation_challenge = 'False'  # Default value
            
            # Write row
            row = [
                identifier,
                ','.join(mzs),
                ','.join([f"{i:.15f}" for i in intensities_normalized]),
                smiles,
                inchikey,
                formula,
                precursor_formula,
                parent_mass,
                precursor_mz,
                adduct,
                instrument_type,
                collision_energy,
                fold,
                simulation_challenge
            ]
            writer.writerow(row)

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_mgf_to_tsv.py <input.mgf> <output.tsv>")
        sys.exit(1)
    
    input_mgf = sys.argv[1]
    output_tsv = sys.argv[2]
    
    print(f"Reading MGF file: {input_mgf}")
    spectra = parse_mgf_file(input_mgf)
    print(f"Found {len(spectra)} spectra")
    
    print(f"Converting to TSV format: {output_tsv}")
    convert_to_tsv(spectra, output_tsv)
    print(f"Conversion complete! Output written to {output_tsv}")

if __name__ == "__main__":
    main()
