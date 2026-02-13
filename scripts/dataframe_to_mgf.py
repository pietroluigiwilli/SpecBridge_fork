import pandas as pd

def convert_pd_to_mgf(df, output_path):
    """
    Convert a pandas DataFrame to MGF format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: smiles, formula, precursor_mz, adduct, 
        instrument_type, collision_energy, fold, mzs, intensities
    output_path : str
        Path to save the MGF file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Start of spectrum block
            f.write("BEGIN IONS\n")
            
            # Write metadata
            f.write(f"TITLE=spectrum_{idx}\n")
            
            if pd.notna(row['smiles']):
                f.write(f"SMILES={row['smiles']}\n")
            
            if pd.notna(row['formula']):
                f.write(f"FORMULA={row['formula']}\n")
            
            if pd.notna(row['precursor_mz']):
                f.write(f"PRECURSOR_MZ={row['precursor_mz']}\n")
            
            if pd.notna(row['adduct']):
                f.write(f"ADDUCT={row['adduct']}\n")
            
            if pd.notna(row['instrument_type']):
                f.write(f"INSTRUMENT_TYPE={row['instrument_type']}\n")
            
            if pd.notna(row['collision_energy']):
                f.write(f"COLLISION_ENERGY={row['collision_energy']}\n")
            
            if pd.notna(row['fold']):
                f.write(f"FOLD={row['fold']}\n")
            
            # Parse and write peak data (m/z intensity pairs)
            # Handle different formats of masses and intensities
            if isinstance(row['mzs'], str):
                masses = [float(m.strip()) for m in row['mzs'].split(',')]
            else:
                masses = row['mzs']  # Already a list
            
            if isinstance(row['intensities'], str):
                intensities = [float(i.strip()) for i in row['intensities'].split(',')]
            else:
                intensities = row['intensities']  # Already a list
            
            # Write m/z intensity pairs
            for mz, intensity in zip(masses, intensities):
                f.write(f"{mz} {intensity}\n")
            
            # End of spectrum block
            f.write("END IONS\n")
            f.write("\n")  # Blank line between spectra (optional but common)
    
    print(f"Wrote {len(df)} spectra to {output_path}")


# Example usage:
# df = pd.read_csv('your_data.csv')
#dataframe_to_mgf(df, 'output.mgf')