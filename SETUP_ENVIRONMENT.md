# Setting Up the SpecBridge Conda Environment

This guide explains how to set up the SpecBridge conda environment from scratch.

## Quick Start

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate specbridge

# Install SpecBridge in editable mode
cd /path/to/SpecBridge
pip install -e .

# Install DreaMS dependency
cd DreaMS
pip install -e .
cd ..
```

## Detailed Setup Instructions

### Method 1: Using environment.yml (Recommended)

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate specbridge
   ```

3. **Install SpecBridge:**
   ```bash
   cd /path/to/SpecBridge
   pip install -e .
   ```

4. **Install DreaMS dependency:**
   ```bash
   cd DreaMS
   pip install -e .
   cd ..
   ```

### Method 2: Manual Setup

If you prefer to set up manually or need to customize:

1. **Create a new conda environment:**
   ```bash
   conda create -n specbridge python=3.11.0
   conda activate specbridge
   ```

2. **Install core dependencies:**
   ```bash
   pip install torch>=2.1 numpy>=1.24 pyteomics>=4.7.5
   ```

3. **Install DreaMS dependencies:**
   ```bash
   pip install numba==0.57.1 pytorch-lightning==2.0.8 torchmetrics==1.3.2
   pip install pandas==2.2.1 pyarrow==15.0.2 h5py==3.11.0
   pip install rdkit==2023.9.5 umap-learn==0.5.6 seaborn==0.13.2
   pip install plotly==5.20.0 ase==3.22.1 wandb>=0.16.4
   pip install pandarallel==1.6.5 matchms==0.24.2 pyopenms==3.0.0
   pip install igraph==0.11.4 molplotly==1.1.7 fire==0.6.0
   pip install huggingface-hub==0.24.5
   pip install git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main
   ```

4. **Install additional utilities:**
   ```bash
   pip install transformers accelerate sentencepiece protobuf
   pip install scipy scikit-learn matplotlib tqdm pyyaml requests
   ```

5. **Install SpecBridge and DreaMS:**
   ```bash
   cd /path/to/SpecBridge
   pip install -e .
   cd DreaMS
   pip install -e .
   cd ..
   ```

## Requirements Summary

### Python Version
- **Python 3.11.0** (required by DreaMS)
- Minimum Python 3.10 (required by SpecBridge)

### Core Dependencies
- **PyTorch >= 2.1** - Deep learning framework
- **NumPy >= 1.24** - Numerical computing
- **pyteomics >= 4.7.5** - Mass spectrometry data handling

### DreaMS Dependencies
- **pytorch-lightning==2.0.8** - Training framework
- **torchmetrics==1.3.2** - Metrics
- **rdkit==2023.9.5** - Molecular processing
- **matchms==0.24.2** - Mass spectrometry matching
- **pyopenms==3.0.0** - OpenMS Python bindings
- **wandb>=0.16.4** - Experiment tracking (optional)

### Optional Dependencies
- **rdkit-pypi>=2022.9.5** - Alternative RDKit installation
- **wandb>=0.16.4** - Weights & Biases for experiment tracking

## Verifying Installation

After setup, verify the installation:

```bash
# Check Python version
python --version  # Should be 3.11.0

# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check SpecBridge
python -c "import specbridge; print('SpecBridge installed successfully')"

# Check DreaMS
python -c "import dreams; print('DreaMS installed successfully')"
```

## Troubleshooting

### Issue: RDKit installation fails
**Solution:** Try installing from conda-forge:
```bash
conda install -c conda-forge rdkit
```

### Issue: PyOpenMS installation fails
**Solution:** PyOpenMS can be tricky. Try:
```bash
conda install -c conda-forge pyopenms
```

### Issue: CUDA/GPU not detected
**Solution:** Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: msml installation fails
**Solution:** The msml package is from a GitHub repository. Ensure git is installed:
```bash
conda install git
pip install git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main
```

## Exporting Your Environment

To export your current environment (useful for sharing):

```bash
conda env export -n specbridge --no-builds > environment.yml
```

Or with build numbers:
```bash
conda env export -n specbridge > environment_full.yml
```

## Notes

- The environment uses **Python 3.11.0** as required by DreaMS
- Some packages have specific version requirements (e.g., DreaMS dependencies)
- The environment includes both conda and pip packages
- DreaMS must be installed separately after creating the environment
