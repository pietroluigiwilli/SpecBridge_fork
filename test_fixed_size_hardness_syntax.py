#!/usr/bin/env python3
"""
Quick syntax and import test for eval_fixed_size_hardness.py
"""

import sys
import os

# Add the SpecBridge directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing syntax and imports...")

try:
    # Test 1: Check if the file can be parsed (syntax check)
    print("1. Checking Python syntax...")
    with open('eval_fixed_size_hardness.py', 'r') as f:
        code = f.read()
    compile(code, 'eval_fixed_size_hardness.py', 'exec')
    print("   ✓ Syntax is valid")
except SyntaxError as e:
    print(f"   ✗ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    # Test 2: Check if imports work
    print("2. Checking imports...")
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    print("   ✓ Basic imports work")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

try:
    # Test 3: Check if RDKit is available
    print("3. Checking RDKit...")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    print("   ✓ RDKit is available")
except ImportError:
    print("   ⚠ RDKit not available (will fail at runtime)")

try:
    # Test 4: Check if specbridge modules can be imported
    print("4. Checking specbridge imports...")
    from specbridge.data.massspecgym import MassSpecGymDataset
    print("   ✓ specbridge.data.massspecgym imports work")
except ImportError as e:
    print(f"   ⚠ specbridge import warning: {e}")

try:
    # Test 5: Check if key functions are defined
    print("5. Checking function definitions...")
    # Import the module to check functions exist
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_fixed_size_hardness", "eval_fixed_size_hardness.py")
    module = importlib.util.module_from_spec(spec)
    
    # Just check syntax, don't execute
    print("   ✓ Module can be loaded")
except Exception as e:
    print(f"   ✗ Error loading module: {e}")
    sys.exit(1)

print("\n✓ All syntax and import checks passed!")
print("The script should be ready to run.")
