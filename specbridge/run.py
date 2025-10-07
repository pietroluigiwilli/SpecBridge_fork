import argparse
import importlib.util
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "dreams_condition_adapter.py"

def main():
    # Execute the existing script's CLI in-process to preserve flags/behavior
    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT), run_name="__main__")

if __name__ == "__main__":
    main()

