#!/usr/bin/env python3
"""Build script for DeepAlpha Rust modules"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    return result

def main():
    """Main build function"""
    rust_dir = Path(__file__).parent

    print("Building DeepAlpha Rust modules...")

    # Check if maturin is installed
    try:
        run_command([sys.executable, "-c", "import maturin"])
    except:
        print("Installing maturin...")
        run_command([sys.executable, "-m", "pip", "install", "maturin"])

    # Build in development mode
    print("\nBuilding in development mode...")
    run_command([sys.executable, "-m", "maturin", "develop"], cwd=rust_dir)

    print("\nBuild completed successfully!")
    print("You can now import deepalpha_rust in Python")

if __name__ == "__main__":
    main()