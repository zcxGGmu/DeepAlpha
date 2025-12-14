#!/bin/bash
# Installation script for DeepAlpha Rust modules

set -e

echo "=== DeepAlpha Rust Modules Installation ==="
echo

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Rust is not installed. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "✓ Rust is already installed"
    rustc --version
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python version: $python_version"

# Install Python dependencies
echo
echo "Installing Python dependencies..."
pip3 install maturin numpy pytest pytest-benchmark

# Navigate to rust directory
cd "$(dirname "$0")/../rust"

# Build the module
echo
echo "Building Rust modules..."
maturin develop --release

echo
echo "=== Installation completed successfully! ==="
echo
echo "You can now use the Rust modules in Python:"
echo
echo ">>> from deepalpha_rust import TechnicalIndicators"
echo ">>> indicators = TechnicalIndicators([1, 2, 3, 4, 5])"
echo ">>> sma = indicators.sma(period=3)"
echo
echo "To run performance benchmarks:"
echo "cd rust/benchmarks && python test_performance.py"