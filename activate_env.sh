#!/bin/bash

# Auto-activate conda environment for TradeML project
# This script should be sourced when entering the project directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    return 1
fi

# Check if environment exists
if ! conda env list | grep -q "trademl"; then
    echo "Creating conda environment 'trademl'..."
    conda env create -f "$SCRIPT_DIR/environment.yml"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment"
        return 1
    fi
fi

# Activate the environment
echo "Activating conda environment 'trademl'..."
conda activate trademl

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "✓ Conda environment 'trademl' activated successfully"
    echo "Python version: $(python --version)"
    echo "Current directory: $(pwd)"
else
    echo "Error: Failed to activate conda environment"
    return 1
fi 