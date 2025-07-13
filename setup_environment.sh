#!/bin/bash

# Setup script for TradeML development environment
# This script installs and configures the conda environment and tools

set -e  # Exit on any error

echo "🚀 Setting up TradeML development environment..."
echo "================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Create conda environment
echo ""
echo "📦 Creating conda environment 'trademl'..."
if conda env list | grep -q "trademl"; then
    echo "⚠️  Environment 'trademl' already exists. Updating..."
    conda env update -f environment.yml
else
    echo "🆕 Creating new environment 'trademl'..."
    conda env create -f environment.yml
fi

# Activate environment
echo ""
echo "🔧 Activating environment..."
source activate_env.sh

# Install direnv if not already installed
echo ""
echo "🔧 Checking direnv installation..."
if ! command -v direnv &> /dev/null; then
    echo "📦 Installing direnv..."
    if command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew install direnv
    elif command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y direnv
    else
        echo "⚠️  Please install direnv manually:"
        echo "  https://direnv.net/docs/installation.html"
    fi
else
    echo "✅ direnv already installed"
fi

# Configure direnv for current shell
echo ""
echo "⚙️  Configuring direnv..."
SHELL_CONFIG=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
fi

if [[ -n "$SHELL_CONFIG" ]]; then
    if ! grep -q "direnv hook" "$SHELL_CONFIG"; then
        echo "Adding direnv hook to $SHELL_CONFIG..."
        echo "" >> "$SHELL_CONFIG"
        echo "# direnv hook for TradeML" >> "$SHELL_CONFIG"
        echo 'eval "$(direnv hook '"${SHELL##*/}"')"' >> "$SHELL_CONFIG"
    else
        echo "✅ direnv hook already configured in $SHELL_CONFIG"
    fi
else
    echo "⚠️  Please add direnv hook to your shell configuration manually:"
    echo "  eval \"\$(direnv hook bash)\"  # for bash"
    echo "  eval \"\$(direnv hook zsh)\"   # for zsh"
fi

# Allow direnv in current directory
echo ""
echo "🔓 Allowing direnv in current directory..."
direnv allow .

# Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/crypto/1_minute
mkdir -p data/crypto/5_minutes
mkdir -p data/crypto/1_hour
mkdir -p data/crypto/1_day

# Make scripts executable
echo ""
echo "🔧 Making scripts executable..."
chmod +x activate_env.sh
chmod +x download_crypto_data.py
chmod +x test_crypto_loader.py

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Restart your terminal or run: source ~/.zshrc (or ~/.bashrc)"
echo "2. Navigate to this directory: cd $(pwd)"
echo "3. The environment should automatically activate"
echo "4. Test the setup: python test_crypto_loader.py"
echo ""
echo "Manual activation (if needed):"
echo "  source activate_env.sh"
echo ""
echo "Environment variables:"
echo "  TRADEML_ROOT: ${PWD}"
echo "  TRADEML_DATA_ROOT: ${PWD}/data" 