# Environment Setup Guide

## Overview

This project uses conda for environment management and direnv for automatic environment activation.

## Quick Start

### 1. Automatic Setup (Recommended)

```bash
# Run the setup script
./setup_environment.sh

# Restart terminal or reload shell config
source ~/.zshrc  # or ~/.bashrc

# Navigate to project directory (auto-activates)
cd /path/to/trademl
```

### 2. Manual Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate manually
conda activate trademl

# Or use the activation script
source activate_env.sh
```

## Environment Files

- `environment.yml` - Conda environment definition
- `.envrc` - direnv configuration for auto-activation
- `activate_env.sh` - Manual activation script
- `setup_environment.sh` - Complete setup script

## Directory Structure

```
trademl/
├── data/
│   └── crypto/
│       ├── 1_minute/
│       ├── 5_minutes/
│       ├── 1_hour/
│       └── 1_day/
├── environment.yml
├── .envrc
├── activate_env.sh
└── setup_environment.sh
```

## Environment Variables

When the environment is activated, these variables are set:

- `TRADEML_ROOT` - Project root directory
- `TRADEML_DATA_ROOT` - Data directory path
- `PYTHONPATH` - Includes project root for imports

## Testing

After setup, test the environment:

```bash
# Test crypto data loader
python test_crypto_loader.py

# Test data download
python download_crypto_data.py
```

## Troubleshooting

### Conda not found
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### direnv not working
```bash
# Install direnv
sudo apt-get install direnv  # Ubuntu/Debian
brew install direnv          # macOS

# Add to shell config
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
```

### Environment activation fails
```bash
# Recreate environment
conda env remove -n trademl
conda env create -f environment.yml
```

## Development Workflow

1. **Enter project directory** - Environment auto-activates
2. **Work on code** - All dependencies available
3. **Leave directory** - Environment deactivates automatically

## Adding Dependencies

To add new dependencies:

1. **Add to environment.yml:**
   ```yaml
   dependencies:
     - new_package>=1.0.0
   ```

2. **Update environment:**
   ```bash
   conda env update -f environment.yml
   ```

3. **Or install manually:**
   ```bash
   conda activate trademl
   conda install new_package
   # or
   pip install new_package
   ``` 