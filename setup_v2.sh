#!/bin/bash

# NSE Delivery Tracker Setup Script - Fixed Version
# This script sets up the environment and installs dependencies

echo "========================================="
echo "NSE Delivery Tracker - Setup Script v2"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python version: $PYTHON_VERSION"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip first
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies from fixed requirements
echo "Installing dependencies..."
if [ -f "requirements_fixed.txt" ]; then
    echo "Using requirements_fixed.txt..."
    pip install -r requirements_fixed.txt
else
    echo "Installing essential packages individually..."
    pip install click
    pip install pandas numpy
    pip install requests pyyaml
    pip install beautifulsoup4 lxml
    pip install openpyxl xlsxwriter
    pip install matplotlib seaborn
    pip install loguru python-dotenv
    pip install python-dateutil pytz
    pip install scipy
    pip install tabulate
    pip install pydantic
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p data
mkdir -p reports
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# NSE Delivery Tracker Environment Variables

# Data Source
NSE_DATA_SOURCE=NSE_BHAVCOPY

# Analysis Parameters
NSE_LOOKBACK_DAYS=20
NSE_SPIKE_MULTIPLIER=5.0

# Email Configuration (optional)
NSE_EMAIL_ENABLED=false
NSE_SMTP_SERVER=smtp.gmail.com
NSE_SMTP_PORT=587
NSE_EMAIL_FROM=
NSE_EMAIL_PASSWORD=

# Debug Mode
NSE_DEBUG_MODE=false
EOF
    echo ".env file created."
fi

# Test imports
echo ""
echo "Testing installation..."
python3 -c "
import sys
try:
    import click
    print('✓ click installed')
    import pandas
    print('✓ pandas installed')
    import numpy
    print('✓ numpy installed')
    import requests
    print('✓ requests installed')
    import yaml
    print('✓ yaml installed')
    import loguru
    print('✓ loguru installed')
    print('')
    print('✅ All essential packages installed successfully!')
except ImportError as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Setup completed successfully!"
    echo "========================================="
    echo ""
    echo "To run the tracker:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Run the tracker: python main.py --help"
    echo ""
    echo "Example commands:"
    echo "  python main.py                    # Run analysis for today"
    echo "  python main.py --date 2024-01-15  # Run for specific date"
    echo "  python main.py --lookback 30      # Use 30-day lookback"
    echo "  python main.py --multiplier 3     # Detect 3x spikes"
    echo "  python main.py --index NIFTY_50   # Filter NIFTY 50 stocks only"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Please check the errors above."
    echo ""
    echo "Try manual installation:"
    echo "  source venv/bin/activate"
    echo "  pip install click pandas numpy requests pyyaml"
    echo ""
fi
