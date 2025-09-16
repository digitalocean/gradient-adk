#!/usr/bin/env bash

# Development setup for Gradient Agent

set -e

echo "🛠️  Setting up Gradient Agent development environment"

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "📍 Using Python $python_version"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode with dev dependencies
echo "🔧 Installing package in development mode..."
pip install -e ".[dev]"

# Install additional development tools
echo "🛠️  Installing development tools..."
pip install build twine pytest-cov pre-commit

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 Quick start:"
echo "source venv/bin/activate  # Activate environment"
echo "python examples/simple_agent.py  # Run example"
echo "pytest tests/ -v  # Run tests"
echo "./publish.sh  # Build and prepare for publishing"
echo ""
echo "📁 Package structure:"
find . -name "*.py" -not -path "./venv/*" | head -10
