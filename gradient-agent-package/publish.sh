#!/usr/bin/env bash

# Gradient Agent Publishing Script

set -e

echo "🚀 Publishing Gradient Agent to PyPI"

# Check if we're in a git repo and everything is committed
if [[ -n $(git status --porcelain) ]]; then
    echo "❌ Working directory is not clean. Please commit all changes first."
    exit 1
fi

# Check if we have the required tools
command -v python >/dev/null 2>&1 || { echo "❌ Python is required but not installed."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "❌ Git is required but not installed."; exit 1; }

# Install/upgrade build tools
echo "📦 Installing build tools..."
python -m pip install --upgrade build twine

# Clean any previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Run tests first
echo "🧪 Running tests..."
if command -v pytest >/dev/null 2>&1; then
    python -m pytest tests/ -v
else
    echo "⚠️  pytest not found, skipping tests"
fi

# Build the package
echo "🔨 Building package..."
python -m build

# Check the build
echo "🔍 Checking package..."
python -m twine check dist/*

echo "✅ Build completed successfully!"
echo ""
echo "📝 Built files:"
ls -la dist/

echo ""
echo "🚀 Ready to publish!"
echo ""
echo "To publish to TestPyPI (recommended first):"
echo "python -m twine upload --repository testpypi dist/*"
echo ""
echo "To publish to PyPI:"
echo "python -m twine upload dist/*"
echo ""
echo "Make sure you have your PyPI API tokens configured:"
echo "~/.pypirc or use --username __token__ --password <your-token>"
