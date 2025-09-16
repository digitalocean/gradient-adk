#!/bin/bash

# Test script to simulate the GitHub Actions workflow locally
# This helps catch issues before pushing tags

set -e

echo "🧪 Testing GitHub Actions workflow locally..."

# Clean up from previous runs
echo "🧹 Cleaning up previous builds..."
rm -rf dist build

# Download doctl binaries
echo "⬇️  Downloading doctl binaries..."
chmod +x download_doctl.sh
./download_doctl.sh

# Build executable (simulate one platform)
echo "🔨 Building executable..."
pyinstaller gradient.spec --distpath dist

# Test the executable
echo "✅ Testing executable..."
if [ -f "dist/gradient" ]; then
    ./dist/gradient --help > /dev/null
    echo "✅ Executable works correctly!"
    
    # Test auth commands (requires doctl integration)
    echo "🔐 Testing auth integration..."
    ./dist/gradient auth --help > /dev/null
    echo "✅ Auth commands work correctly!"
    
    echo "📏 Executable size: $(ls -lh dist/gradient | awk '{print $5}')"
else
    echo "❌ Executable not found!"
    exit 1
fi

echo "🎉 All tests passed! Ready for release."
