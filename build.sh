#!/bin/bash

# Local build script for testing
# Usage: ./build.sh

echo "Installing build dependencies..."
pip install pyinstaller

echo "Downloading doctl binaries for all platforms..."
chmod +x download_doctl.sh
./download_doctl.sh

echo "Building executable..."
pyinstaller gradient.spec --clean

echo "Testing executable..."
./dist/gradient --help

echo "Build complete! Executable is at: ./dist/gradient"
echo "Size:" $(ls -lh dist/gradient | awk '{print $5}')
