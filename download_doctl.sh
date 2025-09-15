#!/bin/bash

# Download doctl binaries for all supported platforms
# This script downloads the official doctl releases from GitHub

set -e

DOCTL_VERSION="1.109.0"  # Update this as needed
BASE_URL="https://github.com/digitalocean/doctl/releases/download/v${DOCTL_VERSION}"

# Create directories
mkdir -p gradient/_vendor/doctl/darwin_amd64
mkdir -p gradient/_vendor/doctl/darwin_arm64
mkdir -p gradient/_vendor/doctl/linux_amd64
mkdir -p gradient/_vendor/doctl/linux_arm64
mkdir -p gradient/_vendor/doctl/windows_amd64

echo "Downloading doctl v${DOCTL_VERSION} for all platforms..."

# macOS Intel (amd64)
echo "Downloading macOS Intel (amd64)..."
curl -L "${BASE_URL}/doctl-${DOCTL_VERSION}-darwin-amd64.tar.gz" -o /tmp/doctl-darwin-amd64.tar.gz
tar -xzf /tmp/doctl-darwin-amd64.tar.gz -C /tmp
mv /tmp/doctl gradient/_vendor/doctl/darwin_amd64/doctl
chmod +x gradient/_vendor/doctl/darwin_amd64/doctl

# macOS Apple Silicon (arm64)  
echo "Downloading macOS Apple Silicon (arm64)..."
curl -L "${BASE_URL}/doctl-${DOCTL_VERSION}-darwin-arm64.tar.gz" -o /tmp/doctl-darwin-arm64.tar.gz
tar -xzf /tmp/doctl-darwin-arm64.tar.gz -C /tmp
mv /tmp/doctl gradient/_vendor/doctl/darwin_arm64/doctl
chmod +x gradient/_vendor/doctl/darwin_arm64/doctl

# Linux amd64
echo "Downloading Linux amd64..."
curl -L "${BASE_URL}/doctl-${DOCTL_VERSION}-linux-amd64.tar.gz" -o /tmp/doctl-linux-amd64.tar.gz
tar -xzf /tmp/doctl-linux-amd64.tar.gz -C /tmp
mv /tmp/doctl gradient/_vendor/doctl/linux_amd64/doctl
chmod +x gradient/_vendor/doctl/linux_amd64/doctl

# Linux arm64
echo "Downloading Linux arm64..."
curl -L "${BASE_URL}/doctl-${DOCTL_VERSION}-linux-arm64.tar.gz" -o /tmp/doctl-linux-arm64.tar.gz
tar -xzf /tmp/doctl-linux-arm64.tar.gz -C /tmp
mv /tmp/doctl gradient/_vendor/doctl/linux_arm64/doctl
chmod +x gradient/_vendor/doctl/linux_arm64/doctl

# Windows amd64
echo "Downloading Windows amd64..."
curl -L "${BASE_URL}/doctl-${DOCTL_VERSION}-windows-amd64.zip" -o /tmp/doctl-windows-amd64.zip
cd /tmp && unzip -o doctl-windows-amd64.zip
cd -
mv /tmp/doctl.exe gradient/_vendor/doctl/windows_amd64/doctl.exe

# Clean up
rm -f /tmp/doctl-*.tar.gz /tmp/doctl-*.zip /tmp/doctl

echo "✅ All doctl binaries downloaded successfully!"
echo "Files:"
find gradient/_vendor/doctl -name "doctl*" -type f