# Gradient CLI

`gradient` is a CLI that enables you to create, test, and deploy agents on DigitalOcean's Gradient™ AI platform.

## Quick Install

## Quick Install

### Pre-built Binaries (Recommended)
Download the latest standalone executable for your platform from the [Releases](https://github.com/your-repo/gradient-agent/releases) page:

- **Linux (x86_64)**: `gradient-linux-x86_64`
- **macOS (Intel)**: `gradient-macos-x86_64`  
- **macOS (Apple Silicon)**: `gradient-macos-arm64`
- **Windows (x86_64)**: `gradient-windows-x86_64.exe`

Make it executable and run:
```bash
chmod +x gradient-*  # Linux/Mac only
./gradient-* --help
```

### From Source
```bash
git clone https://github.com/your-repo/gradient-agent
cd gradient-agent
pip install -e .
gradient --help