## Gradient CLI Release Template

### What's Changed

- New features
- Bug fixes  
- Performance improvements
- Documentation updates

### Installation

Download the appropriate binary for your platform:

- **Linux (x86_64)**: `gradient-linux-x86_64`
- **macOS (Intel)**: `gradient-macos-x86_64`  
- **macOS (Apple Silicon)**: `gradient-macos-arm64`
- **Windows (x86_64)**: `gradient-windows-x86_64.exe`

### Usage

1. Download the binary for your platform
2. Make it executable (Linux/Mac): `chmod +x gradient-*`
3. Run: `./gradient-* --help` (or `gradient-windows-x86_64.exe --help` on Windows)

### Quick Start

```bash
# Initialize an agent
./gradient-* agent init

# Run the agent locally  
./gradient-* agent run
```

### Requirements

- No Python installation required! 
- Standalone executables include all dependencies

For full documentation, see [USAGE.md](https://github.com/your-repo/gradient-agent/blob/main/USAGE.md).
