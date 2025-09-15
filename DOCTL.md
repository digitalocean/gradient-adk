# doctl Integration

The Gradient CLI includes embedded doctl binaries for seamless DigitalOcean integration.

## What's Included

The build process automatically downloads and embeds doctl v1.109.0 for all supported platforms:

- **macOS (Intel)**: darwin_amd64
- **macOS (Apple Silicon)**: darwin_arm64  
- **Linux (amd64)**: linux_amd64
- **Linux (arm64)**: linux_arm64
- **Windows (amd64)**: windows_amd64

## How It Works

1. **Automatic Resolution**: The CLI first tries to use the embedded doctl binary
2. **Fallback**: If the embedded binary isn't found, it falls back to system doctl on PATH
3. **Platform Detection**: Automatically selects the correct binary for the current platform

## Updating doctl

To update the embedded doctl version:

1. Edit `download_doctl.sh` and update `DOCTL_VERSION`
2. Run `./download_doctl.sh` to download new binaries
3. Rebuild the executable with `./build.sh`

## File Locations

```
gradient/_vendor/doctl/
├── darwin_amd64/doctl
├── darwin_arm64/doctl
├── linux_amd64/doctl
├── linux_arm64/doctl
└── windows_amd64/doctl.exe
```

## Authentication Commands

The embedded doctl enables all auth commands:

```bash
gradient auth init         # Initialize authentication
gradient auth list         # List auth contexts  
gradient auth switch       # Switch auth context
gradient auth remove       # Remove auth context
```

All doctl commands are available with the same functionality as the standalone doctl CLI.
