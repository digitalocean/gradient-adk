# GitHub Workflow Fixes

## Issues Found and Fixed

### 1. PyInstaller `--name` Option Error
**Problem**: The workflow was trying to use `--name` option with a `.spec` file, which is not allowed.

**Error**: 
```
ERROR: option(s) not allowed:
  --name
makespec options not valid when a .spec file is given
```

**Fix**: Removed the `--name ${{ matrix.artifact_name }}` parameter from the PyInstaller command.

**Before**:
```yaml
pyinstaller gradient.spec --distpath dist --name ${{ matrix.artifact_name }}
```

**After**:
```yaml
pyinstaller gradient.spec --distpath dist
```

### 2. Invalid Runner Configuration
**Problem**: Both build and release jobs were using `runs-on: prod` which is not a valid GitHub runner.

**Fix**: 
- Build job: Use `runs-on: ${{ matrix.os }}` to use the matrix OS
- Release job: Use `runs-on: ubuntu-latest` for consistency

### 3. Deprecated GitHub Actions
**Problem**: Using old deprecated actions (`actions/create-release@v1`, `actions/upload-release-asset@v1`) which may have reliability issues.

**Fix**: Replaced with modern `softprops/action-gh-release@v1` which handles both release creation and file uploads in one step.

## Testing

Created `test_workflow.sh` script to validate the workflow locally before pushing tags.

## Workflow Now:
1. ✅ Uses correct PyInstaller command without `--name` 
2. ✅ Uses valid GitHub runners
3. ✅ Uses modern, reliable GitHub actions
4. ✅ Includes doctl binary download step
5. ✅ Builds for all platforms (Linux, macOS Intel, macOS ARM64, Windows)
6. ✅ Creates releases with proper asset uploads
