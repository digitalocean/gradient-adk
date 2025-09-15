"""
Universal runner that finds and runs @entrypoint decorated functions.
"""
import os
import sys
import importlib
import pkgutil
from pathlib import Path
import typer
from gradient.sdk.decorator import get_app, run_server


def find_entrypoint_function():
    """
    Recursively search for Python modules and find @entrypoint decorated functions.
    """
    # Add current directory to Python path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Look for Python files in the current directory and subdirectories
    python_files = []
    for py_file in current_dir.rglob("*.py"):
        # Skip files in .gradient, __pycache__, and other common exclusions
        if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
            continue
        python_files.append(py_file)
    
    # Try to import each Python file and check for entrypoint
    for py_file in python_files:
        try:
            # Convert file path to module name
            relative_path = py_file.relative_to(current_dir)
            module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
            
            # Skip if it looks like a test file or other non-main files
            if any(skip in module_name.lower() for skip in ['test', 'tests', '__pycache__']):
                continue
            
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Check if this module has registered an entrypoint
            try:
                app = get_app()
                print(f"Found entrypoint in module: {module_name}")
                return app
            except RuntimeError:
                # No entrypoint in this module, continue searching
                continue
                
        except Exception as e:
            # Skip modules that can't be imported
            continue
    
    return None


def main():
    """Main entry point for the universal runner."""
    print("Starting Gradient Agent...")
    print("Searching for @entrypoint decorated functions...")
    
    app = find_entrypoint_function()
    
    if app is None:
        print("Error: No @entrypoint decorated function found in your codebase.")
        print("Please make sure you have:")
        print("1. Imported gradient.sdk: from gradient.sdk import entrypoint")
        print("2. Decorated a function: @entrypoint")
        print("3. The decorated function is in a .py file in your project")
        sys.exit(1)
    
    print("Found entrypoint! Starting FastAPI server...")
    print("Server will be available at http://0.0.0.0:8080")
    
    # Run the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
