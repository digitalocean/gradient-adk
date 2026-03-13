"""Utilities for reading .gradientignore files."""

from __future__ import annotations

from pathlib import Path

from gradient_adk.logging import get_logger

logger = get_logger(__name__)

GRADIENTIGNORE_PATH = ".gradient/.gradientignore"

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    # Archives
    "*.zip",
    # Python
    "__pycache__/",
    "*.pyc",
    "*.egg-info",
    "dist/",
    "build/",
    # Virtual environments
    "env/",
    "venv/",
    ".venv/",
    # Version control
    ".git/",
    # IDE / caches
    ".pytest_cache/",
    ".mypy_cache/",
]

DEFAULT_GRADIENTIGNORE_CONTENT = """\
# Files and directories to exclude from deployment uploads.
# Lines starting with # are comments. Blank lines are ignored.
#
# Patterns:
#   dir_name/     - exclude any directory with this name (anywhere in tree)
#   *.ext         - exclude files matching this extension
#   exact_name    - exclude exact file/directory name matches

# Archives
*.zip

# Python
__pycache__/
*.pyc
*.egg-info
dist/
build/

# Virtual environments
env/
venv/
.venv/

# Version control
.git/

# IDE / caches
.pytest_cache/
.mypy_cache/
"""


def load_gradientignore(project_dir: Path) -> list[str]:
    """Load exclude patterns from .gradient/.gradientignore.

    If the file does not exist, returns the default exclude patterns
    so that existing projects without a .gradientignore behave the same.

    Args:
        project_dir: Root directory of the project.

    Returns:
        List of exclude pattern strings.
    """
    ignore_file = project_dir / GRADIENTIGNORE_PATH

    if not ignore_file.exists():
        logger.debug(
            f"No .gradientignore found at {ignore_file}, using default patterns"
        )
        return list(DEFAULT_EXCLUDE_PATTERNS)

    logger.debug(f"Loading .gradientignore from {ignore_file}")

    patterns: list[str] = []
    for line in ignore_file.read_text().splitlines():
        stripped = line.strip()
        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)

    logger.debug(f"Loaded {len(patterns)} patterns from .gradientignore")
    return patterns


def ensure_gradientignore(project_dir: Path) -> None:
    """Create .gradient/.gradientignore with defaults if it doesn't exist.

    Args:
        project_dir: Root directory of the project.
    """
    gradient_dir = project_dir / ".gradient"
    gradient_dir.mkdir(exist_ok=True)

    ignore_file = gradient_dir / ".gradientignore"
    if not ignore_file.exists():
        ignore_file.write_text(DEFAULT_GRADIENTIGNORE_CONTENT)
        logger.debug(f"Created default .gradientignore at {ignore_file}")
