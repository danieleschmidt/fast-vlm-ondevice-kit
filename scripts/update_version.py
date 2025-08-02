#!/usr/bin/env python3
"""
Update version across project files for semantic release.
"""

import sys
import re
from pathlib import Path


def update_pyproject_version(version: str):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Update version line
    updated_content = re.sub(
        r'version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        content
    )
    
    pyproject_path.write_text(updated_content)
    print(f"✅ Updated pyproject.toml to version {version}")


def update_init_version(version: str):
    """Update version in __init__.py."""
    init_path = Path("src/fast_vlm_ondevice/__init__.py")
    
    if not init_path.exists():
        # Create __init__.py if it doesn't exist
        init_content = f'"""FastVLM On-Device Kit."""\n\n__version__ = "{version}"\n'
        init_path.write_text(init_content)
    else:
        content = init_path.read_text()
        
        # Update or add version
        if "__version__" in content:
            updated_content = re.sub(
                r'__version__\s*=\s*"[^"]*"',
                f'__version__ = "{version}"',
                content
            )
        else:
            updated_content = content + f'\n__version__ = "{version}"\n'
        
        init_path.write_text(updated_content)
    
    print(f"✅ Updated __init__.py to version {version}")


def update_package_json_version(version: str):
    """Update version in package.json if it exists."""
    package_json_path = Path("package.json")
    
    if package_json_path.exists():
        import json
        
        with open(package_json_path) as f:
            package_data = json.load(f)
        
        package_data["version"] = version
        
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)
        
        print(f"✅ Updated package.json to version {version}")


def main():
    """Main function to update version across project files."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    
    # Validate version format (basic semver check)
    if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9-]+)?$', version):
        print(f"❌ Invalid version format: {version}")
        print("Expected format: X.Y.Z or X.Y.Z-prerelease")
        sys.exit(1)
    
    print(f"🔄 Updating project to version {version}...")
    
    try:
        update_pyproject_version(version)
        update_init_version(version)
        update_package_json_version(version)
        
        print(f"🎉 Successfully updated all files to version {version}")
        
    except Exception as e:
        print(f"❌ Error updating version: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()