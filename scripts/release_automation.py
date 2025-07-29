#!/usr/bin/env python3
"""
Automated Release Management for Fast VLM On-Device Kit

Comprehensive release automation for MATURING SDLC environments.
"""

import re
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReleaseConfig:
    """Release configuration"""
    version_pattern: str = r"^\d+\.\d+\.\d+$"
    changelog_path: Path = Path("CHANGELOG.md")
    version_files: List[Path] = None
    pre_release_commands: List[str] = None
    post_release_commands: List[str] = None
    
    def __post_init__(self):
        if self.version_files is None:
            self.version_files = [
                Path("pyproject.toml"),
                Path("ios/Package.swift"),
                Path("src/fast_vlm_ondevice/__init__.py")
            ]
        if self.pre_release_commands is None:
            self.pre_release_commands = [
                "python -m pytest tests/",
                "python -m black --check src/",
                "python -m mypy src/",
                "swift test --package-path ios/"
            ]
        if self.post_release_commands is None:
            self.post_release_commands = [
                "python -m build",
                "twine check dist/*"
            ]


class VersionManager:
    """Manages version bumping and validation"""
    
    def __init__(self, config: ReleaseConfig):
        self.config = config
    
    def get_current_version(self) -> str:
        """Extract current version from pyproject.toml"""
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        content = pyproject_path.read_text()
        version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
        
        if not version_match:
            raise ValueError("Version not found in pyproject.toml")
        
        return version_match.group(1)
    
    def validate_version(self, version: str) -> bool:
        """Validate version format"""
        return bool(re.match(self.config.version_pattern, version))
    
    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Bump version based on type (major, minor, patch)"""
        major, minor, patch = map(int, current_version.split('.'))
        
        if bump_type == "major":
            return f"{major + 1}.0.0"
        elif bump_type == "minor":
            return f"{major}.{minor + 1}.0"
        elif bump_type == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
    
    def update_version_files(self, new_version: str) -> List[Path]:
        """Update version in all configured files"""
        updated_files = []
        
        for file_path in self.config.version_files:
            if not file_path.exists():
                logger.warning(f"Version file not found: {file_path}")
                continue
            
            content = file_path.read_text()
            original_content = content
            
            # Update based on file type
            if file_path.suffix == ".toml":
                content = re.sub(
                    r'version\s*=\s*"[^"]+"',
                    f'version = "{new_version}"',
                    content
                )
            elif file_path.suffix == ".py":
                content = re.sub(
                    r'__version__\s*=\s*"[^"]+"',
                    f'__version__ = "{new_version}"',
                    content
                )
            elif file_path.suffix == ".swift":
                # Update Swift package version if present
                content = re.sub(
                    r'let version\s*=\s*"[^"]+"',
                    f'let version = "{new_version}"',
                    content
                )
            
            if content != original_content:
                file_path.write_text(content)
                updated_files.append(file_path)
                logger.info(f"Updated version in {file_path}")
            
        return updated_files


class ChangelogManager:
    """Manages changelog generation and updates"""
    
    def __init__(self, config: ReleaseConfig):
        self.config = config
    
    def get_git_log_since_tag(self, since_tag: str) -> List[Dict]:
        """Get git commits since last tag"""
        try:
            cmd = [
                "git", "log", f"{since_tag}..HEAD",
                "--pretty=format:%H|%s|%an|%ad",
                "--date=short"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    hash_val, subject, author, date = line.split('|', 3)
                    commits.append({
                        'hash': hash_val[:8],
                        'subject': subject,
                        'author': author,
                        'date': date
                    })
            
            return commits
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get git log: {e}")
            return []
    
    def categorize_commits(self, commits: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize commits by type"""
        categories = {
            'features': [],
            'fixes': [],
            'docs': [],
            'performance': [],
            'security': [],
            'maintenance': [],
            'other': []
        }
        
        for commit in commits:
            subject = commit['subject'].lower()
            
            if any(keyword in subject for keyword in ['feat', 'feature', 'add']):
                categories['features'].append(commit)
            elif any(keyword in subject for keyword in ['fix', 'bug', 'resolve']):
                categories['fixes'].append(commit)
            elif any(keyword in subject for keyword in ['doc', 'readme', 'guide']):
                categories['docs'].append(commit)
            elif any(keyword in subject for keyword in ['perf', 'optimize', 'speed']):
                categories['performance'].append(commit)
            elif any(keyword in subject for keyword in ['sec', 'security', 'vuln']):
                categories['security'].append(commit)
            elif any(keyword in subject for keyword in ['refactor', 'clean', 'maintain']):
                categories['maintenance'].append(commit)
            else:
                categories['other'].append(commit)
        
        return categories
    
    def generate_changelog_section(self, version: str, categorized_commits: Dict[str, List[Dict]]) -> str:
        """Generate changelog section for new version"""
        date = datetime.now().strftime("%Y-%m-%d")
        changelog = f"\n## [{version}] - {date}\n\n"
        
        category_headers = {
            'features': '### 🚀 New Features',
            'fixes': '### 🐛 Bug Fixes',
            'docs': '### 📚 Documentation',
            'performance': '### ⚡ Performance',
            'security': '### 🔒 Security',
            'maintenance': '### 🔧 Maintenance',
            'other': '### Other Changes'
        }
        
        for category, commits in categorized_commits.items():
            if commits:
                changelog += f"{category_headers[category]}\n\n"
                for commit in commits:
                    changelog += f"- {commit['subject']} ([{commit['hash']}])\n"
                changelog += "\n"
        
        return changelog
    
    def update_changelog(self, version: str, changelog_section: str):
        """Update CHANGELOG.md with new version"""
        changelog_path = self.config.changelog_path
        
        if changelog_path.exists():
            content = changelog_path.read_text()
            
            # Find insertion point (after header, before first version)
            lines = content.split('\n')
            insert_index = 0
            
            for i, line in enumerate(lines):
                if line.startswith('## [') or line.startswith('## '):
                    insert_index = i
                    break
            
            # Insert new changelog section
            lines.insert(insert_index, changelog_section.rstrip())
            updated_content = '\n'.join(lines)
        else:
            # Create new changelog
            header = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
"""
            updated_content = header + changelog_section
        
        changelog_path.write_text(updated_content)
        logger.info(f"Updated {changelog_path}")


class ReleaseAutomation:
    """Main release automation orchestrator"""
    
    def __init__(self, config: ReleaseConfig = None):
        self.config = config or ReleaseConfig()
        self.version_manager = VersionManager(self.config)
        self.changelog_manager = ChangelogManager(self.config)
    
    def run_pre_release_checks(self) -> bool:
        """Run pre-release validation commands"""
        logger.info("Running pre-release checks...")
        
        for command in self.config.pre_release_commands:
            try:
                logger.info(f"Running: {command}")
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"✓ {command} passed")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ {command} failed: {e.stderr}")
                return False
        
        return True
    
    def run_post_release_tasks(self) -> bool:
        """Run post-release tasks"""
        logger.info("Running post-release tasks...")
        
        for command in self.config.post_release_commands:
            try:
                logger.info(f"Running: {command}")
                subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"✓ {command} completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ {command} failed: {e.stderr}")
                return False
        
        return True
    
    def create_release(self, bump_type: str = "patch", dry_run: bool = False) -> bool:
        """Create new release"""
        try:
            # Get current version
            current_version = self.version_manager.get_current_version()
            logger.info(f"Current version: {current_version}")
            
            # Calculate new version
            new_version = self.version_manager.bump_version(current_version, bump_type)
            logger.info(f"New version: {new_version}")
            
            if not self.version_manager.validate_version(new_version):
                raise ValueError(f"Invalid version format: {new_version}")
            
            if dry_run:
                logger.info("DRY RUN - would perform release steps")
                return True
            
            # Run pre-release checks
            if not self.run_pre_release_checks():
                logger.error("Pre-release checks failed")
                return False
            
            # Get last tag for changelog
            try:
                last_tag = subprocess.run(
                    ["git", "describe", "--tags", "--abbrev=0"],
                    capture_output=True,
                    text=True,
                    check=True
                ).stdout.strip()
            except subprocess.CalledProcessError:
                last_tag = ""
                logger.warning("No previous tags found")
            
            # Generate changelog
            if last_tag:
                commits = self.changelog_manager.get_git_log_since_tag(last_tag)
                categorized_commits = self.changelog_manager.categorize_commits(commits)
                changelog_section = self.changelog_manager.generate_changelog_section(
                    new_version, categorized_commits
                )
                self.changelog_manager.update_changelog(new_version, changelog_section)
            
            # Update version files
            updated_files = self.version_manager.update_version_files(new_version)
            
            # Create git commit
            files_to_commit = updated_files + [self.config.changelog_path]
            existing_files = [f for f in files_to_commit if f.exists()]
            
            if existing_files:
                subprocess.run(["git", "add"] + [str(f) for f in existing_files], check=True)
                subprocess.run([
                    "git", "commit", "-m", f"chore: release version {new_version}"
                ], check=True)
            
            # Create git tag
            subprocess.run([
                "git", "tag", "-a", f"v{new_version}",
                "-m", f"Release version {new_version}"
            ], check=True)
            
            # Run post-release tasks
            if not self.run_post_release_tasks():
                logger.warning("Some post-release tasks failed")
            
            logger.info(f"✓ Release {new_version} created successfully")
            logger.info("Next steps:")
            logger.info(f"  git push origin main")
            logger.info(f"  git push origin v{new_version}")
            logger.info(f"  Create GitHub release from tag v{new_version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Release failed: {e}")
            return False


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated release management")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to release configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = ReleaseConfig()
    if args.config and args.config.exists():
        logger.info(f"Loading config from {args.config}")
        # In a real implementation, load config from file
    
    # Run release process
    automation = ReleaseAutomation(config)
    success = automation.create_release(args.bump_type, args.dry_run)
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()