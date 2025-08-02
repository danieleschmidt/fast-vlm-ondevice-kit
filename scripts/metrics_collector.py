#!/usr/bin/env python3
"""
Automated metrics collection script for FastVLM On-Device Kit.
Collects and reports various project metrics including code quality,
performance, and community engagement.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MetricsCollector:
    """Collects comprehensive project metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics = {}
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        self.metrics = {
            "collection_timestamp": datetime.utcnow().isoformat(),
            "code_metrics": self.collect_code_metrics(),
            "git_metrics": self.collect_git_metrics(),
            "test_metrics": self.collect_test_metrics(),
            "performance_metrics": self.collect_performance_metrics(),
            "dependency_metrics": self.collect_dependency_metrics(),
            "quality_metrics": self.collect_quality_metrics(),
            "automation_metrics": self.collect_automation_metrics()
        }
        
        return self.metrics
    
    def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code-related metrics."""
        print("  📊 Collecting code metrics...")
        
        try:
            # Count lines of code by language
            loc_by_language = {}
            
            # Python files
            python_files = list(self.repo_path.rglob("*.py"))
            python_loc = sum(len(f.read_text().splitlines()) for f in python_files if f.exists())
            loc_by_language["python"] = python_loc
            
            # Swift files
            swift_files = list(self.repo_path.rglob("*.swift"))
            swift_loc = sum(len(f.read_text().splitlines()) for f in swift_files if f.exists())
            loc_by_language["swift"] = swift_loc
            
            # Markdown files
            md_files = list(self.repo_path.rglob("*.md"))
            md_loc = sum(len(f.read_text().splitlines()) for f in md_files if f.exists())
            loc_by_language["markdown"] = md_loc
            
            # YAML files
            yaml_files = list(self.repo_path.rglob("*.yml")) + list(self.repo_path.rglob("*.yaml"))
            yaml_loc = sum(len(f.read_text().splitlines()) for f in yaml_files if f.exists())
            loc_by_language["yaml"] = yaml_loc
            
            loc_by_language["total"] = sum(loc_by_language.values())
            
            # Count files
            file_counts = {
                "python_files": len(python_files),
                "swift_files": len(swift_files),
                "test_files": len(list(self.repo_path.rglob("test_*.py"))),
                "documentation_files": len(md_files)
            }
            
            return {
                "lines_of_code": loc_by_language,
                "file_counts": file_counts,
                "code_complexity": self._analyze_complexity()
            }
            
        except Exception as e:
            print(f"    ⚠️  Error collecting code metrics: {e}")
            return {"error": str(e)}
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        print("  🔀 Collecting Git metrics...")
        
        try:
            # Get recent commit activity
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=30 days ago"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            recent_commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Get contributor count
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--since=30 days ago"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            contributors = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Get branch count
            result = subprocess.run(
                ["git", "branch", "-a"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            branches = len([line for line in result.stdout.split('\n') if line.strip() and not line.strip().startswith('*')])
            
            # Get repository size
            result = subprocess.run(
                ["du", "-sh", ".git"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            repo_size = result.stdout.split()[0] if result.stdout.strip() else "unknown"
            
            return {
                "commits_last_30_days": recent_commits,
                "active_contributors_last_30_days": contributors,
                "total_branches": branches,
                "repository_size": repo_size,
                "last_commit": self._get_last_commit_info()
            }
            
        except Exception as e:
            print(f"    ⚠️  Error collecting Git metrics: {e}")
            return {"error": str(e)}
    
    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect testing-related metrics."""
        print("  🧪 Collecting test metrics...")
        
        try:
            # Count test files and functions
            test_files = list(self.repo_path.rglob("test_*.py"))
            test_count = 0
            
            for test_file in test_files:
                try:
                    content = test_file.read_text()
                    test_count += content.count("def test_")
                except Exception:
                    continue
            
            # Try to get coverage info if available
            coverage_info = self._get_coverage_info()
            
            return {
                "test_files": len(test_files),
                "total_tests": test_count,
                "coverage": coverage_info,
                "test_frameworks": self._detect_test_frameworks()
            }
            
        except Exception as e:
            print(f"    ⚠️  Error collecting test metrics: {e}")
            return {"error": str(e)}
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        print("  ⚡ Collecting performance metrics...")
        
        try:
            # Check for benchmark files
            benchmark_files = list(self.repo_path.rglob("*benchmark*.py"))
            performance_tests = list(self.repo_path.rglob("*performance*.py"))
            
            # Try to load performance data if available
            perf_data = {}
            perf_results_file = self.repo_path / "benchmarks" / "results" / "latest.json"
            if perf_results_file.exists():
                try:
                    perf_data = json.loads(perf_results_file.read_text())
                except Exception:
                    pass
            
            return {
                "benchmark_files": len(benchmark_files),
                "performance_test_files": len(performance_tests),
                "latest_benchmark_results": perf_data,
                "has_performance_monitoring": len(benchmark_files) > 0
            }
            
        except Exception as e:
            print(f"    ⚠️  Error collecting performance metrics: {e}")
            return {"error": str(e)}
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        print("  📦 Collecting dependency metrics...")
        
        try:
            dependencies = {}
            
            # Python dependencies
            requirements_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
            for req_file in requirements_files:
                file_path = self.repo_path / req_file
                if file_path.exists():
                    if req_file.endswith(".txt"):
                        deps = self._parse_requirements_txt(file_path)
                    else:
                        deps = self._parse_pyproject_toml(file_path)
                    dependencies[req_file] = deps
            
            # Swift dependencies
            swift_package = self.repo_path / "ios" / "Package.swift"
            if swift_package.exists():
                dependencies["Package.swift"] = self._parse_swift_package(swift_package)
            
            # Check for outdated dependencies
            outdated_info = self._check_outdated_dependencies()
            
            return {
                "dependency_files": dependencies,
                "outdated_check": outdated_info,
                "total_dependencies": sum(len(deps) for deps in dependencies.values() if isinstance(deps, list))
            }
            
        except Exception as e:
            print(f"    ⚠️  Error collecting dependency metrics: {e}")
            return {"error": str(e)}
    
    def collect_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("  ✨ Collecting quality metrics...")
        
        try:
            quality_info = {
                "linting_tools": self._detect_quality_tools(),
                "pre_commit_hooks": self._check_pre_commit(),
                "security_tools": self._detect_security_tools(),
                "documentation_coverage": self._calculate_doc_coverage()
            }
            
            return quality_info
            
        except Exception as e:
            print(f"    ⚠️  Error collecting quality metrics: {e}")
            return {"error": str(e)}
    
    def collect_automation_metrics(self) -> Dict[str, Any]:
        """Collect automation-related metrics."""
        print("  🤖 Collecting automation metrics...")
        
        try:
            # Check for CI/CD files
            ci_files = []
            github_workflows = self.repo_path / ".github" / "workflows"
            if github_workflows.exists():
                ci_files = list(github_workflows.glob("*.yml")) + list(github_workflows.glob("*.yaml"))
            
            # Check for automation scripts
            script_dir = self.repo_path / "scripts"
            automation_scripts = list(script_dir.glob("*.py")) if script_dir.exists() else []
            
            # Check for Docker
            docker_files = list(self.repo_path.glob("Dockerfile*")) + list(self.repo_path.glob("docker-compose*.yml"))
            
            return {
                "ci_cd_workflows": len(ci_files),
                "automation_scripts": len(automation_scripts),
                "docker_files": len(docker_files),
                "has_makefile": (self.repo_path / "Makefile").exists(),
                "has_taskfile": (self.repo_path / "Taskfile.yml").exists(),
                "workflow_files": [f.name for f in ci_files]
            }
            
        except Exception as e:
            print(f"    ⚠️  Error collecting automation metrics: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity."""
        try:
            # Basic complexity analysis - count functions, classes, etc.
            python_files = list(self.repo_path.rglob("*.py"))
            
            total_functions = 0
            total_classes = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    total_functions += content.count("def ")
                    total_classes += content.count("class ")
                except Exception:
                    continue
            
            return {
                "total_functions": total_functions,
                "total_classes": total_classes,
                "average_functions_per_file": total_functions / len(python_files) if python_files else 0
            }
        except Exception:
            return {}
    
    def _get_last_commit_info(self) -> Dict[str, str]:
        """Get information about the last commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%an|%ad|%s"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.stdout.strip():
                parts = result.stdout.strip().split('|', 3)
                return {
                    "sha": parts[0][:8],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                }
        except Exception:
            pass
        return {}
    
    def _get_coverage_info(self) -> Dict[str, Any]:
        """Get test coverage information."""
        try:
            # Try to read coverage report
            coverage_file = self.repo_path / ".coverage"
            if coverage_file.exists():
                # Run coverage report
                result = subprocess.run(
                    ["coverage", "report", "--format=json"],
                    capture_output=True, text=True, cwd=self.repo_path
                )
                if result.returncode == 0:
                    return json.loads(result.stdout)
        except Exception:
            pass
        return {"status": "not_available"}
    
    def _detect_test_frameworks(self) -> list:
        """Detect testing frameworks in use."""
        frameworks = []
        
        # Check pyproject.toml and requirements files
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                frameworks.append("pytest")
            if "unittest" in content:
                frameworks.append("unittest")
        
        # Check for specific test files
        if list(self.repo_path.rglob("conftest.py")):
            frameworks.append("pytest")
        
        return frameworks
    
    def _parse_requirements_txt(self, file_path: Path) -> list:
        """Parse requirements.txt file."""
        try:
            lines = file_path.read_text().splitlines()
            deps = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    deps.append(line.split('==')[0].split('>=')[0].split('<=')[0])
            return deps
        except Exception:
            return []
    
    def _parse_pyproject_toml(self, file_path: Path) -> Dict[str, Any]:
        """Parse pyproject.toml file."""
        try:
            import re
            content = file_path.read_text()
            
            # Simple regex to extract dependencies
            deps = re.findall(r'"([^"]+)>=?[^"]*"', content)
            return {"dependencies": deps, "format": "pyproject.toml"}
        except Exception:
            return {}
    
    def _parse_swift_package(self, file_path: Path) -> Dict[str, Any]:
        """Parse Swift Package.swift file."""
        try:
            content = file_path.read_text()
            # Simple parsing - look for dependencies
            import re
            deps = re.findall(r'\.package\([^)]+\)', content)
            return {"dependencies": len(deps), "format": "swift_package"}
        except Exception:
            return {}
    
    def _check_outdated_dependencies(self) -> Dict[str, Any]:
        """Check for outdated dependencies."""
        try:
            # Try pip list --outdated
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                return {"outdated_count": len(outdated), "packages": outdated}
        except Exception:
            pass
        return {"status": "check_failed"}
    
    def _detect_quality_tools(self) -> list:
        """Detect code quality tools in use."""
        tools = []
        
        # Check configuration files
        config_files = {
            "black": [".black", "pyproject.toml"],
            "isort": [".isort.cfg", "pyproject.toml"],
            "mypy": ["mypy.ini", "pyproject.toml"],
            "flake8": [".flake8", "setup.cfg"],
            "pylint": [".pylintrc", "pyproject.toml"],
            "bandit": [".bandit", "pyproject.toml"]
        }
        
        for tool, config_files_list in config_files.items():
            for config_file in config_files_list:
                if (self.repo_path / config_file).exists():
                    if tool not in tools:
                        tools.append(tool)
                    break
        
        return tools
    
    def _check_pre_commit(self) -> Dict[str, Any]:
        """Check pre-commit configuration."""
        pre_commit_config = self.repo_path / ".pre-commit-config.yaml"
        if pre_commit_config.exists():
            try:
                content = pre_commit_config.read_text()
                import re
                hooks = re.findall(r'- id: ([^\s]+)', content)
                return {"enabled": True, "hooks": hooks}
            except Exception:
                return {"enabled": True, "hooks": []}
        return {"enabled": False}
    
    def _detect_security_tools(self) -> list:
        """Detect security tools in use."""
        tools = []
        
        # Check for security tool configs
        if (self.repo_path / ".bandit").exists() or "bandit" in (self.repo_path / "pyproject.toml").read_text() if (self.repo_path / "pyproject.toml").exists() else "":
            tools.append("bandit")
        
        if (self.repo_path / ".safety-policy.json").exists():
            tools.append("safety")
        
        # Check for GitHub security features
        github_dir = self.repo_path / ".github"
        if github_dir.exists():
            if (github_dir / "dependabot.yml").exists():
                tools.append("dependabot")
            
            workflows_dir = github_dir / "workflows"
            if workflows_dir.exists():
                for workflow in workflows_dir.glob("*.yml"):
                    content = workflow.read_text()
                    if "codeql" in content.lower():
                        tools.append("codeql")
                    if "snyk" in content.lower():
                        tools.append("snyk")
        
        return tools
    
    def _calculate_doc_coverage(self) -> float:
        """Calculate documentation coverage."""
        try:
            # Count Python functions/classes vs docstrings
            python_files = list(self.repo_path.rglob("*.py"))
            
            total_items = 0
            documented_items = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    
                    # Count functions and classes
                    import re
                    functions = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
                    classes = re.findall(r'class\s+\w+[^:]*:', content)
                    
                    total_items += len(functions) + len(classes)
                    
                    # Count docstrings (simplified)
                    docstrings = re.findall(r'"""[^"]*"""', content) + re.findall(r"'''[^']*'''", content)
                    documented_items += min(len(docstrings), len(functions) + len(classes))
                    
                except Exception:
                    continue
            
            return documented_items / total_items if total_items > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def save_metrics(self, output_file: str = ".github/project-metrics.json"):
        """Save collected metrics to file."""
        output_path = self.repo_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Metrics saved to {output_path}")
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        if not self.metrics:
            return "No metrics collected yet."
        
        report = []
        report.append("# FastVLM On-Device Kit - Metrics Report")
        report.append(f"Generated: {self.metrics.get('collection_timestamp', 'Unknown')}")
        report.append("")
        
        # Code metrics
        if "code_metrics" in self.metrics:
            code = self.metrics["code_metrics"]
            report.append("## Code Metrics")
            if "lines_of_code" in code:
                loc = code["lines_of_code"]
                report.append(f"- Total lines of code: {loc.get('total', 0):,}")
                report.append(f"- Python: {loc.get('python', 0):,}")
                report.append(f"- Swift: {loc.get('swift', 0):,}")
                report.append(f"- Documentation: {loc.get('markdown', 0):,}")
            report.append("")
        
        # Git metrics
        if "git_metrics" in self.metrics:
            git = self.metrics["git_metrics"]
            report.append("## Development Activity")
            report.append(f"- Commits (last 30 days): {git.get('commits_last_30_days', 0)}")
            report.append(f"- Active contributors: {git.get('active_contributors_last_30_days', 0)}")
            report.append(f"- Total branches: {git.get('total_branches', 0)}")
            report.append("")
        
        # Test metrics
        if "test_metrics" in self.metrics:
            test = self.metrics["test_metrics"]
            report.append("## Testing")
            report.append(f"- Test files: {test.get('test_files', 0)}")
            report.append(f"- Total tests: {test.get('total_tests', 0)}")
            if "coverage" in test and isinstance(test["coverage"], dict):
                coverage = test["coverage"]
                if "totals" in coverage:
                    report.append(f"- Coverage: {coverage['totals'].get('percent_covered', 0):.1f}%")
            report.append("")
        
        # Quality metrics
        if "quality_metrics" in self.metrics:
            quality = self.metrics["quality_metrics"]
            report.append("## Code Quality")
            tools = quality.get("linting_tools", [])
            report.append(f"- Quality tools: {', '.join(tools) if tools else 'None'}")
            
            pre_commit = quality.get("pre_commit_hooks", {})
            if pre_commit.get("enabled"):
                report.append(f"- Pre-commit hooks: {len(pre_commit.get('hooks', []))}")
            
            doc_coverage = quality.get("documentation_coverage", 0)
            report.append(f"- Documentation coverage: {doc_coverage:.1%}")
            report.append("")
        
        # Automation metrics
        if "automation_metrics" in self.metrics:
            auto = self.metrics["automation_metrics"]
            report.append("## Automation")
            report.append(f"- CI/CD workflows: {auto.get('ci_cd_workflows', 0)}")
            report.append(f"- Automation scripts: {auto.get('automation_scripts', 0)}")
            report.append(f"- Docker support: {'Yes' if auto.get('docker_files', 0) > 0 else 'No'}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect FastVLM project metrics")
    parser.add_argument("--output", "-o", default=".github/project-metrics.json",
                       help="Output file for metrics JSON")
    parser.add_argument("--report", "-r", action="store_true",
                       help="Generate human-readable report")
    parser.add_argument("--print-report", action="store_true",
                       help="Print report to stdout")
    
    args = parser.parse_args()
    
    # Collect metrics
    collector = MetricsCollector()
    metrics = collector.collect_all_metrics()
    
    # Save metrics
    collector.save_metrics(args.output)
    
    # Generate report if requested
    if args.report or args.print_report:
        report = collector.generate_report()
        
        if args.print_report:
            print("\n" + report)
        
        if args.report:
            report_path = Path(args.output).parent / "metrics-report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"📊 Report saved to {report_path}")
    
    print("\n✨ Metrics collection completed!")


if __name__ == "__main__":
    main()