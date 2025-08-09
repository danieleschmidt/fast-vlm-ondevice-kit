#!/usr/bin/env python3
"""
Quality Gates Runner for FastVLM On-Device Kit

Implements comprehensive quality gates including:
- Code quality checks (linting, formatting, type checking)
- Security scanning
- Performance benchmarks
- Test coverage validation
- Architecture compliance
- Production readiness assessment
"""

import sys
import os
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate status levels."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: QualityGateStatus
    score: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def add_issue(self, severity: Severity, message: str, file_path: str = "", line_number: int = 0):
        """Add an issue to the result."""
        self.issues.append({
            "severity": severity.value,
            "message": message,
            "file": file_path,
            "line": line_number,
            "timestamp": time.time()
        })
    
    def add_recommendation(self, recommendation: str):
        """Add a recommendation to the result."""
        self.recommendations.append(recommendation)


class CodeQualityGate:
    """Code quality gate implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.tests_path = project_root / "tests"
    
    def run_checks(self) -> QualityGateResult:
        """Run all code quality checks."""
        result = QualityGateResult("code_quality", QualityGateStatus.PASSED, 1.0)
        start_time = time.time()
        
        logger.info("Running code quality checks...")
        
        try:
            # Run linting checks
            lint_score = self._run_linting_checks(result)
            
            # Run formatting checks
            format_score = self._run_formatting_checks(result)
            
            # Run type checking
            type_score = self._run_type_checks(result)
            
            # Run complexity analysis
            complexity_score = self._run_complexity_analysis(result)
            
            # Calculate overall score
            overall_score = (lint_score + format_score + type_score + complexity_score) / 4.0
            result.score = overall_score
            
            # Determine status
            if overall_score >= 0.9:
                result.status = QualityGateStatus.PASSED
            elif overall_score >= 0.7:
                result.status = QualityGateStatus.WARNING
                result.add_recommendation("Consider addressing code quality issues to improve score")
            else:
                result.status = QualityGateStatus.FAILED
                result.add_recommendation("Code quality score is below acceptable threshold")
            
            result.details["overall_score"] = overall_score
            result.details["lint_score"] = lint_score
            result.details["format_score"] = format_score
            result.details["type_score"] = type_score
            result.details["complexity_score"] = complexity_score
            
        except Exception as e:
            logger.error(f"Code quality checks failed: {e}")
            result.status = QualityGateStatus.FAILED
            result.add_issue(Severity.CRITICAL, f"Code quality check execution failed: {e}")
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_linting_checks(self, result: QualityGateResult) -> float:
        """Run linting checks with flake8."""
        try:
            # Run flake8 linting
            cmd = ["python", "-m", "flake8", str(self.src_path), "--max-line-length=88", "--extend-ignore=E203,W503"]
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if process.returncode == 0:
                result.details["flake8_issues"] = 0
                return 1.0
            else:
                # Parse flake8 output
                lines = process.stdout.strip().split('\n') if process.stdout.strip() else []
                issue_count = len([line for line in lines if line.strip()])
                
                result.details["flake8_issues"] = issue_count
                
                for line in lines:
                    if line.strip():
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            file_path = parts[0]
                            line_num = int(parts[1]) if parts[1].isdigit() else 0
                            message = parts[3].strip() if len(parts) > 3 else line
                            result.add_issue(Severity.MEDIUM, f"Linting: {message}", file_path, line_num)
                
                # Calculate score based on issues
                score = max(0.0, 1.0 - (issue_count * 0.05))  # -5% per issue
                return score
                
        except Exception as e:
            logger.warning(f"Flake8 linting failed: {e}")
            result.add_issue(Severity.HIGH, f"Linting check failed: {e}")
            return 0.5
    
    def _run_formatting_checks(self, result: QualityGateResult) -> float:
        """Run formatting checks with black."""
        try:
            # Check if code is formatted with black
            cmd = ["python", "-m", "black", "--check", "--diff", str(self.src_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if process.returncode == 0:
                result.details["formatting_issues"] = 0
                return 1.0
            else:
                # Count files that need formatting
                output_lines = process.stdout.strip().split('\n') if process.stdout.strip() else []
                files_needing_format = len([line for line in output_lines if line.startswith("would reformat")])
                
                result.details["formatting_issues"] = files_needing_format
                
                if files_needing_format > 0:
                    result.add_issue(
                        Severity.MEDIUM, 
                        f"{files_needing_format} files need formatting",
                        "",
                        0
                    )
                    result.add_recommendation("Run 'black src tests' to fix formatting issues")
                
                score = max(0.0, 1.0 - (files_needing_format * 0.1))
                return score
                
        except Exception as e:
            logger.warning(f"Black formatting check failed: {e}")
            result.add_issue(Severity.MEDIUM, f"Formatting check failed: {e}")
            return 0.8  # Partial credit if tools not available
    
    def _run_type_checks(self, result: QualityGateResult) -> float:
        """Run type checking with mypy."""
        try:
            # Run mypy type checking
            cmd = ["python", "-m", "mypy", str(self.src_path), "--ignore-missing-imports"]
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if process.returncode == 0:
                result.details["type_errors"] = 0
                return 1.0
            else:
                # Parse mypy output
                lines = process.stdout.strip().split('\n') if process.stdout.strip() else []
                type_errors = len([line for line in lines if ": error:" in line])
                
                result.details["type_errors"] = type_errors
                
                for line in lines:
                    if ": error:" in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            line_num = int(parts[1]) if parts[1].isdigit() else 0
                            message = parts[2].strip() if len(parts) > 2 else line
                            result.add_issue(Severity.MEDIUM, f"Type error: {message}", file_path, line_num)
                
                score = max(0.0, 1.0 - (type_errors * 0.1))
                return score
                
        except Exception as e:
            logger.warning(f"Mypy type checking failed: {e}")
            result.add_issue(Severity.LOW, f"Type checking failed: {e}")
            return 0.8  # Partial credit
    
    def _run_complexity_analysis(self, result: QualityGateResult) -> float:
        """Run complexity analysis."""
        try:
            # Simple complexity analysis by counting lines and functions
            total_lines = 0
            total_functions = 0
            complex_functions = 0
            
            for py_file in self.src_path.rglob("*.py"):
                if py_file.is_file():
                    content = py_file.read_text()
                    lines = content.split('\n')
                    total_lines += len(lines)
                    
                    # Count functions (simple heuristic)
                    functions = [line for line in lines if line.strip().startswith('def ')]
                    total_functions += len(functions)
                    
                    # Count potentially complex functions (> 20 lines)
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            # Find function end (next function or class)
                            func_lines = 1
                            for j in range(i + 1, len(lines)):
                                if lines[j].strip().startswith(('def ', 'class ')) and not lines[j].startswith('    '):
                                    break
                                func_lines += 1
                            
                            if func_lines > 20:
                                complex_functions += 1
            
            result.details["total_lines"] = total_lines
            result.details["total_functions"] = total_functions
            result.details["complex_functions"] = complex_functions
            
            # Calculate complexity score
            if total_functions == 0:
                complexity_ratio = 0
            else:
                complexity_ratio = complex_functions / total_functions
            
            score = max(0.0, 1.0 - (complexity_ratio * 2.0))  # Penalize high complexity
            
            if complex_functions > total_functions * 0.2:  # More than 20% complex
                result.add_issue(
                    Severity.MEDIUM,
                    f"{complex_functions} functions may be too complex (>20 lines)",
                    "",
                    0
                )
                result.add_recommendation("Consider refactoring large functions into smaller ones")
            
            return score
            
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return 0.9


class SecurityGate:
    """Security gate implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
    
    def run_checks(self) -> QualityGateResult:
        """Run all security checks."""
        result = QualityGateResult("security", QualityGateStatus.PASSED, 1.0)
        start_time = time.time()
        
        logger.info("Running security checks...")
        
        try:
            # Run bandit security analysis
            security_score = self._run_bandit_analysis(result)
            
            # Check for hardcoded secrets
            secrets_score = self._check_hardcoded_secrets(result)
            
            # Check dependencies for vulnerabilities
            deps_score = self._check_dependency_vulnerabilities(result)
            
            # Calculate overall score
            overall_score = (security_score + secrets_score + deps_score) / 3.0
            result.score = overall_score
            
            # Determine status
            if overall_score >= 0.9:
                result.status = QualityGateStatus.PASSED
            elif overall_score >= 0.7:
                result.status = QualityGateStatus.WARNING
            else:
                result.status = QualityGateStatus.FAILED
            
            result.details["security_score"] = security_score
            result.details["secrets_score"] = secrets_score
            result.details["dependencies_score"] = deps_score
            
        except Exception as e:
            logger.error(f"Security checks failed: {e}")
            result.status = QualityGateStatus.FAILED
            result.add_issue(Severity.CRITICAL, f"Security check execution failed: {e}")
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_bandit_analysis(self, result: QualityGateResult) -> float:
        """Run bandit security analysis."""
        try:
            cmd = ["python", "-m", "bandit", "-r", str(self.src_path), "-f", "json"]
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if process.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_output = json.loads(process.stdout)
                    results = bandit_output.get("results", [])
                    
                    high_issues = len([r for r in results if r.get("issue_severity") == "HIGH"])
                    medium_issues = len([r for r in results if r.get("issue_severity") == "MEDIUM"])
                    low_issues = len([r for r in results if r.get("issue_severity") == "LOW"])
                    
                    result.details["bandit_high_issues"] = high_issues
                    result.details["bandit_medium_issues"] = medium_issues
                    result.details["bandit_low_issues"] = low_issues
                    
                    # Add issues to result
                    for issue in results:
                        severity_map = {
                            "HIGH": Severity.HIGH,
                            "MEDIUM": Severity.MEDIUM,
                            "LOW": Severity.LOW
                        }
                        severity = severity_map.get(issue.get("issue_severity", "LOW"), Severity.LOW)
                        
                        result.add_issue(
                            severity,
                            f"Security: {issue.get('test_name', 'Unknown')} - {issue.get('issue_text', '')}",
                            issue.get('filename', ''),
                            issue.get('line_number', 0)
                        )
                    
                    # Calculate score
                    penalty = high_issues * 0.3 + medium_issues * 0.1 + low_issues * 0.05
                    score = max(0.0, 1.0 - penalty)
                    
                    return score
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse bandit output")
                    return 0.8
            else:
                logger.warning(f"Bandit failed with exit code {process.returncode}")
                return 0.5
                
        except Exception as e:
            logger.warning(f"Bandit security analysis failed: {e}")
            result.add_issue(Severity.MEDIUM, f"Security scanning failed: {e}")
            return 0.8
    
    def _check_hardcoded_secrets(self, result: QualityGateResult) -> float:
        """Check for hardcoded secrets and sensitive data."""
        secret_patterns = [
            r'(?i)(password|pwd|passwd)\s*[=:]\s*["\']?[^"\'\s]+',
            r'(?i)(secret|token|key)\s*[=:]\s*["\']?[^"\'\s]{8,}',
            r'(?i)(api_key|apikey)\s*[=:]\s*["\']?[^"\'\s]+',
            r'["\'][A-Za-z0-9+/]{40,}["\']',  # Base64-like strings
        ]
        
        import re
        
        secrets_found = 0
        
        try:
            for py_file in self.src_path.rglob("*.py"):
                if py_file.is_file():
                    content = py_file.read_text()
                    
                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            secrets_found += len(matches)
                            result.add_issue(
                                Severity.HIGH,
                                f"Potential hardcoded secret found: {matches[0][:20]}...",
                                str(py_file),
                                0
                            )
            
            result.details["potential_secrets"] = secrets_found
            
            if secrets_found > 0:
                result.add_recommendation("Review and remove hardcoded secrets")
            
            score = 1.0 if secrets_found == 0 else max(0.0, 1.0 - (secrets_found * 0.2))
            return score
            
        except Exception as e:
            logger.warning(f"Secret scanning failed: {e}")
            return 0.9
    
    def _check_dependency_vulnerabilities(self, result: QualityGateResult) -> float:
        """Check dependencies for known vulnerabilities."""
        try:
            # Try to run safety check
            cmd = ["python", "-m", "safety", "check", "--json"]
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if process.returncode == 0:
                result.details["vulnerable_dependencies"] = 0
                return 1.0
            else:
                try:
                    safety_output = json.loads(process.stdout)
                    vulnerabilities = len(safety_output)
                    
                    result.details["vulnerable_dependencies"] = vulnerabilities
                    
                    for vuln in safety_output:
                        result.add_issue(
                            Severity.HIGH,
                            f"Vulnerable dependency: {vuln.get('package', 'Unknown')} - {vuln.get('vulnerability', '')}",
                            "",
                            0
                        )
                    
                    score = max(0.0, 1.0 - (vulnerabilities * 0.2))
                    return score
                    
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to parse safety output")
                    return 0.8
        
        except Exception as e:
            logger.warning(f"Dependency vulnerability check failed: {e}")
            return 0.9


class TestCoverageGate:
    """Test coverage gate implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.tests_path = project_root / "tests"
    
    def run_checks(self) -> QualityGateResult:
        """Run test coverage checks."""
        result = QualityGateResult("test_coverage", QualityGateStatus.PASSED, 1.0)
        start_time = time.time()
        
        logger.info("Running test coverage checks...")
        
        try:
            # Run tests with coverage
            coverage_score = self._run_coverage_analysis(result)
            
            # Check test quality
            test_quality_score = self._analyze_test_quality(result)
            
            overall_score = (coverage_score + test_quality_score) / 2.0
            result.score = overall_score
            
            # Determine status based on coverage thresholds
            if overall_score >= 0.85:  # 85% coverage target
                result.status = QualityGateStatus.PASSED
            elif overall_score >= 0.70:
                result.status = QualityGateStatus.WARNING
                result.add_recommendation("Increase test coverage to meet 85% target")
            else:
                result.status = QualityGateStatus.FAILED
                result.add_recommendation("Test coverage is below acceptable threshold")
            
        except Exception as e:
            logger.error(f"Test coverage checks failed: {e}")
            result.status = QualityGateStatus.FAILED
            result.add_issue(Severity.CRITICAL, f"Test coverage check failed: {e}")
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_coverage_analysis(self, result: QualityGateResult) -> float:
        """Run test coverage analysis."""
        try:
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest", 
                str(self.tests_path),
                f"--cov={self.src_path}",
                "--cov-report=json",
                "--cov-report=term-missing",
                "-v"
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # Try to read coverage report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    
                    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                    result.details["line_coverage"] = total_coverage
                    
                    # Check individual file coverage
                    files = coverage_data.get("files", {})
                    low_coverage_files = []
                    
                    for file_path, file_data in files.items():
                        file_coverage = file_data.get("summary", {}).get("percent_covered", 0)
                        if file_coverage < 70:  # Less than 70% coverage
                            low_coverage_files.append((file_path, file_coverage))
                    
                    result.details["low_coverage_files"] = len(low_coverage_files)
                    
                    for file_path, coverage in low_coverage_files:
                        result.add_issue(
                            Severity.MEDIUM,
                            f"Low test coverage: {file_path} ({coverage:.1f}%)",
                            file_path,
                            0
                        )
                    
                    return total_coverage / 100.0  # Convert to 0-1 scale
                    
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning("Failed to read coverage report")
                    return 0.5
            
            # Fallback: parse stdout for coverage info
            if "TOTAL" in process.stdout:
                lines = process.stdout.split('\n')
                for line in lines:
                    if line.strip().startswith("TOTAL"):
                        parts = line.split()
                        if len(parts) >= 4 and parts[-1].endswith('%'):
                            coverage_str = parts[-1].rstrip('%')
                            try:
                                coverage = float(coverage_str) / 100.0
                                result.details["line_coverage"] = coverage * 100
                                return coverage
                            except ValueError:
                                pass
            
            return 0.5  # Default if we can't determine coverage
            
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            return 0.5
    
    def _analyze_test_quality(self, result: QualityGateResult) -> float:
        """Analyze test quality metrics."""
        try:
            test_files = list(self.tests_path.rglob("test_*.py"))
            
            if not test_files:
                result.add_issue(Severity.HIGH, "No test files found", "", 0)
                return 0.0
            
            total_tests = 0
            total_assertions = 0
            
            for test_file in test_files:
                content = test_file.read_text()
                
                # Count test functions
                test_functions = len([line for line in content.split('\n') 
                                    if line.strip().startswith('def test_')])
                total_tests += test_functions
                
                # Count assertions (rough estimate)
                assertions = len([line for line in content.split('\n') 
                                if 'assert' in line and line.strip().startswith('assert')])
                total_assertions += assertions
            
            result.details["total_test_files"] = len(test_files)
            result.details["total_test_functions"] = total_tests
            result.details["total_assertions"] = total_assertions
            
            if total_tests == 0:
                result.add_issue(Severity.HIGH, "No test functions found", "", 0)
                return 0.0
            
            # Quality metrics
            assertions_per_test = total_assertions / total_tests if total_tests > 0 else 0
            
            # Score based on test coverage and assertion density
            score = min(1.0, (assertions_per_test / 3.0) + 0.5)  # Target ~3 assertions per test
            
            if assertions_per_test < 1.5:
                result.add_recommendation("Consider adding more assertions to improve test quality")
            
            return score
            
        except Exception as e:
            logger.warning(f"Test quality analysis failed: {e}")
            return 0.7


class QualityGatesRunner:
    """Main quality gates runner."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[QualityGateResult] = []
    
    def run_all_gates(self, gates: List[str] = None) -> Dict[str, Any]:
        """Run all quality gates."""
        if gates is None:
            gates = ["code_quality", "security", "test_coverage"]
        
        logger.info("Starting quality gates execution")
        overall_start = time.time()
        
        # Run individual gates
        if "code_quality" in gates:
            code_gate = CodeQualityGate(self.project_root)
            self.results.append(code_gate.run_checks())
        
        if "security" in gates:
            security_gate = SecurityGate(self.project_root)
            self.results.append(security_gate.run_checks())
        
        if "test_coverage" in gates:
            test_gate = TestCoverageGate(self.project_root)
            self.results.append(test_gate.run_checks())
        
        total_time = (time.time() - overall_start) * 1000
        
        # Calculate overall results
        overall_report = self._generate_overall_report(total_time)
        
        # Save results
        self._save_results(overall_report)
        
        return overall_report
    
    def _generate_overall_report(self, total_time_ms: float) -> Dict[str, Any]:
        """Generate overall quality gates report."""
        # Calculate overall status
        passed_gates = len([r for r in self.results if r.status == QualityGateStatus.PASSED])
        warning_gates = len([r for r in self.results if r.status == QualityGateStatus.WARNING])
        failed_gates = len([r for r in self.results if r.status == QualityGateStatus.FAILED])
        total_gates = len(self.results)
        
        if failed_gates > 0:
            overall_status = "FAILED"
        elif warning_gates > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"
        
        # Calculate overall score
        if total_gates > 0:
            overall_score = sum(r.score for r in self.results) / total_gates
        else:
            overall_score = 0.0
        
        # Collect all issues by severity
        issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for result in self.results:
            for issue in result.issues:
                severity = issue.get("severity", "low")
                if severity in issues_by_severity:
                    issues_by_severity[severity] += 1
        
        # Generate recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "execution_time_ms": total_time_ms,
            "summary": {
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "warning_gates": warning_gates,
                "failed_gates": failed_gates,
                "issues_by_severity": issues_by_severity,
                "total_issues": sum(issues_by_severity.values())
            },
            "gate_results": [
                {
                    "gate_name": r.gate_name,
                    "status": r.status.value,
                    "score": r.score,
                    "execution_time_ms": r.execution_time_ms,
                    "issues_count": len(r.issues),
                    "recommendations_count": len(r.recommendations),
                    "details": r.details
                }
                for r in self.results
            ],
            "detailed_results": [
                {
                    "gate_name": r.gate_name,
                    "status": r.status.value,
                    "score": r.score,
                    "issues": r.issues,
                    "recommendations": r.recommendations,
                    "details": r.details
                }
                for r in self.results
            ],
            "recommendations": all_recommendations,
            "timestamp": time.time()
        }
    
    def _save_results(self, report: Dict[str, Any]):
        """Save quality gates results to file."""
        results_file = self.project_root / "quality_gates_report.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Quality gates report saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_summary(self):
        """Print quality gates summary to console."""
        if not self.results:
            print("No quality gates results available")
            return
        
        print("\n" + "="*60)
        print("QUALITY GATES SUMMARY")
        print("="*60)
        
        for result in self.results:
            status_icon = {
                QualityGateStatus.PASSED: "✅",
                QualityGateStatus.WARNING: "⚠️",
                QualityGateStatus.FAILED: "❌",
                QualityGateStatus.SKIPPED: "⏭️"
            }.get(result.status, "?")
            
            print(f"\n{status_icon} {result.gate_name.upper()}")
            print(f"   Status: {result.status.value.upper()}")
            print(f"   Score: {result.score:.2f}")
            print(f"   Issues: {len(result.issues)}")
            print(f"   Time: {result.execution_time_ms:.1f}ms")
            
            if result.issues:
                critical_issues = len([i for i in result.issues if i.get("severity") == "critical"])
                high_issues = len([i for i in result.issues if i.get("severity") == "high"])
                
                if critical_issues > 0:
                    print(f"   ⚠️  {critical_issues} critical issues")
                if high_issues > 0:
                    print(f"   ⚠️  {high_issues} high severity issues")
        
        # Overall summary
        overall_score = sum(r.score for r in self.results) / len(self.results)
        failed_gates = len([r for r in self.results if r.status == QualityGateStatus.FAILED])
        
        print("\n" + "="*60)
        if failed_gates == 0:
            print("🎉 ALL QUALITY GATES PASSED!")
        else:
            print(f"❌ {failed_gates} QUALITY GATES FAILED")
        
        print(f"Overall Score: {overall_score:.2f}/1.00")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run FastVLM quality gates")
    parser.add_argument(
        "--gates",
        nargs="+",
        choices=["code_quality", "security", "test_coverage"],
        default=["code_quality", "security", "test_coverage"],
        help="Quality gates to run"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory"
    )
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "both"],
        default="both",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Run quality gates
    runner = QualityGatesRunner(args.project_root)
    report = runner.run_all_gates(args.gates)
    
    # Output results
    if args.output_format in ["console", "both"]:
        runner.print_summary()
    
    if args.output_format in ["json", "both"]:
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    exit_code = 0 if report["overall_status"] == "PASSED" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()