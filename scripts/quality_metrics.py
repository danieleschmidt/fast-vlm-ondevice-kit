#!/usr/bin/env python3
"""
Code Quality Metrics Collection and Analysis

Comprehensive quality metrics system for MATURING SDLC environments.
"""

import ast
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import logging
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Code quality metrics"""
    lines_of_code: int
    cyclomatic_complexity: float
    cognitive_complexity: float
    maintainability_index: float
    test_coverage: float
    code_duplication: float
    technical_debt_hours: float
    security_hotspots: int
    code_smells: int


@dataclass
class QualityGateResult:
    """Quality gate evaluation result"""
    gate_name: str
    status: str  # PASS, WARN, FAIL
    actual_value: float
    threshold_value: float
    message: str


class PythonMetricsCollector:
    """Collect metrics for Python code"""
    
    def __init__(self, source_dir: Path = Path("src")):
        self.source_dir = source_dir
    
    def count_lines_of_code(self) -> int:
        """Count total lines of code"""
        total_lines = 0
        for py_file in self.source_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Count non-empty, non-comment lines
                    code_lines = [
                        line for line in lines 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    total_lines += len(code_lines)
            except Exception as e:
                logger.warning(f"Could not process {py_file}: {e}")
        
        return total_lines
    
    def calculate_cyclomatic_complexity(self) -> float:
        """Calculate average cyclomatic complexity"""
        try:
            result = subprocess.run([
                'radon', 'cc', str(self.source_dir), '-j'
            ], capture_output=True, text=True, check=True)
            
            complexity_data = json.loads(result.stdout)
            total_complexity = 0
            function_count = 0
            
            for file_path, metrics in complexity_data.items():
                for item in metrics:
                    if item['type'] in ['function', 'method']:
                        total_complexity += item['complexity']
                        function_count += 1
            
            return total_complexity / function_count if function_count > 0 else 0.0
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not calculate cyclomatic complexity")
            return 0.0
    
    def calculate_maintainability_index(self) -> float:
        """Calculate maintainability index"""
        try:
            result = subprocess.run([
                'radon', 'mi', str(self.source_dir), '-j'
            ], capture_output=True, text=True, check=True)
            
            mi_data = json.loads(result.stdout)
            total_mi = 0
            file_count = 0
            
            for file_path, mi_value in mi_data.items():
                if isinstance(mi_value, (int, float)):
                    total_mi += mi_value
                    file_count += 1
            
            return total_mi / file_count if file_count > 0 else 0.0
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not calculate maintainability index")
            return 0.0
    
    def get_test_coverage(self) -> float:
        """Get test coverage percentage"""
        try:
            # Run coverage and generate XML report
            subprocess.run(['coverage', 'run', '-m', 'pytest'], check=True)
            subprocess.run(['coverage', 'xml'], check=True)
            
            # Parse coverage XML
            tree = ET.parse('coverage.xml')
            root = tree.getroot()
            
            line_rate = float(root.get('line-rate', 0)) * 100
            return line_rate
            
        except (subprocess.CalledProcessError, FileNotFoundError, ET.ParseError):
            logger.warning("Could not get test coverage")
            return 0.0
    
    def detect_code_duplication(self) -> float:
        """Detect code duplication percentage"""
        try:
            result = subprocess.run([
                'jscpd', str(self.source_dir), '--reporters', 'json'
            ], capture_output=True, text=True, check=True)
            
            duplication_data = json.loads(result.stdout)
            total_lines = duplication_data.get('statistics', {}).get('total', {}).get('lines', 1)
            duplicated_lines = duplication_data.get('statistics', {}).get('clones', {}).get('duplicatedLines', 0)
            
            return (duplicated_lines / total_lines) * 100 if total_lines > 0 else 0.0
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not detect code duplication")
            return 0.0
    
    def run_security_analysis(self) -> Tuple[int, int]:
        """Run security analysis and return hotspots and issues count"""
        try:
            result = subprocess.run([
                'bandit', '-r', str(self.source_dir), '-f', 'json'
            ], capture_output=True, text=True)
            
            # Bandit returns non-zero exit code when issues found
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                results = bandit_data.get('results', [])
                
                high_severity = len([r for r in results if r.get('issue_severity') == 'HIGH'])
                medium_severity = len([r for r in results if r.get('issue_severity') == 'MEDIUM'])
                
                return high_severity, medium_severity
            
            return 0, 0
            
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not run security analysis")
            return 0, 0


class SwiftMetricsCollector:
    """Collect metrics for Swift code"""
    
    def __init__(self, source_dir: Path = Path("ios")):
        self.source_dir = source_dir
    
    def count_lines_of_code(self) -> int:
        """Count Swift lines of code"""
        total_lines = 0
        for swift_file in self.source_dir.rglob("*.swift"):
            try:
                with open(swift_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    code_lines = [
                        line for line in lines 
                        if line.strip() and not line.strip().startswith('//')
                    ]
                    total_lines += len(code_lines)
            except Exception as e:
                logger.warning(f"Could not process {swift_file}: {e}")
        
        return total_lines
    
    def run_swiftlint(self) -> Tuple[int, int]:
        """Run SwiftLint and return warning and error counts"""
        try:
            result = subprocess.run([
                'swiftlint', 'lint', '--reporter', 'json', str(self.source_dir)
            ], capture_output=True, text=True)
            
            if result.stdout:
                lint_results = json.loads(result.stdout)
                warnings = len([r for r in lint_results if r.get('severity') == 'warning'])
                errors = len([r for r in lint_results if r.get('severity') == 'error'])
                return warnings, errors
            
            return 0, 0
            
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not run SwiftLint")
            return 0, 0


class QualityGateEvaluator:
    """Evaluate code quality against defined gates"""
    
    def __init__(self, config_path: Path = Path(".github/code-quality-config.yml")):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load quality gate configuration"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, ImportError):
            # Default configuration
            return {
                'quality_gates': {
                    'coverage': {'minimum_threshold': 80},
                    'complexity': {'cyclomatic_complexity': 10},
                    'security': {'max_high_severity_issues': 0}
                }
            }
    
    def evaluate_coverage(self, coverage: float) -> QualityGateResult:
        """Evaluate test coverage gate"""
        threshold = self.config['quality_gates']['coverage']['minimum_threshold']
        
        if coverage >= threshold:
            status = "PASS"
            message = f"Coverage {coverage:.1f}% meets threshold"
        elif coverage >= threshold * 0.9:
            status = "WARN"
            message = f"Coverage {coverage:.1f}% below threshold but acceptable"
        else:
            status = "FAIL"
            message = f"Coverage {coverage:.1f}% significantly below threshold"
        
        return QualityGateResult(
            gate_name="test_coverage",
            status=status,
            actual_value=coverage,
            threshold_value=threshold,
            message=message
        )
    
    def evaluate_complexity(self, complexity: float) -> QualityGateResult:
        """Evaluate cyclomatic complexity gate"""
        threshold = self.config['quality_gates']['complexity']['cyclomatic_complexity']
        
        if complexity <= threshold:
            status = "PASS"
            message = f"Complexity {complexity:.1f} within acceptable range"
        elif complexity <= threshold * 1.2:
            status = "WARN"
            message = f"Complexity {complexity:.1f} slightly above threshold"
        else:
            status = "FAIL"
            message = f"Complexity {complexity:.1f} too high"
        
        return QualityGateResult(
            gate_name="cyclomatic_complexity",
            status=status,
            actual_value=complexity,
            threshold_value=threshold,
            message=message
        )
    
    def evaluate_security(self, high_issues: int, medium_issues: int) -> List[QualityGateResult]:
        """Evaluate security gates"""
        results = []
        
        high_threshold = self.config['quality_gates']['security']['max_high_severity_issues']
        medium_threshold = self.config['quality_gates']['security'].get('max_medium_severity_issues', 5)
        
        # High severity issues
        high_status = "PASS" if high_issues <= high_threshold else "FAIL"
        results.append(QualityGateResult(
            gate_name="high_severity_security",
            status=high_status,
            actual_value=high_issues,
            threshold_value=high_threshold,
            message=f"{high_issues} high severity security issues"
        ))
        
        # Medium severity issues
        medium_status = "PASS" if medium_issues <= medium_threshold else "WARN"
        results.append(QualityGateResult(
            gate_name="medium_severity_security",
            status=medium_status,
            actual_value=medium_issues,
            threshold_value=medium_threshold,
            message=f"{medium_issues} medium severity security issues"
        ))
        
        return results


class QualityMetricsOrchestrator:
    """Main orchestrator for quality metrics collection"""
    
    def __init__(self):
        self.python_collector = PythonMetricsCollector()
        self.swift_collector = SwiftMetricsCollector()
        self.gate_evaluator = QualityGateEvaluator()
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all quality metrics"""
        logger.info("Collecting code quality metrics...")
        
        # Python metrics
        python_loc = self.python_collector.count_lines_of_code()
        cyclomatic_complexity = self.python_collector.calculate_cyclomatic_complexity()
        maintainability_index = self.python_collector.calculate_maintainability_index()
        test_coverage = self.python_collector.get_test_coverage()
        code_duplication = self.python_collector.detect_code_duplication()
        high_security, medium_security = self.python_collector.run_security_analysis()
        
        # Swift metrics
        swift_loc = self.swift_collector.count_lines_of_code()
        swift_warnings, swift_errors = self.swift_collector.run_swiftlint()
        
        # Overall metrics
        total_loc = python_loc + swift_loc
        
        metrics = {
            'timestamp': str(Path().cwd()),
            'overall': {
                'total_lines_of_code': total_loc,
                'python_lines_of_code': python_loc,
                'swift_lines_of_code': swift_loc
            },
            'python': {
                'cyclomatic_complexity': cyclomatic_complexity,
                'maintainability_index': maintainability_index,
                'test_coverage': test_coverage,
                'code_duplication': code_duplication,
                'security_issues': {
                    'high': high_security,
                    'medium': medium_security
                }
            },
            'swift': {
                'lint_warnings': swift_warnings,
                'lint_errors': swift_errors
            }
        }
        
        return metrics
    
    def evaluate_quality_gates(self, metrics: Dict[str, Any]) -> List[QualityGateResult]:
        """Evaluate all quality gates"""
        results = []
        
        # Coverage gate
        coverage = metrics['python']['test_coverage']
        results.append(self.gate_evaluator.evaluate_coverage(coverage))
        
        # Complexity gate
        complexity = metrics['python']['cyclomatic_complexity']
        results.append(self.gate_evaluator.evaluate_complexity(complexity))
        
        # Security gates
        high_issues = metrics['python']['security_issues']['high']
        medium_issues = metrics['python']['security_issues']['medium']
        results.extend(self.gate_evaluator.evaluate_security(high_issues, medium_issues))
        
        return results
    
    def generate_quality_report(self, metrics: Dict[str, Any], 
                              gate_results: List[QualityGateResult]) -> str:
        """Generate quality report"""
        report_lines = [
            "# Code Quality Report\n",
            f"**Total Lines of Code**: {metrics['overall']['total_lines_of_code']:,}",
            f"**Test Coverage**: {metrics['python']['test_coverage']:.1f}%",
            f"**Cyclomatic Complexity**: {metrics['python']['cyclomatic_complexity']:.1f}",
            f"**Maintainability Index**: {metrics['python']['maintainability_index']:.1f}",
            f"**Code Duplication**: {metrics['python']['code_duplication']:.1f}%",
            "",
            "## Quality Gates\n"
        ]
        
        passed = sum(1 for r in gate_results if r.status == "PASS")
        failed = sum(1 for r in gate_results if r.status == "FAIL")
        warned = sum(1 for r in gate_results if r.status == "WARN")
        
        report_lines.append(f"**Passed**: {passed} | **Failed**: {failed} | **Warnings**: {warned}\n")
        
        for result in gate_results:
            status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[result.status]
            report_lines.append(f"{status_emoji} **{result.gate_name}**: {result.message}")
        
        return "\n".join(report_lines)
    
    def save_metrics(self, metrics: Dict[str, Any], gate_results: List[QualityGateResult]):
        """Save metrics to files"""
        output_dir = Path("quality-reports")
        output_dir.mkdir(exist_ok=True)
        
        # Save raw metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save gate results
        gate_data = [asdict(result) for result in gate_results]
        with open(output_dir / "quality-gates.json", 'w') as f:
            json.dump(gate_data, f, indent=2)
        
        # Save quality report
        report = self.generate_quality_report(metrics, gate_results)
        with open(output_dir / "quality-report.md", 'w') as f:
            f.write(report)
        
        logger.info(f"Quality reports saved to {output_dir}")
    
    def run_quality_analysis(self) -> bool:
        """Run complete quality analysis"""
        try:
            # Collect metrics
            metrics = self.collect_all_metrics()
            
            # Evaluate quality gates
            gate_results = self.evaluate_quality_gates(metrics)
            
            # Save results
            self.save_metrics(metrics, gate_results)
            
            # Determine overall status
            failed_gates = [r for r in gate_results if r.status == "FAIL"]
            
            if failed_gates:
                logger.error(f"Quality gates failed: {len(failed_gates)} failures")
                for result in failed_gates:
                    logger.error(f"  - {result.gate_name}: {result.message}")
                return False
            else:
                logger.info("All quality gates passed")
                return True
                
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return False


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code quality metrics analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quality-reports"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--fail-on-gate-failure",
        action="store_true",
        help="Exit with error code if quality gates fail"
    )
    
    args = parser.parse_args()
    
    orchestrator = QualityMetricsOrchestrator()
    success = orchestrator.run_quality_analysis()
    
    if args.fail_on_gate_failure and not success:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()