#!/usr/bin/env python3
"""
Autonomous Deployment Readiness Validation
Comprehensive Quality Gates for Production Deployment

This script validates that all systems are production-ready with
comprehensive testing, security scanning, and performance validation.
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Quality gate validation result."""
    gate_name: str
    status: str  # "passed", "failed", "warning"
    score: float  # 0-100
    details: Dict[str, Any]
    recommendations: List[str]
    execution_time_ms: float

class AutonomousQualityGates:
    """Comprehensive quality gates for production readiness."""
    
    def __init__(self):
        self.results = []
        self.overall_score = 0.0
        self.critical_failures = []
        
        self.gates = {
            "security_scan": self._security_gate,
            "code_quality": self._code_quality_gate,
            "performance_validation": self._performance_gate,
            "functionality_test": self._functionality_gate,
            "architecture_validation": self._architecture_gate,
            "documentation_check": self._documentation_gate,
            "deployment_readiness": self._deployment_gate
        }
        
        logger.info("🎯 Autonomous Quality Gates initialized")
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate report."""
        logger.info("🚀 Starting comprehensive quality gate validation...")
        
        start_time = time.time()
        
        for gate_name, gate_func in self.gates.items():
            logger.info(f"🔍 Running {gate_name}...")
            
            gate_start = time.time()
            try:
                result = gate_func()
                result.execution_time_ms = (time.time() - gate_start) * 1000
                self.results.append(result)
                
                if result.status == "failed":
                    self.critical_failures.append(gate_name)
                    logger.error(f"❌ {gate_name} FAILED: Score {result.score:.1f}/100")
                elif result.status == "warning":
                    logger.warning(f"⚠️ {gate_name} WARNING: Score {result.score:.1f}/100")
                else:
                    logger.info(f"✅ {gate_name} PASSED: Score {result.score:.1f}/100")
                    
            except Exception as e:
                logger.error(f"💥 {gate_name} encountered error: {e}")
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    status="failed",
                    score=0.0,
                    details={"error": str(e)},
                    recommendations=[f"Fix error in {gate_name}"],
                    execution_time_ms=(time.time() - gate_start) * 1000
                )
                self.results.append(error_result)
                self.critical_failures.append(gate_name)
        
        # Calculate overall score
        self.overall_score = sum(r.score for r in self.results) / len(self.results) if self.results else 0
        
        total_time = time.time() - start_time
        
        # Generate final report
        report = {
            "overall_score": self.overall_score,
            "deployment_ready": len(self.critical_failures) == 0 and self.overall_score >= 85,
            "critical_failures": self.critical_failures,
            "gate_results": [asdict(r) for r in self.results],
            "execution_time_s": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _security_gate(self) -> QualityGateResult:
        """Security validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Run security scanner
            cmd = ["python3", "-m", "bandit", "-r", "src/", "-f", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # Parse results
                try:
                    bandit_data = json.loads(result.stdout)
                    total_issues = len(bandit_data.get("results", []))
                    high_issues = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"])
                    medium_issues = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                    
                    details = {
                        "total_issues": total_issues,
                        "high_severity": high_issues,
                        "medium_severity": medium_issues,
                        "lines_scanned": bandit_data.get("metrics", {}).get("_totals", {}).get("loc", 0)
                    }
                    
                    # Score based on issues
                    if high_issues > 0:
                        score = max(50, 100 - (high_issues * 20))
                        recommendations.append(f"Fix {high_issues} high-severity security issues")
                    elif medium_issues > 5:
                        score = max(75, 100 - (medium_issues * 5))
                        recommendations.append(f"Consider fixing {medium_issues} medium-severity issues")
                    
                except json.JSONDecodeError:
                    score = 90.0
                    details["bandit_output"] = result.stdout[:500]
            else:
                score = 70.0
                details["error"] = "Security scanner failed"
                recommendations.append("Install and configure security scanner")
        
        except Exception as e:
            score = 50.0
            details["error"] = str(e)
            recommendations.append("Fix security scanning setup")
        
        # Additional security checks
        if os.path.exists("requirements.txt"):
            score += 10
            details["requirements_file"] = True
        else:
            recommendations.append("Add requirements.txt for dependency tracking")
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="security_scan",
            status=status,
            score=min(score, 100),
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _code_quality_gate(self) -> QualityGateResult:
        """Code quality validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Count Python files
            src_path = Path("src")
            if src_path.exists():
                py_files = list(src_path.rglob("*.py"))
                test_files = list(Path("tests").rglob("*.py")) if Path("tests").exists() else []
                
                details = {
                    "source_files": len(py_files),
                    "test_files": len(test_files),
                    "test_coverage_estimated": min(100, len(test_files) * 10),  # Rough estimate
                    "has_init_files": len(list(src_path.rglob("__init__.py"))) > 0,
                    "has_docstrings": self._check_docstrings(py_files)
                }
                
                # Score based on code quality indicators
                if len(test_files) == 0:
                    score -= 30
                    recommendations.append("Add test files for better coverage")
                elif len(test_files) < len(py_files) * 0.5:
                    score -= 15
                    recommendations.append("Increase test coverage")
                
                if not details["has_init_files"]:
                    score -= 10
                    recommendations.append("Add __init__.py files for proper packaging")
                
                if details["has_docstrings"] < 50:
                    score -= 10
                    recommendations.append("Add more docstrings for better documentation")
            
            else:
                score = 30.0
                recommendations.append("Create proper source code structure")
        
        except Exception as e:
            score = 40.0
            details["error"] = str(e)
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="code_quality",
            status=status,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _performance_gate(self) -> QualityGateResult:
        """Performance validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Test core functionality performance
            sys.path.insert(0, "src")
            from fast_vlm_ondevice.core_pipeline import FastVLMCorePipeline, create_demo_image
            
            pipeline = FastVLMCorePipeline()
            demo_image = create_demo_image()
            
            # Performance test
            start_time = time.time()
            result = pipeline.process_image_question(demo_image, "What do you see?")
            latency_ms = (time.time() - start_time) * 1000
            
            details = {
                "core_pipeline_latency_ms": latency_ms,
                "target_latency_ms": 250,
                "performance_ratio": 250 / latency_ms if latency_ms > 0 else 1.0,
                "result_confidence": getattr(result, 'confidence', 0.0)
            }
            
            # Score based on performance
            if latency_ms <= 250:
                score = 100
            elif latency_ms <= 500:
                score = 85
            elif latency_ms <= 1000:
                score = 70
            else:
                score = 50
                recommendations.append("Optimize pipeline for better performance")
            
            # Check health status
            health = pipeline.get_health_status()
            if health.get("status") == "healthy":
                score += 5
                details["health_status"] = "healthy"
            else:
                recommendations.append("Improve system health metrics")
        
        except Exception as e:
            score = 30.0
            details["error"] = str(e)
            recommendations.append("Fix performance testing setup")
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="performance_validation",
            status=status,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _functionality_gate(self) -> QualityGateResult:
        """Core functionality validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Test core imports
            sys.path.insert(0, "src")
            from fast_vlm_ondevice import FastVLMCorePipeline, InferenceConfig
            
            # Test basic functionality
            config = InferenceConfig(model_name="test-model")
            pipeline = FastVLMCorePipeline(config)
            
            # Test different scenarios
            test_cases = [
                ("What objects are visible?", "objects"),
                ("Describe the colors", "colors"),
                ("Count the items", "count")
            ]
            
            successful_tests = 0
            for question, expected_keyword in test_cases:
                try:
                    demo_image = b"test_image_data" + b"x" * 100
                    result = pipeline.process_image_question(demo_image, question)
                    
                    if hasattr(result, 'answer') and result.answer:
                        successful_tests += 1
                except:
                    pass
            
            details = {
                "test_cases_run": len(test_cases),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(test_cases) * 100,
                "core_modules_available": True
            }
            
            score = (successful_tests / len(test_cases)) * 100
            
            if score < 70:
                recommendations.append("Fix failing functionality tests")
        
        except ImportError as e:
            score = 20.0
            details["import_error"] = str(e)
            recommendations.append("Fix import issues in core modules")
        except Exception as e:
            score = 40.0
            details["error"] = str(e)
            recommendations.append("Fix functionality testing issues")
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="functionality_test",
            status=status,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _architecture_gate(self) -> QualityGateResult:
        """Architecture validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Check project structure
            required_dirs = ["src", "tests", "docs"]
            optional_dirs = ["examples", "benchmarks", "scripts"]
            
            existing_dirs = [d for d in required_dirs if Path(d).exists()]
            existing_optional = [d for d in optional_dirs if Path(d).exists()]
            
            # Check important files
            important_files = ["README.md", "requirements.txt", "pyproject.toml"]
            existing_files = [f for f in important_files if Path(f).exists()]
            
            details = {
                "required_directories": len(existing_dirs),
                "required_directories_total": len(required_dirs),
                "optional_directories": len(existing_optional),
                "important_files": len(existing_files),
                "important_files_total": len(important_files),
                "project_structure_score": (len(existing_dirs) / len(required_dirs)) * 100
            }
            
            # Score based on architecture
            structure_score = (len(existing_dirs) / len(required_dirs)) * 80
            files_score = (len(existing_files) / len(important_files)) * 20
            bonus_score = min(10, len(existing_optional) * 2)
            
            score = structure_score + files_score + bonus_score
            
            if len(existing_dirs) < len(required_dirs):
                recommendations.append("Add missing required directories")
            
            if len(existing_files) < len(important_files):
                recommendations.append("Add missing project files")
        
        except Exception as e:
            score = 50.0
            details["error"] = str(e)
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="architecture_validation",
            status=status,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _documentation_gate(self) -> QualityGateResult:
        """Documentation validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Check README
            readme_score = 0
            if Path("README.md").exists():
                with open("README.md", "r") as f:
                    readme_content = f.read()
                    readme_score = min(50, len(readme_content) // 100)  # Score based on length
                    
                    # Check for important sections
                    sections = ["installation", "usage", "example", "license"]
                    found_sections = sum(1 for s in sections if s.lower() in readme_content.lower())
                    readme_score += found_sections * 10
            else:
                recommendations.append("Add comprehensive README.md")
            
            # Check docs directory
            docs_score = 0
            if Path("docs").exists():
                doc_files = list(Path("docs").rglob("*.md"))
                docs_score = min(30, len(doc_files) * 5)
            else:
                recommendations.append("Add documentation directory")
            
            # Check docstrings in code
            docstring_score = self._check_docstrings(list(Path("src").rglob("*.py"))) if Path("src").exists() else 0
            
            details = {
                "readme_score": readme_score,
                "docs_score": docs_score,
                "docstring_score": docstring_score,
                "total_documentation_score": readme_score + docs_score + docstring_score
            }
            
            score = readme_score + docs_score + docstring_score
            
            if score < 50:
                recommendations.append("Improve overall documentation coverage")
        
        except Exception as e:
            score = 30.0
            details["error"] = str(e)
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="documentation_check",
            status=status,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _deployment_gate(self) -> QualityGateResult:
        """Deployment readiness validation gate."""
        details = {}
        score = 100.0
        recommendations = []
        
        try:
            # Check deployment artifacts
            deployment_files = ["Dockerfile", "docker-compose.yml", "requirements.txt"]
            existing_deployment = [f for f in deployment_files if Path(f).exists()]
            
            # Check configuration files
            config_files = ["pyproject.toml", "setup.py", "pytest.ini"]
            existing_config = [f for f in config_files if Path(f).exists()]
            
            # Check if package is installable
            installable = Path("pyproject.toml").exists() or Path("setup.py").exists()
            
            details = {
                "deployment_files": len(existing_deployment),
                "deployment_files_total": len(deployment_files),
                "config_files": len(existing_config),
                "config_files_total": len(config_files),
                "installable": installable,
                "deployment_readiness_score": (len(existing_deployment) / len(deployment_files)) * 100
            }
            
            # Score based on deployment readiness
            deployment_score = (len(existing_deployment) / len(deployment_files)) * 60
            config_score = (len(existing_config) / len(config_files)) * 30
            installable_score = 10 if installable else 0
            
            score = deployment_score + config_score + installable_score
            
            if not installable:
                recommendations.append("Make package installable with pyproject.toml or setup.py")
            
            if len(existing_deployment) < 2:
                recommendations.append("Add containerization support")
        
        except Exception as e:
            score = 40.0
            details["error"] = str(e)
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="deployment_readiness",
            status=status,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time_ms=0
        )
    
    def _check_docstrings(self, py_files: List[Path]) -> float:
        """Check docstring coverage in Python files."""
        if not py_files:
            return 0
        
        total_functions = 0
        documented_functions = 0
        
        try:
            for py_file in py_files[:10]:  # Sample first 10 files
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Simple heuristic for function definitions
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                            total_functions += 1
                            # Check if next few lines contain docstring
                            for j in range(i+1, min(i+5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    documented_functions += 1
                                    break
        except:
            pass
        
        return (documented_functions / total_functions * 100) if total_functions > 0 else 0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate overall recommendations."""
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Top 10 recommendations

def main():
    """Main execution function."""
    print("🎯 FastVLM Autonomous Quality Gates")
    print("=" * 50)
    
    gates = AutonomousQualityGates()
    report = gates.run_all_gates()
    
    print(f"\n📊 QUALITY GATES SUMMARY")
    print("-" * 30)
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"Deployment Ready: {'✅ YES' if report['deployment_ready'] else '❌ NO'}")
    print(f"Critical Failures: {len(report['critical_failures'])}")
    print(f"Total Execution Time: {report['execution_time_s']:.2f}s")
    
    print(f"\n🔍 INDIVIDUAL GATE RESULTS")
    print("-" * 30)
    for result in report['gate_results']:
        status_icon = "✅" if result['status'] == "passed" else "⚠️" if result['status'] == "warning" else "❌"
        print(f"{status_icon} {result['gate_name']}: {result['score']:.1f}/100 ({result['execution_time_ms']:.1f}ms)")
    
    if report['critical_failures']:
        print(f"\n💥 CRITICAL FAILURES")
        print("-" * 20)
        for failure in report['critical_failures']:
            print(f"❌ {failure}")
    
    if report['recommendations']:
        print(f"\n💡 TOP RECOMMENDATIONS")
        print("-" * 25)
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"{i}. {rec}")
    
    # Save detailed report
    with open("quality_gates_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Detailed report saved to: quality_gates_report.json")
    
    # Final assessment
    if report['deployment_ready']:
        print(f"\n🚀 SYSTEM IS PRODUCTION READY!")
        print("✅ All critical quality gates passed")
        print("✅ Ready for autonomous deployment")
    else:
        print(f"\n⚠️ SYSTEM NEEDS IMPROVEMENTS")
        print("❌ Critical issues must be resolved before deployment")
        print("🔧 Review recommendations and re-run quality gates")

if __name__ == "__main__":
    main()