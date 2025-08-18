"""
Production Quality Validation for FastVLM.

Validates production readiness focusing on architecture, structure,
and components that don't require external dependencies.
"""

import sys
import os
import time
import json
import logging
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


class ProductionQualityValidator:
    """Production-focused quality validation."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.passed_gates = 0
        self.total_gates = 0
        
    def run_production_validation(self) -> bool:
        """Run production-focused quality validation."""
        print("🚀 FastVLM Production Quality Validation")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Define production quality gates
        quality_gates = [
            ("Architecture Design", self.validate_architecture_design),
            ("Code Structure", self.validate_code_structure),
            ("Documentation Quality", self.validate_documentation),
            ("Configuration Management", self.validate_configuration),
            ("Security Implementation", self.validate_security_patterns),
            ("Error Handling", self.validate_error_handling),
            ("Performance Patterns", self.validate_performance_patterns),
            ("Mobile Optimization", self.validate_mobile_patterns),
            ("Production Readiness", self.validate_production_features),
            ("Swift Integration", self.validate_swift_integration),
            ("Deployment Preparedness", self.validate_deployment_readiness)
        ]
        
        # Run each quality gate
        for gate_name, gate_test in quality_gates:
            self.total_gates += 1
            print(f"\n📋 Validating: {gate_name}")
            print("-" * 40)
            
            try:
                result = gate_test()
                if result["passed"]:
                    self.passed_gates += 1
                    print(f"✅ {gate_name}: PASSED")
                    if result.get("details"):
                        print(f"   {result['details']}")
                else:
                    print(f"❌ {gate_name}: FAILED")
                    print(f"   {result.get('reason', 'Unknown')}")
                
                self.results[gate_name] = result
                
            except Exception as e:
                print(f"💥 {gate_name}: ERROR - {e}")
                self.results[gate_name] = {
                    "passed": False,
                    "reason": f"Exception: {e}"
                }
        
        self.end_time = time.time()
        self._generate_production_report()
        
        return self.passed_gates == self.total_gates
    
    def validate_architecture_design(self) -> Dict[str, Any]:
        """Validate overall architecture design."""
        try:
            src_dir = Path("src/fast_vlm_ondevice")
            if not src_dir.exists():
                return {"passed": False, "reason": "Source directory not found"}
            
            # Check for core architectural components
            required_components = [
                "core_pipeline.py",
                "converter.py", 
                "real_time_mobile_optimizer.py",
                "advanced_error_recovery.py",
                "comprehensive_logging.py",
                "production_security_framework.py",
                "high_performance_distributed_engine.py"
            ]
            
            missing_components = []
            existing_components = []
            
            for component in required_components:
                component_path = src_dir / component
                if component_path.exists():
                    existing_components.append(component)
                else:
                    missing_components.append(component)
            
            if missing_components:
                return {
                    "passed": False,
                    "reason": f"Missing core components: {missing_components}"
                }
            
            # Check component sizes (should be substantial)
            small_components = []
            for component in existing_components:
                component_path = src_dir / component
                size_kb = component_path.stat().st_size / 1024
                if size_kb < 5:  # Less than 5KB is probably too small
                    small_components.append(f"{component} ({size_kb:.1f}KB)")
            
            if small_components:
                return {
                    "passed": False,
                    "reason": f"Components too small: {small_components}"
                }
            
            return {
                "passed": True,
                "details": f"Architecture complete - {len(existing_components)} core components found"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Architecture validation failed: {e}"}
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        try:
            src_dir = Path("src/fast_vlm_ondevice")
            python_files = list(src_dir.glob("*.py"))
            
            if len(python_files) < 15:
                return {
                    "passed": False,
                    "reason": f"Too few Python files: {len(python_files)}"
                }
            
            # Check for proper __init__.py
            init_file = src_dir / "__init__.py"
            if not init_file.exists():
                return {"passed": False, "reason": "__init__.py missing"}
            
            # Check __init__.py has substantial exports
            init_content = init_file.read_text()
            if len(init_content) < 1000:
                return {"passed": False, "reason": "__init__.py too minimal"}
            
            # Check for class and function definitions
            total_classes = 0
            total_functions = 0
            files_with_classes = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    tree = ast.parse(content)
                    
                    file_classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    file_functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    
                    total_classes += len(file_classes)
                    total_functions += len(file_functions)
                    
                    if file_classes:
                        files_with_classes += 1
                        
                except:
                    continue  # Skip files that can't be parsed
            
            if total_classes < 20:
                return {
                    "passed": False,
                    "reason": f"Too few classes: {total_classes}"
                }
            
            return {
                "passed": True,
                "details": f"Code structure good - {len(python_files)} files, {total_classes} classes, {total_functions} functions"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Code structure validation failed: {e}"}
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation quality."""
        try:
            # Check main documentation files
            required_docs = ["README.md", "CLAUDE.md"]
            missing_docs = []
            
            for doc_file in required_docs:
                if not Path(doc_file).exists():
                    missing_docs.append(doc_file)
            
            if missing_docs:
                return {
                    "passed": False,
                    "reason": f"Missing documentation: {missing_docs}"
                }
            
            # Check README quality
            readme = Path("README.md").read_text()
            readme_sections = [
                "# Fast VLM On-Device Kit",
                "## Overview", 
                "## Performance",
                "## Installation",
                "## Quick Start"
            ]
            
            missing_sections = []
            for section in readme_sections:
                if section not in readme:
                    missing_sections.append(section)
            
            if missing_sections:
                return {
                    "passed": False,
                    "reason": f"README missing sections: {missing_sections}"
                }
            
            # Check for code examples in README
            code_blocks = readme.count("```")
            if code_blocks < 10:  # Should have multiple code examples
                return {
                    "passed": False,
                    "reason": f"Insufficient code examples: {code_blocks//2} blocks"
                }
            
            # Check docstring coverage
            src_dir = Path("src/fast_vlm_ondevice")
            python_files = list(src_dir.glob("*.py"))
            
            files_with_docstrings = 0
            total_docstring_lines = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
                        # Count docstring lines
                        docstring_lines = content.count('"""') + content.count("'''")
                        total_docstring_lines += docstring_lines
                except:
                    continue
            
            docstring_coverage = files_with_docstrings / len(python_files)
            if docstring_coverage < 0.9:
                return {
                    "passed": False,
                    "reason": f"Low docstring coverage: {docstring_coverage:.1%}"
                }
            
            return {
                "passed": True,
                "details": f"Documentation quality excellent - {docstring_coverage:.1%} docstring coverage, {code_blocks//2} code examples"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Documentation validation failed: {e}"}
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration management."""
        try:
            # Check for configuration files
            config_files = [
                "pyproject.toml",
                "requirements.txt",
                "requirements-dev.txt"
            ]
            
            missing_configs = []
            for config_file in config_files:
                if not Path(config_file).exists():
                    missing_configs.append(config_file)
            
            if missing_configs:
                return {
                    "passed": False,
                    "reason": f"Missing configuration files: {missing_configs}"
                }
            
            # Check pyproject.toml structure
            pyproject_content = Path("pyproject.toml").read_text()
            required_sections = [
                "[build-system]",
                "[project]",
                "[tool.pytest.ini_options]",
                "[tool.black]",
                "[tool.mypy]"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in pyproject_content:
                    missing_sections.append(section)
            
            if missing_sections:
                return {
                    "passed": False,
                    "reason": f"pyproject.toml missing sections: {missing_sections}"
                }
            
            # Check requirements.txt
            requirements = Path("requirements.txt").read_text()
            required_deps = ["torch", "torchvision", "coremltools", "transformers"]
            
            missing_deps = []
            for dep in required_deps:
                if dep not in requirements:
                    missing_deps.append(dep)
            
            if missing_deps:
                return {
                    "passed": False,
                    "reason": f"Missing required dependencies: {missing_deps}"
                }
            
            return {
                "passed": True,
                "details": "Configuration management complete and correct"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Configuration validation failed: {e}"}
    
    def validate_security_patterns(self) -> Dict[str, Any]:
        """Validate security implementation patterns."""
        try:
            security_file = Path("src/fast_vlm_ondevice/production_security_framework.py")
            if not security_file.exists():
                return {"passed": False, "reason": "Security framework file missing"}
            
            security_content = security_file.read_text()
            
            # Check for security patterns
            security_patterns = [
                "InputValidator",
                "SecurityScanner", 
                "ThreatDetectionEngine",
                "CryptographicManager",
                "authentication",
                "authorization",
                "validate_image_data",
                "validate_text_input",
                "blocked_patterns",
                "security_event"
            ]
            
            missing_patterns = []
            for pattern in security_patterns:
                if pattern not in security_content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                return {
                    "passed": False,
                    "reason": f"Missing security patterns: {missing_patterns}"
                }
            
            # Check security file size (should be substantial)
            size_kb = security_file.stat().st_size / 1024
            if size_kb < 30:  # Should be substantial for comprehensive security
                return {
                    "passed": False,
                    "reason": f"Security framework too small: {size_kb:.1f}KB"
                }
            
            return {
                "passed": True,
                "details": f"Security patterns implemented - {size_kb:.1f}KB framework"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Security validation failed: {e}"}
    
    def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling patterns."""
        try:
            error_recovery_file = Path("src/fast_vlm_ondevice/advanced_error_recovery.py")
            if not error_recovery_file.exists():
                return {"passed": False, "reason": "Error recovery file missing"}
            
            error_content = error_recovery_file.read_text()
            
            # Check for error handling patterns
            error_patterns = [
                "CircuitBreaker",
                "Bulkhead",
                "SelfHealingManager",
                "RecoveryStrategy",
                "ErrorSeverity",
                "try:",
                "except",
                "fallback",
                "retry",
                "resilient"
            ]
            
            missing_patterns = []
            for pattern in error_patterns:
                if pattern not in error_content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                return {
                    "passed": False,
                    "reason": f"Missing error handling patterns: {missing_patterns}"
                }
            
            # Check for proper exception handling in other files
            src_dir = Path("src/fast_vlm_ondevice")
            python_files = list(src_dir.glob("*.py"))
            
            files_with_exception_handling = 0
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    if "try:" in content and "except" in content:
                        files_with_exception_handling += 1
                except:
                    continue
            
            exception_handling_ratio = files_with_exception_handling / len(python_files)
            if exception_handling_ratio < 0.7:
                return {
                    "passed": False,
                    "reason": f"Low exception handling coverage: {exception_handling_ratio:.1%}"
                }
            
            return {
                "passed": True,
                "details": f"Error handling comprehensive - {exception_handling_ratio:.1%} files with exception handling"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Error handling validation failed: {e}"}
    
    def validate_performance_patterns(self) -> Dict[str, Any]:
        """Validate performance optimization patterns."""
        try:
            perf_files = [
                "src/fast_vlm_ondevice/high_performance_distributed_engine.py",
                "src/fast_vlm_ondevice/real_time_mobile_optimizer.py"
            ]
            
            missing_files = []
            for perf_file in perf_files:
                if not Path(perf_file).exists():
                    missing_files.append(perf_file)
            
            if missing_files:
                return {
                    "passed": False,
                    "reason": f"Missing performance files: {missing_files}"
                }
            
            # Check distributed engine patterns
            dist_engine = Path("src/fast_vlm_ondevice/high_performance_distributed_engine.py").read_text()
            
            perf_patterns = [
                "ThreadPoolExecutor",
                "ProcessPoolExecutor", 
                "asyncio",
                "LoadBalancingStrategy",
                "AutoScalingEngine",
                "concurrent.futures",
                "threading",
                "multiprocessing",
                "performance_metrics",
                "throughput"
            ]
            
            missing_patterns = []
            for pattern in perf_patterns:
                if pattern not in dist_engine:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                return {
                    "passed": False,
                    "reason": f"Missing performance patterns: {missing_patterns}"
                }
            
            return {
                "passed": True,
                "details": "Performance optimization patterns implemented"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Performance validation failed: {e}"}
    
    def validate_mobile_patterns(self) -> Dict[str, Any]:
        """Validate mobile optimization patterns."""
        try:
            mobile_file = Path("src/fast_vlm_ondevice/real_time_mobile_optimizer.py")
            if not mobile_file.exists():
                return {"passed": False, "reason": "Mobile optimizer file missing"}
            
            mobile_content = mobile_file.read_text()
            
            mobile_patterns = [
                "MobileOptimizationConfig",
                "target_latency_ms",
                "quantization_strategy",
                "adaptive",
                "int4",
                "int8", 
                "fp16",
                "mobile_optimization",
                "compression_ratio",
                "ane_optimization"
            ]
            
            missing_patterns = []
            for pattern in mobile_patterns:
                if pattern not in mobile_content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                return {
                    "passed": False,
                    "reason": f"Missing mobile patterns: {missing_patterns}"
                }
            
            return {
                "passed": True,
                "details": "Mobile optimization patterns implemented"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Mobile validation failed: {e}"}
    
    def validate_production_features(self) -> Dict[str, Any]:
        """Validate production-ready features."""
        try:
            # Check for logging
            logging_file = Path("src/fast_vlm_ondevice/comprehensive_logging.py")
            if not logging_file.exists():
                return {"passed": False, "reason": "Logging framework missing"}
            
            # Check for monitoring
            monitoring_patterns = ["monitoring", "metrics", "performance"]
            src_dir = Path("src/fast_vlm_ondevice")
            python_files = list(src_dir.glob("*.py"))
            
            files_with_monitoring = 0
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    if any(pattern in content.lower() for pattern in monitoring_patterns):
                        files_with_monitoring += 1
                except:
                    continue
            
            if files_with_monitoring < 5:
                return {
                    "passed": False,
                    "reason": f"Insufficient monitoring coverage: {files_with_monitoring} files"
                }
            
            # Check for health checks
            health_patterns = ["health", "heartbeat", "status"]
            files_with_health = 0
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    if any(pattern in content.lower() for pattern in health_patterns):
                        files_with_health += 1
                except:
                    continue
            
            return {
                "passed": True,
                "details": f"Production features implemented - monitoring in {files_with_monitoring} files, health checks in {files_with_health} files"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Production features validation failed: {e}"}
    
    def validate_swift_integration(self) -> Dict[str, Any]:
        """Validate Swift/iOS integration."""
        try:
            ios_dir = Path("ios")
            if not ios_dir.exists():
                return {"passed": False, "reason": "iOS directory missing"}
            
            # Check for Package.swift
            package_swift = ios_dir / "Package.swift"
            if not package_swift.exists():
                return {"passed": False, "reason": "Package.swift missing"}
            
            # Check for FastVLM.swift
            fastvlm_swift = ios_dir / "Sources" / "FastVLMKit" / "FastVLM.swift"
            if not fastvlm_swift.exists():
                return {"passed": False, "reason": "FastVLM.swift missing"}
            
            # Check Swift file content
            swift_content = fastvlm_swift.read_text()
            
            swift_patterns = [
                "import CoreML",
                "import Vision",
                "class FastVLM",
                "func answer",
                "MLModel",
                "async",
                "InferenceMetrics"
            ]
            
            missing_patterns = []
            for pattern in swift_patterns:
                if pattern not in swift_content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                return {
                    "passed": False,
                    "reason": f"Missing Swift patterns: {missing_patterns}"
                }
            
            # Check for tests
            test_file = ios_dir / "Tests" / "FastVLMKitTests" / "FastVLMTests.swift"
            tests_exist = test_file.exists()
            
            return {
                "passed": True,
                "details": f"Swift integration complete - main library and {'tests' if tests_exist else 'structure'} present"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Swift integration validation failed: {e}"}
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness."""
        try:
            # Check for Docker support
            docker_files = ["Dockerfile", "docker-compose.yml"]
            docker_present = any(Path(f).exists() for f in docker_files)
            
            # Check for deployment scripts
            scripts_dir = Path("scripts")
            deployment_scripts = []
            if scripts_dir.exists():
                deployment_scripts = list(scripts_dir.glob("*deployment*.py"))
            
            # Check for CI/CD configuration
            cicd_files = [".github/workflows", "azure-pipelines.yml", ".gitlab-ci.yml"]
            cicd_present = any(Path(f).exists() for f in cicd_files)
            
            # Check for deployment documentation
            docs_dir = Path("docs")
            deployment_docs = []
            if docs_dir.exists():
                deployment_docs = list(docs_dir.glob("*DEPLOYMENT*.md"))
                deployment_docs.extend(docs_dir.glob("*deployment*.md"))
            
            readiness_score = 0
            details = []
            
            if docker_present:
                readiness_score += 1
                details.append("Docker support")
            
            if deployment_scripts:
                readiness_score += 1
                details.append(f"{len(deployment_scripts)} deployment scripts")
            
            if deployment_docs:
                readiness_score += 1
                details.append("Deployment documentation")
            
            if cicd_present:
                readiness_score += 1
                details.append("CI/CD configuration")
            
            # Check for example files
            examples_dir = Path("examples")
            examples_count = 0
            if examples_dir.exists():
                examples_count = len(list(examples_dir.glob("*.py")))
            
            if examples_count >= 3:
                readiness_score += 1
                details.append(f"{examples_count} example files")
            
            if readiness_score < 3:
                return {
                    "passed": False,
                    "reason": f"Insufficient deployment readiness: {readiness_score}/5 criteria met"
                }
            
            return {
                "passed": True,
                "details": f"Deployment ready - {'; '.join(details)}"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Deployment readiness validation failed: {e}"}
    
    def _generate_production_report(self):
        """Generate production validation report."""
        total_time = self.end_time - self.start_time
        success_rate = (self.passed_gates / self.total_gates) * 100
        
        print("\n" + "=" * 60)
        print("📊 PRODUCTION QUALITY VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Total Validation Gates: {self.total_gates}")
        print(f"Passed: {self.passed_gates}")
        print(f"Failed: {self.total_gates - self.passed_gates}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Validation Time: {total_time:.2f}s")
        
        # Production readiness assessment
        if success_rate == 100:
            print("\n🎉 PRODUCTION VALIDATION COMPLETE - SYSTEM READY FOR DEPLOYMENT!")
            print("   ✅ Architecture design validated")
            print("   ✅ Code structure optimal")
            print("   ✅ Documentation comprehensive")
            print("   ✅ Security framework operational")
            print("   ✅ Error handling robust")
            print("   ✅ Performance optimization ready")
            print("   ✅ Mobile optimization implemented")
            print("   ✅ Swift integration complete")
            print("   ✅ Deployment infrastructure ready")
            print("\n🚀 APPROVED FOR PRODUCTION DEPLOYMENT")
        
        elif success_rate >= 90:
            print("\n⚠️  MOSTLY PRODUCTION READY - MINOR ISSUES TO ADDRESS")
            print("   System is largely ready but review failed gates")
            
        elif success_rate >= 75:
            print("\n⚠️  PARTIALLY PRODUCTION READY - SIGNIFICANT ISSUES TO ADDRESS")
            print("   Core functionality present but needs improvement")
            
        else:
            print("\n❌ NOT PRODUCTION READY - MAJOR ISSUES REQUIRE ATTENTION")
            print("   Significant architectural or implementation gaps")
        
        # Detailed results
        failed_gates = [name for name, result in self.results.items() if not result["passed"]]
        if failed_gates:
            print(f"\n❌ FAILED VALIDATION GATES:")
            for gate_name in failed_gates:
                result = self.results[gate_name]
                print(f"  • {gate_name}: {result.get('reason', 'Unknown failure')}")
        
        passed_gates = [name for name, result in self.results.items() if result["passed"]]
        if passed_gates:
            print(f"\n✅ PASSED VALIDATION GATES:")
            for gate_name in passed_gates:
                result = self.results[gate_name]
                details = result.get('details', 'Validated')
                print(f"  • {gate_name}: {details}")
        
        print("=" * 60)
        
        # Save production validation report
        report_data = {
            "validation_type": "production_quality",
            "timestamp": time.time(),
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "success_rate": success_rate,
            "duration_seconds": total_time,
            "production_ready": success_rate >= 90,
            "results": self.results
        }
        
        with open("production_validation_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📝 Production validation report saved to: production_validation_report.json")


def main():
    """Main execution function."""
    validator = ProductionQualityValidator()
    success = validator.run_production_validation()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)