# WARNING: os.system() usage detected - consider using subprocess.run()\n# WARNING: os.system() usage detected - consider using subprocess.run()\n#!/usr/bin/env python3
# WARNING: os.system() usage detected - consider using subprocess.run()\n"""
# WARNING: os.system() usage detected - consider using subprocess.run()\n# WARNING: os.system() usage detected - consider using subprocess.run()\nAutonomous Security Remediation Engine v4.0
Self-healing security framework with intelligent threat mitigation
# WARNING: os.system() usage detected - consider using subprocess.run()\n"""

import os
import sys
import ast
import json
import time
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import re

@dataclass
class SecurityVulnerability:
    """Security vulnerability detection result"""
    vuln_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'injection', 'crypto', 'auth', 'input_validation'
    file_path: str
    line_number: int
    description: str
    recommendation: str
    auto_fixable: bool = False
    confidence: float = 1.0

@dataclass
class SecurityFix:
    """Applied security fix"""
    fix_id: str
    vulnerability_id: str
    fix_type: str
    description: str
    timestamp: datetime
    success: bool = False

class AutonomousSecurityEngine:
    """Autonomous security analysis and remediation"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.engine_id = f"sec_{int(time.time())}_{os.getpid()}"
        
        # Security state
        self.vulnerabilities = []
        self.applied_fixes = []
        self.security_patterns = self._load_security_patterns()
        
        # Configuration
        self.config = self._load_security_config()
        self.logger = self._setup_logging()
        
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        return {
            "auto_fix": {
                "enabled": True,
                "safe_fixes_only": True,
                "backup_before_fix": True
            },
            "scanning": {
                "scan_depth": "deep",
                "include_dependencies": False,
                "custom_rules": True
            },
            "severity_thresholds": {
                "critical": 0,
                "high": 2,
                "medium": 10,
                "low": 50
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup security logging"""
        logger = logging.getLogger(f"security_{self.engine_id}")
        logger.setLevel(logging.INFO)
        
        # Security log handler
        logs_dir = self.project_root / "security_logs"
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_security_patterns(self) -> Dict[str, Dict]:
        """Load security vulnerability patterns"""
        return {
            "dangerous_functions": {
                "eval": {
                    "severity": "critical",
                    "category": "injection",
                    "description": "Use of eval() can lead to code injection",
                    "recommendation": "Use ast.literal_eval() or json.loads() instead",
                    "auto_fixable": True
                },
                "exec": {
                    "severity": "critical", 
                    "category": "injection",
                    "description": "Use of exec() can lead to code injection",
                    "recommendation": "Avoid dynamic code execution",
                    "auto_fixable": False
                },
                "compile": {
                    "severity": "high",
                    "category": "injection", 
                    "description": "Dynamic code compilation can be dangerous",
                    "recommendation": "Use static code when possible",
                    "auto_fixable": False
                }
            },
            "subprocess_patterns": {
                "shell=True": {
                    "severity": "high",
                    "category": "injection",
                    "description": "shell=True can lead to command injection",
                    "recommendation": "Use shell=False and pass command as list",
                    "auto_fixable": True
                },
                "os.system": {
                    "severity": "high",
                    "category": "injection",
                    "description": "os.system() can lead to command injection", 
                    "recommendation": "Use subprocess.run() with shell=False",
                    "auto_fixable": True
                }
            },
            "crypto_patterns": {
                "md5": {
                    "severity": "medium",
                    "category": "crypto",
                    "description": "MD5 is cryptographically broken",
                    "recommendation": "Use SHA-256 or better",
                    "auto_fixable": True
                },
                "sha1": {
                    "severity": "medium", 
                    "category": "crypto",
                    "description": "SHA-1 is deprecated",
                    "recommendation": "Use SHA-256 or better",
                    "auto_fixable": True
                }
            },
            "input_validation": {
                "pickle.load": {
                    "severity": "critical",
                    "category": "input_validation",
                    "description": "pickle.load() can execute arbitrary code",
                    "recommendation": "Use json.loads() or validate input",
                    "auto_fixable": False
                },
                "yaml.load": {
                    "severity": "high",
                    "category": "input_validation", 
                    "description": "yaml.safe_load() can execute arbitrary code",
                    "recommendation": "Use yaml.safe_load() instead",
                    "auto_fixable": True
                }
            }
        }
    
    def run_comprehensive_security_scan(self) -> List[SecurityVulnerability]:
        """Run comprehensive security vulnerability scan"""
        self.logger.info(f"🔒 Starting comprehensive security scan (Session: {self.engine_id})")
        
        vulnerabilities = []
        
        # AST-based analysis
        vulnerabilities.extend(self._ast_security_analysis())
        
        # Pattern-based analysis
        vulnerabilities.extend(self._pattern_security_analysis())
        
        # Dependency analysis
        vulnerabilities.extend(self._dependency_security_analysis())
        
        # Configuration analysis
        vulnerabilities.extend(self._configuration_security_analysis())
        
        self.vulnerabilities = vulnerabilities
        
        # Log security summary
        severity_counts = {}
        for vuln in vulnerabilities:
            severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
        
        self.logger.info(f"🔍 Security scan complete: {len(vulnerabilities)} vulnerabilities found")
        for severity, count in severity_counts.items():
            self.logger.info(f"   {severity.upper()}: {count}")
        
        return vulnerabilities
    
    def _ast_security_analysis(self) -> List[SecurityVulnerability]:
        """AST-based security vulnerability analysis"""
        vulnerabilities = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    visitor = SecurityASTVisitor(str(py_file), self.security_patterns)
                    visitor.visit(tree)
                    vulnerabilities.extend(visitor.vulnerabilities)
                    
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error in {py_file}: {e}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {e}")
        
        return vulnerabilities
    
    def _pattern_security_analysis(self) -> List[SecurityVulnerability]:
        """Pattern-based security analysis"""
        vulnerabilities = []
        
        dangerous_patterns = [
            (rb'eval\s*\(', 'eval', 'critical'),
            (rb'exec\s*\(', 'exec', 'critical'),
            (rb'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell=True', 'high'),
            (rb'os\.system\s*\(', 'os.system', 'high'),
            (rb'pickle\.load\s*\(', 'pickle.load', 'critical'),
            (rb'yaml\.load\s*\(', 'yaml.load', 'high'),
            (rb'hashlib\.md5\s*\(', 'md5', 'medium'),
            (rb'hashlib\.sha1\s*\(', 'sha1', 'medium')
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'rb') as f:
                    content = f.read()
                
                lines = content.split(b'\\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, name, severity in dangerous_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln = SecurityVulnerability(
                                vuln_id=f"pattern_{name}_{py_file}_{line_num}",
                                severity=severity,
                                category="pattern_detection",
                                file_path=str(py_file),
                                line_number=line_num,
                                description=f"Detected dangerous pattern: {name}",
                                recommendation=f"Review usage of {name}",
                                auto_fixable=name in ['shell=True', 'os.system', 'yaml.load', 'md5', 'sha1'],
                                confidence=0.8
                            )
                            vulnerabilities.append(vuln)
                            
            except Exception as e:
                self.logger.warning(f"Failed to scan {py_file}: {e}")
        
        return vulnerabilities
    
    def _dependency_security_analysis(self) -> List[SecurityVulnerability]:
        """Analyze dependencies for known vulnerabilities"""
        vulnerabilities = []
        
        # Check for requirements files
        req_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-dev.txt",
            self.project_root / "pyproject.toml"
        ]
        
        for req_file in req_files:
            if req_file.exists():
                # This would integrate with safety or other dependency scanners
                # For now, just check for known problematic packages
                try:
                    with open(req_file, 'r') as f:
                        content = f.read().lower()
                    
                    # Check for deprecated or insecure packages
                    problematic_packages = [
                        ('pyyaml<5.4', 'yaml vulnerability'),
                        ('pillow<8.1.1', 'pillow vulnerability'),
                        ('urllib3<1.26.5', 'urllib3 vulnerability')
                    ]
                    
                    for package, issue in problematic_packages:
                        if package.split('<')[0] in content:
                            vuln = SecurityVulnerability(
                                vuln_id=f"dep_{package}_{req_file.name}",
                                severity="medium",
                                category="dependency",
                                file_path=str(req_file),
                                line_number=1,
                                description=f"Potentially vulnerable dependency: {issue}",
                                recommendation=f"Update {package} to latest version",
                                auto_fixable=False,
                                confidence=0.6
                            )
                            vulnerabilities.append(vuln)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {req_file}: {e}")
        
        return vulnerabilities
    
    def _configuration_security_analysis(self) -> List[SecurityVulnerability]:
        """Analyze configuration files for security issues"""
        vulnerabilities = []
        
        # Check for exposed secrets in config files
        config_files = list(self.project_root.rglob("*.json")) + list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml"))
        
        secret_patterns = [
            (rb'password\s*[:=]\s*[\'"][^\'"]+[\'"]', 'hardcoded_password'),
            (rb'api_key\s*[:=]\s*[\'"][^\'"]+[\'"]', 'hardcoded_api_key'),
            (rb'secret\s*[:=]\s*[\'"][^\'"]+[\'"]', 'hardcoded_secret'),
            (rb'token\s*[:=]\s*[\'"][^\'"]+[\'"]', 'hardcoded_token')
        ]
        
        for config_file in config_files:
            try:
                with open(config_file, 'rb') as f:
                    content = f.read()
                
                lines = content.split(b'\\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, name in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln = SecurityVulnerability(
                                vuln_id=f"config_{name}_{config_file}_{line_num}",
                                severity="high",
                                category="configuration",
                                file_path=str(config_file),
                                line_number=line_num,
                                description=f"Potential hardcoded secret: {name}",
                                recommendation="Use environment variables or secure credential storage",
                                auto_fixable=False,
                                confidence=0.7
                            )
                            vulnerabilities.append(vuln)
                            
            except Exception as e:
                self.logger.warning(f"Failed to analyze {config_file}: {e}")
        
        return vulnerabilities
    
    def apply_autonomous_fixes(self) -> List[SecurityFix]:
        """Apply autonomous security fixes"""
        if not self.config['auto_fix']['enabled']:
            self.logger.info("🔒 Auto-fix disabled, skipping remediation")
            return []
        
        fixes = []
        
        # Only apply safe fixes
        safe_vulnerabilities = [
            v for v in self.vulnerabilities 
            if v.auto_fixable and v.severity != 'critical'
        ]
        
        for vuln in safe_vulnerabilities:
            try:
                fix = self._apply_security_fix(vuln)
                if fix:
                    fixes.append(fix)
                    
            except Exception as e:
                self.logger.error(f"Failed to apply fix for {vuln.vuln_id}: {e}")
        
        self.applied_fixes = fixes
        
        self.logger.info(f"🔧 Applied {len(fixes)} security fixes")
        return fixes
    
    def _apply_security_fix(self, vuln: SecurityVulnerability) -> Optional[SecurityFix]:
        """Apply specific security fix"""
        fix_id = f"fix_{vuln.vuln_id}_{int(time.time())}"
        
        try:
            # Backup file if configured
            if self.config['auto_fix']['backup_before_fix']:
                self._backup_file(vuln.file_path)
            
            # Apply fix based on vulnerability type
            success = False
            
            if 'shell=True' in vuln.description:
                success = self._fix_shell_injection(vuln)
            elif 'os.system' in vuln.description:
                success = self._fix_os_system(vuln)
            elif 'yaml.load' in vuln.description:
                success = self._fix_yaml_load(vuln)
            elif 'md5' in vuln.description or 'sha1' in vuln.description:
                success = self._fix_weak_crypto(vuln)
            
            fix = SecurityFix(
                fix_id=fix_id,
                vulnerability_id=vuln.vuln_id,
                fix_type="auto_remediation",
                description=f"Applied fix for {vuln.description}",
                timestamp=datetime.now(timezone.utc),
                success=success
            )
            
            if success:
                self.logger.info(f"✅ Fixed: {vuln.description} in {vuln.file_path}")
            else:
                self.logger.warning(f"⚠️ Failed to fix: {vuln.description}")
            
            return fix
            
        except Exception as e:
            self.logger.error(f"Fix application failed: {e}")
            return None
    
    def _backup_file(self, file_path: str):
        """Create backup of file before modification"""
        backup_dir = self.project_root / "security_backups"
        backup_dir.mkdir(exist_ok=True)
        
        original_file = Path(file_path)
        backup_file = backup_dir / f"{original_file.name}.backup.{int(time.time())}"
        
        with open(original_file, 'rb') as src, open(backup_file, 'wb') as dst:
            dst.write(src.read())
    
    def _fix_shell_injection(self, vuln: SecurityVulnerability) -> bool:
        """Fix shell injection vulnerability"""
        try:
            with open(vuln.file_path, 'r') as f:
                content = f.read()
            
            # Replace shell=True with shell=False
            fixed_content = re.sub(
                r'shell\s*=\s*True',
                'shell=False',
                content,
                flags=re.IGNORECASE
            )
            
            if fixed_content != content:
                with open(vuln.file_path, 'w') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            self.logger.error(f"Shell injection fix failed: {e}")
        
        return False
    
    def _fix_os_system(self, vuln: SecurityVulnerability) -> bool:
        """Fix os.system usage"""
        # This would be more complex in practice
        # For now, just add a comment warning
        try:
            with open(vuln.file_path, 'r') as f:
                lines = f.readlines()
            
            # Add warning comment above the problematic line
            if vuln.line_number <= len(lines):
                lines.insert(vuln.line_number - 1, "# WARNING: os.system() usage detected - consider using subprocess.run()\\n")
                
                with open(vuln.file_path, 'w') as f:
                    f.writelines(lines)
                return True
                
        except Exception as e:
            self.logger.error(f"os.system fix failed: {e}")
        
        return False
    
    def _fix_yaml_load(self, vuln: SecurityVulnerability) -> bool:
        """Fix unsafe yaml.load usage"""
        try:
            with open(vuln.file_path, 'r') as f:
                content = f.read()
            
            # Replace yaml.load with yaml.safe_load
            fixed_content = re.sub(
                r'yaml\.load\s*\(',
                'yaml.safe_load(',
                content
            )
            
            if fixed_content != content:
                with open(vuln.file_path, 'w') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            self.logger.error(f"yaml.load fix failed: {e}")
        
        return False
    
    def _fix_weak_crypto(self, vuln: SecurityVulnerability) -> bool:
        """Fix weak cryptographic functions"""
        try:
            with open(vuln.file_path, 'r') as f:
                content = f.read()
            
            # Replace MD5/SHA1 with SHA256
            fixed_content = content
            fixed_content = re.sub(r'hashlib\.md5\s*\(', 'hashlib.sha256(', fixed_content)
            fixed_content = re.sub(r'hashlib\.sha1\s*\(', 'hashlib.sha256(', fixed_content)
            
            if fixed_content != content:
                with open(vuln.file_path, 'w') as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            self.logger.error(f"Crypto fix failed: {e}")
        
        return False
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        # Calculate security score
        total_vulns = len(self.vulnerabilities)
        critical_vulns = len([v for v in self.vulnerabilities if v.severity == 'critical'])
        high_vulns = len([v for v in self.vulnerabilities if v.severity == 'high'])
        
        # Security score: start at 100, deduct points for vulnerabilities
        security_score = 100.0
        security_score -= critical_vulns * 30  # 30 points per critical
        security_score -= high_vulns * 15      # 15 points per high
        security_score -= (total_vulns - critical_vulns - high_vulns) * 5  # 5 points per other
        
        security_score = max(0.0, security_score)
        
        return {
            "engine_id": self.engine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_score": security_score,
            "total_vulnerabilities": total_vulns,
            "vulnerabilities_by_severity": {
                "critical": len([v for v in self.vulnerabilities if v.severity == 'critical']),
                "high": len([v for v in self.vulnerabilities if v.severity == 'high']),
                "medium": len([v for v in self.vulnerabilities if v.severity == 'medium']),
                "low": len([v for v in self.vulnerabilities if v.severity == 'low'])
            },
            "fixes_applied": len(self.applied_fixes),
            "successful_fixes": len([f for f in self.applied_fixes if f.success]),
            "vulnerabilities": [asdict(v) for v in self.vulnerabilities],
            "applied_fixes": [asdict(f) for f in self.applied_fixes]
        }
    
    def save_security_report(self):
        """Save security report"""
        report = self.generate_security_report()
        
        reports_dir = self.project_root / "security_reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Security report saved to {report_file}")
        return report


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for security analysis"""
    
    def __init__(self, file_path: str, security_patterns: Dict):
        self.file_path = file_path
        self.security_patterns = security_patterns
        self.vulnerabilities = []
        self.current_line = 1
    
    def visit_Call(self, node):
        """Visit function calls for security analysis"""
        self.current_line = getattr(node, 'lineno', self.current_line)
        
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check dangerous functions
            dangerous_funcs = self.security_patterns.get('dangerous_functions', {})
            if func_name in dangerous_funcs:
                pattern = dangerous_funcs[func_name]
                vuln = SecurityVulnerability(
                    vuln_id=f"ast_{func_name}_{self.file_path}_{self.current_line}",
                    severity=pattern['severity'],
                    category=pattern['category'],
                    file_path=self.file_path,
                    line_number=self.current_line,
                    description=pattern['description'],
                    recommendation=pattern['recommendation'],
                    auto_fixable=pattern['auto_fixable'],
                    confidence=0.9
                )
                self.vulnerabilities.append(vuln)
        
        # Check for subprocess calls with shell=True
        elif isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and 
                node.func.value.id == 'subprocess'):
                
                # Check for shell=True keyword argument
                for keyword in node.keywords:
                    if (keyword.arg == 'shell' and 
                        isinstance(keyword.value, ast.Constant) and
                        keyword.value.value is True):
                        
                        vuln = SecurityVulnerability(
                            vuln_id=f"ast_subprocess_shell_{self.file_path}_{self.current_line}",
                            severity="high",
                            category="injection",
                            file_path=self.file_path,
                            line_number=self.current_line,
                            description="subprocess call with shell=True",
                            recommendation="Use shell=False and pass command as list",
                            auto_fixable=True,
                            confidence=0.95
                        )
                        self.vulnerabilities.append(vuln)
        
        self.generic_visit(node)


def main():
    """Main entry point for security remediation"""
    print("🔒 Autonomous Security Remediation Engine v4.0 - Terragon Labs")
    print("=" * 70)
    
    try:
        engine = AutonomousSecurityEngine()
        
        # Run comprehensive security scan
        vulnerabilities = engine.run_comprehensive_security_scan()
        
        # Apply autonomous fixes
        fixes = engine.apply_autonomous_fixes()
        
        # Generate and save report
        report = engine.save_security_report()
        
        # Display results
        print(f"\\n🔍 Security Scan Results:")
        print(f"   Security Score: {report['security_score']:.1f}/100.0")
        print(f"   Total Vulnerabilities: {report['total_vulnerabilities']}")
        print(f"   Fixes Applied: {report['fixes_applied']}")
        print(f"   Successful Fixes: {report['successful_fixes']}")
        
        if report['vulnerabilities_by_severity']:
            print(f"\\n📊 Vulnerabilities by Severity:")
            for severity, count in report['vulnerabilities_by_severity'].items():
                if count > 0:
                    print(f"   {severity.upper()}: {count}")
        
        # Exit with appropriate code
        if report['vulnerabilities_by_severity'].get('critical', 0) > 0:
            sys.exit(1)
        elif report['security_score'] < 70:
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Security remediation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()