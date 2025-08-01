#!/usr/bin/env python3
"""
Continuous SDLC Optimization Engine
Monitors repository health and executes continuous improvements.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousOptimizer:
    """Continuous optimization engine for advanced repositories."""
    
    def __init__(self, repo_path: Path = Path.cwd()):
        self.repo_path = repo_path
        self.metrics_path = repo_path / ".terragon" / "optimization-metrics.json"
        self.metrics_path.parent.mkdir(exist_ok=True)
        
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive repository analysis."""
        logger.info("🔍 Running comprehensive repository analysis...")
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'repository_health': {},
            'code_quality': {},
            'security_posture': {},
            'performance_metrics': {},
            'automation_status': {},
            'recommendations': []
        }
        
        # Parallel analysis tasks
        analysis_tasks = [
            self._analyze_code_quality(),
            self._analyze_security_posture(),
            self._analyze_test_coverage(),
            self._analyze_dependency_health(),
            self._analyze_automation_status(),
            self._analyze_documentation_coverage()
        ]
        
        task_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Compile results
        for i, result in enumerate(task_results):
            if isinstance(result, dict):
                task_name = [
                    'code_quality', 'security_posture', 'test_coverage',
                    'dependency_health', 'automation_status', 'documentation'
                ][i]
                results[task_name] = result
            elif isinstance(result, Exception):
                logger.warning(f"Analysis task {i} failed: {result}")
        
        # Generate overall health score
        results['repository_health'] = self._calculate_health_score(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    async def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        metrics = {
            'complexity_analysis': {},
            'type_coverage': 0.0,
            'lint_issues': 0,
            'code_duplication': 0.0
        }
        
        try:
            # Run MyPy for type analysis
            result = subprocess.run([
                'mypy', 'src/', '--txt-report', '/tmp/mypy-report'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                metrics['type_coverage'] = 95.0  # High type coverage
            else:
                # Count type errors
                type_errors = result.stdout.count('error:')
                metrics['type_coverage'] = max(0, 90 - (type_errors * 2))
            
            # Run flake8 for linting
            result = subprocess.run([
                'flake8', 'src/', '--statistics'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                # Count total issues
                lines = result.stdout.strip().split('\n')
                metrics['lint_issues'] = len([l for l in lines if l.strip() and not l.startswith('--')])
            
        except Exception as e:
            logger.warning(f"Code quality analysis failed: {e}")
        
        return metrics
    
    async def _analyze_security_posture(self) -> Dict[str, Any]:
        """Analyze security posture."""
        metrics = {
            'vulnerability_count': 0,
            'security_score': 100.0,
            'outdated_dependencies': 0,
            'security_tools_active': []
        }
        
        try:
            # Run safety check
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulns = len(safety_data.get('vulnerabilities', []))
                metrics['vulnerability_count'] = vulns
                metrics['security_score'] = max(0, 100 - (vulns * 10))
            
            # Check for security tools
            if (self.repo_path / '.pre-commit-config.yaml').exists():
                with open(self.repo_path / '.pre-commit-config.yaml') as f:
                    content = f.read()
                    if 'bandit' in content:
                        metrics['security_tools_active'].append('bandit')
                    if 'safety' in content:
                        metrics['security_tools_active'].append('safety')
                    if 'ggshield' in content or 'gitguardian' in content:
                        metrics['security_tools_active'].append('gitguardian')
            
        except Exception as e:
            logger.warning(f"Security analysis failed: {e}")
        
        return metrics
    
    async def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage."""
        metrics = {
            'coverage_percentage': 0.0,
            'test_count': 0,
            'missing_coverage_files': []
        }
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'pytest', '--cov=src', '--cov-report=json:/tmp/coverage.json', '--quiet'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            coverage_file = Path('/tmp/coverage.json')
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    
                metrics['coverage_percentage'] = coverage_data.get('totals', {}).get('percent_covered', 0.0)
                
                # Find files with low coverage
                for filename, file_data in coverage_data.get('files', {}).items():
                    if file_data.get('summary', {}).get('percent_covered', 0) < 80:
                        metrics['missing_coverage_files'].append({
                            'file': filename,
                            'coverage': file_data.get('summary', {}).get('percent_covered', 0)
                        })
            
            # Count test files
            test_files = list(self.repo_path.rglob('test_*.py'))
            metrics['test_count'] = len(test_files)
            
        except Exception as e:
            logger.warning(f"Test coverage analysis failed: {e}")
        
        return metrics
    
    async def _analyze_dependency_health(self) -> Dict[str, Any]:
        """Analyze dependency health."""
        metrics = {
            'total_dependencies': 0,
            'outdated_count': 0,
            'vulnerable_count': 0,
            'license_issues': []
        }
        
        try:
            # Check for outdated packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                metrics['outdated_count'] = len(outdated)
            
            # Count total dependencies
            if (self.repo_path / 'requirements.txt').exists():
                with open(self.repo_path / 'requirements.txt') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    metrics['total_dependencies'] = len(deps)
            
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
        
        return metrics
    
    async def _analyze_automation_status(self) -> Dict[str, Any]:
        """Analyze automation status."""
        metrics = {
            'ci_cd_active': False,
            'pre_commit_configured': False,
            'automated_testing': False,
            'security_scanning': False,
            'dependency_updates': False
        }
        
        # Check for GitHub Actions
        github_workflows = self.repo_path / '.github' / 'workflows'
        if github_workflows.exists() and list(github_workflows.glob('*.yml')):
            metrics['ci_cd_active'] = True
            metrics['automated_testing'] = True
        
        # Check for pre-commit
        if (self.repo_path / '.pre-commit-config.yaml').exists():
            metrics['pre_commit_configured'] = True
            
            with open(self.repo_path / '.pre-commit-config.yaml') as f:
                content = f.read()
                if any(tool in content for tool in ['bandit', 'safety', 'ggshield']):
                    metrics['security_scanning'] = True
        
        # Check for Dependabot
        dependabot_config = self.repo_path / '.github' / 'dependabot.yml'
        if dependabot_config.exists():
            metrics['dependency_updates'] = True
        
        return metrics
    
    async def _analyze_documentation_coverage(self) -> Dict[str, Any]:
        """Analyze documentation coverage."""
        metrics = {
            'api_docs_coverage': 0.0,
            'readme_quality': 0.0,
            'missing_docs': []
        }
        
        try:
            # Check README quality
            readme_path = self.repo_path / 'README.md'
            if readme_path.exists():
                with open(readme_path) as f:
                    readme_content = f.read()
                    
                # Simple quality scoring based on content
                quality_indicators = [
                    'installation' in readme_content.lower(),
                    'usage' in readme_content.lower(),
                    'example' in readme_content.lower(),
                    'contributing' in readme_content.lower(),
                    'license' in readme_content.lower(),
                    len(readme_content) > 1000
                ]
                metrics['readme_quality'] = (sum(quality_indicators) / len(quality_indicators)) * 100
            
            # Check for API documentation
            docs_dir = self.repo_path / 'docs'
            if docs_dir.exists():
                api_docs = docs_dir / 'api'
                if api_docs.exists():
                    metrics['api_docs_coverage'] = 80.0  # Assume good coverage if exists
                else:
                    metrics['missing_docs'].append('API documentation')
            
        except Exception as e:
            logger.warning(f"Documentation analysis failed: {e}")
        
        return metrics
    
    def _calculate_health_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall repository health score."""
        scores = []
        
        # Code quality score (0-100)
        code_quality = results.get('code_quality', {})
        if code_quality:
            cq_score = (
                code_quality.get('type_coverage', 0) * 0.4 +
                max(0, 100 - code_quality.get('lint_issues', 0) * 2) * 0.6
            )
            scores.append(('code_quality', cq_score))
        
        # Security score (0-100)
        security = results.get('security_posture', {})
        if security:
            sec_score = security.get('security_score', 0)
            if security.get('security_tools_active'):
                sec_score += 10  # Bonus for active tools
            scores.append(('security', min(100, sec_score)))
        
        # Test coverage score (0-100)
        test_cov = results.get('test_coverage', {})
        if test_cov:
            scores.append(('testing', test_cov.get('coverage_percentage', 0)))
        
        # Automation score (0-100)
        automation = results.get('automation_status', {})
        if automation:
            auto_score = sum([
                25 if automation.get('ci_cd_active') else 0,
                20 if automation.get('pre_commit_configured') else 0,
                20 if automation.get('automated_testing') else 0,
                20 if automation.get('security_scanning') else 0,
                15 if automation.get('dependency_updates') else 0
            ])
            scores.append(('automation', auto_score))
        
        # Documentation score (0-100)
        docs = results.get('documentation', {})
        if docs:
            doc_score = (
                docs.get('readme_quality', 0) * 0.6 +
                docs.get('api_docs_coverage', 0) * 0.4
            )
            scores.append(('documentation', doc_score))
        
        # Calculate weighted overall score
        if scores:
            weights = {
                'code_quality': 0.25,
                'security': 0.25,
                'testing': 0.20,
                'automation': 0.20,
                'documentation': 0.10
            }
            
            overall_score = sum(
                score * weights.get(category, 0.1)
                for category, score in scores
            )
            
            return {
                'overall_score': round(overall_score, 1),
                'category_scores': dict(scores),
                'maturity_level': self._determine_maturity_level(overall_score),
                'improvement_priority': self._get_improvement_priorities(dict(scores))
            }
        
        return {'overall_score': 0.0, 'category_scores': {}}
    
    def _determine_maturity_level(self, score: float) -> str:
        """Determine maturity level based on score."""
        if score >= 85:
            return "Advanced"
        elif score >= 70:
            return "Maturing"
        elif score >= 50:
            return "Developing"
        else:
            return "Nascent"
    
    def _get_improvement_priorities(self, scores: Dict[str, float]) -> List[str]:
        """Get improvement priorities based on scores."""
        priority_order = []
        
        # Sort by score (lowest first)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        for category, score in sorted_scores:
            if score < 70:
                priority_order.append(category)
        
        return priority_order
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Security recommendations
        security = results.get('security_posture', {})
        if security.get('vulnerability_count', 0) > 0:
            recommendations.append({
                'category': 'security',
                'priority': 'high',
                'title': 'Address security vulnerabilities',
                'description': f"Found {security['vulnerability_count']} vulnerabilities in dependencies",
                'action': 'Run `safety check` and update vulnerable packages'
            })
        
        # Code quality recommendations
        code_quality = results.get('code_quality', {})
        if code_quality.get('type_coverage', 0) < 80:
            recommendations.append({
                'category': 'code_quality',
                'priority': 'medium',
                'title': 'Improve type annotations',
                'description': f"Type coverage is {code_quality.get('type_coverage', 0):.1f}%",
                'action': 'Add type hints to functions and methods'
            })
        
        # Test coverage recommendations
        test_cov = results.get('test_coverage', {})
        if test_cov.get('coverage_percentage', 0) < 80:
            recommendations.append({
                'category': 'testing',
                'priority': 'medium',
                'title': 'Increase test coverage',
                'description': f"Test coverage is {test_cov.get('coverage_percentage', 0):.1f}%",
                'action': 'Add tests for uncovered code paths'
            })
        
        # Automation recommendations
        automation = results.get('automation_status', {})
        if not automation.get('ci_cd_active', False):
            recommendations.append({
                'category': 'automation',
                'priority': 'high',
                'title': 'Set up CI/CD pipeline',
                'description': 'No automated testing and deployment detected',
                'action': 'Create GitHub Actions workflows for testing and deployment'
            })
        
        return recommendations
    
    async def save_metrics(self, analysis_results: Dict[str, Any]) -> None:
        """Save optimization metrics."""
        with open(self.metrics_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Optimization metrics saved to {self.metrics_path}")
    
    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run complete optimization analysis cycle."""
        logger.info("🚀 Starting continuous optimization cycle...")
        
        try:
            # Run comprehensive analysis
            results = await self.run_comprehensive_analysis()
            
            # Save metrics
            await self.save_metrics(results)
            
            # Log key metrics
            health = results.get('repository_health', {})
            overall_score = health.get('overall_score', 0)
            maturity = health.get('maturity_level', 'Unknown')
            
            logger.info(f"📊 Repository Health Score: {overall_score}/100 ({maturity})")
            
            recommendations = results.get('recommendations', [])
            if recommendations:
                logger.info(f"💡 Generated {len(recommendations)} optimization recommendations")
                for rec in recommendations[:3]:  # Show top 3
                    logger.info(f"  - {rec['title']} ({rec['priority']} priority)")
            
            logger.info("✅ Optimization cycle completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Optimization cycle failed: {e}")
            raise


async def main():
    """Main entry point for continuous optimization."""
    optimizer = ContinuousOptimizer()
    
    try:
        results = await optimizer.run_optimization_cycle()
        
        # Print summary
        health = results.get('repository_health', {})
        print(f"\n🏆 Repository Health: {health.get('overall_score', 0)}/100 ({health.get('maturity_level', 'Unknown')})")
        
        return 0
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)