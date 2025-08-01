#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for FastVLM On-Device Kit
Continuously discovers, scores, and executes highest-value SDLC improvements.
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
# import yaml  # Optional dependency, will use JSON fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValueItem:
    id: str
    title: str
    description: str
    category: str
    estimated_effort_hours: float
    
    # Scoring components
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Additional metadata
    files_affected: List[str] = None
    dependencies: List[str] = None
    risk_level: str = "medium"
    discovered_date: str = ""
    source: str = ""
    
    def __post_init__(self):
        if self.files_affected is None:
            self.files_affected = []
        if self.dependencies is None:
            self.dependencies = []
        if not self.discovered_date:
            self.discovered_date = datetime.now(timezone.utc).isoformat()


class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization engine."""
    
    def __init__(self, repo_path: Path = Path.cwd()):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.metrics_path = repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = repo_path / "BACKLOG.md"
        
        self.config = self._load_config()
        self.discovered_items: List[ValueItem] = []
        self.execution_history: List[Dict] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    return yaml.safe_load(f)
            except ImportError:
                # Fallback to JSON if yaml not available
                logger.info("PyYAML not available, using default configuration")
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced repositories."""
        return {
            'scoring': {
                'weights': {
                    'advanced': {
                        'wsjf': 0.5,
                        'ice': 0.1,
                        'technicalDebt': 0.3,
                        'security': 0.1
                    }
                },
                'thresholds': {
                    'minScore': 15,
                    'maxRisk': 0.7,
                    'securityBoost': 2.0
                }
            },
            'repository': {
                'maturity': 'advanced'
            }
        }
    
    async def discover_value_items(self) -> List[ValueItem]:
        """Discover value items from multiple sources."""
        logger.info("Starting comprehensive value discovery...")
        
        items = []
        
        # Parallel discovery from multiple sources
        discovery_tasks = [
            self._discover_from_git_history(),
            self._discover_from_static_analysis(),
            self._discover_from_security_scans(),
            self._discover_from_performance_analysis(),
            self._discover_from_technical_debt(),
            self._discover_modernization_opportunities()
        ]
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Discovery task failed: {result}")
        
        # Score and prioritize items
        scored_items = [self._calculate_composite_score(item) for item in items]
        
        # Remove duplicates and sort by score
        unique_items = self._deduplicate_items(scored_items)
        sorted_items = sorted(unique_items, key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"Discovered {len(sorted_items)} unique value items")
        return sorted_items
    
    async def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover items from Git history analysis."""
        items = []
        
        try:
            # Find TODO/FIXME comments
            result = subprocess.run([
                'git', 'grep', '-n', '-i', 
                '-E', '(TODO|FIXME|HACK|XXX|DEPRECATED)',
                '--', '*.py', '*.swift'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path, line_num, content = parts
                            items.append(ValueItem(
                                id=f"debt-{hash(line) & 0x7fffffff}",
                                title=f"Resolve technical debt in {file_path}",
                                description=f"Line {line_num}: {content.strip()}",
                                category="technical_debt",
                                estimated_effort_hours=1.5,
                                files_affected=[file_path],
                                source="git_grep"
                            ))
            
        except Exception as e:
            logger.warning(f"Git history analysis failed: {e}")
        
        return items
    
    async def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static analysis tools."""
        items = []
        
        # MyPy analysis for type improvements
        try:
            result = subprocess.run([
                'mypy', 'src/', '--report', '/tmp/mypy-report',
                '--json-report', '/tmp/mypy.json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if Path('/tmp/mypy.json').exists():
                with open('/tmp/mypy.json') as f:
                    mypy_data = json.load(f)
                    
                for file_data in mypy_data.get('files', []):
                    errors = len(file_data.get('errors', []))
                    if errors > 0:
                        items.append(ValueItem(
                            id=f"types-{hash(file_data['path']) & 0x7fffffff}",
                            title=f"Improve type annotations in {file_data['path']}",
                            description=f"Found {errors} type-related issues",
                            category="code_quality",
                            estimated_effort_hours=errors * 0.3,
                            files_affected=[file_data['path']],
                            source="mypy"
                        ))
        except Exception as e:
            logger.warning(f"MyPy analysis failed: {e}")
        
        return items
    
    async def _discover_from_security_scans(self) -> List[ValueItem]:
        """Discover security-related value items."""
        items = []
        
        # Bandit security scanning
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json', '-o', '/tmp/bandit.json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if Path('/tmp/bandit.json').exists():
                with open('/tmp/bandit.json') as f:
                    bandit_data = json.load(f)
                    
                for issue in bandit_data.get('results', []):
                    severity_multiplier = {
                        'HIGH': 3.0,
                        'MEDIUM': 2.0,
                        'LOW': 1.0
                    }.get(issue.get('issue_severity', 'LOW'), 1.0)
                    
                    items.append(ValueItem(
                        id=f"security-{hash(str(issue)) & 0x7fffffff}",
                        title=f"Fix {issue.get('issue_severity', 'MEDIUM')} security issue",
                        description=f"{issue.get('issue_text', 'Security issue')} in {issue.get('filename', 'unknown')}",
                        category="security",
                        estimated_effort_hours=1.0 * severity_multiplier,
                        files_affected=[issue.get('filename', '')],
                        risk_level=issue.get('issue_severity', 'MEDIUM').lower(),
                        source="bandit"
                    ))
        except Exception as e:
            logger.warning(f"Bandit analysis failed: {e}")
        
        # Safety dependency scanning
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data.get('vulnerabilities', []):
                    items.append(ValueItem(
                        id=f"vuln-{hash(str(vuln)) & 0x7fffffff}",
                        title=f"Update vulnerable dependency: {vuln.get('package_name', 'unknown')}",
                        description=f"Vulnerability: {vuln.get('advisory', 'Unknown vulnerability')}",
                        category="security",
                        estimated_effort_hours=0.5,
                        source="safety"
                    ))
        except Exception as e:
            logger.warning(f"Safety analysis failed: {e}")
        
        return items
    
    async def _discover_from_performance_analysis(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Check for large files that might need optimization
        try:
            for py_file in self.repo_path.rglob("*.py"):
                if py_file.stat().st_size > 10000:  # Files > 10KB
                    with open(py_file) as f:
                        lines = len(f.readlines())
                        if lines > 500:
                            items.append(ValueItem(
                                id=f"perf-{hash(str(py_file)) & 0x7fffffff}",
                                title=f"Optimize large module: {py_file.name}",
                                description=f"Large file with {lines} lines, consider refactoring",
                                category="performance",
                                estimated_effort_hours=4.0,
                                files_affected=[str(py_file.relative_to(self.repo_path))],
                                source="file_analysis"
                            ))
        except Exception as e:
            logger.warning(f"Performance analysis failed: {e}")
        
        return items
    
    async def _discover_from_technical_debt(self) -> List[ValueItem]:
        """Discover technical debt items."""
        items = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:5]:  # Limit to top 5 outdated packages
                    items.append(ValueItem(
                        id=f"dep-{hash(pkg['name']) & 0x7fffffff}",
                        title=f"Update dependency: {pkg['name']}",
                        description=f"Update from {pkg['version']} to {pkg['latest_version']}",
                        category="maintenance",
                        estimated_effort_hours=0.5,
                        files_affected=["requirements.txt", "pyproject.toml"],
                        source="pip_outdated"
                    ))
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
        
        return items
    
    async def _discover_modernization_opportunities(self) -> List[ValueItem]:
        """Discover modernization and enhancement opportunities."""
        items = []
        
        # Check for missing pre-commit hooks activation
        if not (self.repo_path / ".pre-commit-config.yaml").exists():
            items.append(ValueItem(
                id="precommit-setup",
                title="Set up pre-commit hooks",
                description="Configure and activate pre-commit hooks for code quality",
                category="automation",
                estimated_effort_hours=1.0,
                source="structure_analysis"
            ))
        
        # Check for missing GitHub Actions
        github_dir = self.repo_path / ".github" / "workflows"
        if not github_dir.exists() or not list(github_dir.glob("*.yml")):
            items.append(ValueItem(
                id="ci-cd-setup",
                title="Implement CI/CD workflows",
                description="Set up automated testing, security scanning, and deployment",
                category="automation",
                estimated_effort_hours=6.0,
                source="structure_analysis"
            ))
        
        # Check for missing documentation
        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            api_docs = docs_dir / "api"
            if not api_docs.exists():
                items.append(ValueItem(
                    id="api-docs-generation",
                    title="Set up automated API documentation generation",
                    description="Generate API docs from docstrings automatically",
                    category="documentation",
                    estimated_effort_hours=3.0,
                    source="structure_analysis"
                ))
        
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> ValueItem:
        """Calculate composite value score for an item."""
        weights = self.config['scoring']['weights']['advanced']
        
        # WSJF Calculation (Weighted Shortest Job First)
        user_value = self._score_user_business_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = max(item.estimated_effort_hours, 0.1)  # Avoid division by zero
        
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE Calculation (Impact, Confidence, Ease)
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = self._score_ease(item)
        
        item.ice_score = impact * confidence * ease
        
        # Technical Debt Score
        item.technical_debt_score = self._score_technical_debt(item)
        
        # Composite Score with adaptive weights
        item.composite_score = (
            weights['wsjf'] * self._normalize_score(item.wsjf_score, 0, 50) +
            weights['ice'] * self._normalize_score(item.ice_score, 0, 1000) +
            weights['technicalDebt'] * self._normalize_score(item.technical_debt_score, 0, 100) +
            weights.get('security', 0) * (2.0 if item.category == 'security' else 1.0)
        )
        
        # Apply boosts for high-priority categories
        if item.category == 'security':
            item.composite_score *= self.config['scoring']['thresholds']['securityBoost']
        elif item.risk_level == 'high':
            item.composite_score *= 1.5
        
        return item
    
    def _score_user_business_value(self, item: ValueItem) -> float:
        """Score user/business value impact."""
        category_scores = {
            'security': 9.0,
            'performance': 8.0,
            'automation': 7.0,
            'code_quality': 6.0,
            'technical_debt': 5.0,
            'documentation': 4.0,
            'maintenance': 3.0
        }
        return category_scores.get(item.category, 5.0)
    
    def _score_time_criticality(self, item: ValueItem) -> float:
        """Score time criticality."""
        if item.category == 'security':
            return 8.0
        elif item.risk_level == 'high':
            return 7.0
        elif item.category in ['automation', 'performance']:
            return 6.0
        return 4.0
    
    def _score_risk_reduction(self, item: ValueItem) -> float:
        """Score risk reduction benefit."""
        if item.category == 'security':
            return 9.0
        elif item.category == 'technical_debt':
            return 7.0
        elif len(item.files_affected) > 5:
            return 6.0
        return 3.0
    
    def _score_opportunity_enablement(self, item: ValueItem) -> float:
        """Score opportunity enablement."""
        if item.category == 'automation':
            return 8.0
        elif item.category in ['performance', 'code_quality']:
            return 6.0
        return 4.0
    
    def _score_impact(self, item: ValueItem) -> int:
        """Score impact (1-10 scale)."""
        category_impacts = {
            'security': 10,
            'performance': 9,
            'automation': 8,
            'code_quality': 7,
            'technical_debt': 6,
            'documentation': 5,
            'maintenance': 4
        }
        return category_impacts.get(item.category, 5)
    
    def _score_confidence(self, item: ValueItem) -> int:
        """Score execution confidence (1-10 scale)."""
        if item.estimated_effort_hours <= 2.0:
            return 9
        elif item.estimated_effort_hours <= 4.0:
            return 7
        elif item.estimated_effort_hours <= 8.0:
            return 6
        else:
            return 4
    
    def _score_ease(self, item: ValueItem) -> int:
        """Score implementation ease (1-10 scale)."""
        if item.estimated_effort_hours <= 1.0:
            return 10
        elif item.estimated_effort_hours <= 3.0:
            return 8
        elif len(item.dependencies) == 0:
            return 7
        else:
            return 5
    
    def _score_technical_debt(self, item: ValueItem) -> float:
        """Score technical debt impact."""
        if item.category == 'technical_debt':
            return 80.0
        elif item.category == 'security':
            return 60.0
        elif item.category == 'performance':
            return 50.0
        return 20.0
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val == min_val:
            return 50.0
        return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))
    
    def _deduplicate_items(self, items: List[ValueItem]) -> List[ValueItem]:
        """Remove duplicate items based on similarity."""
        unique_items = []
        seen_titles = set()
        
        for item in items:
            # Simple deduplication based on title similarity
            similar_exists = any(
                abs(len(item.title) - len(existing)) < 5 and 
                item.title[:20] == existing[:20]
                for existing in seen_titles
            )
            
            if not similar_exists:
                unique_items.append(item)
                seen_titles.add(item.title)
        
        return unique_items
    
    async def select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best value item to execute."""
        if not items:
            return None
        
        # Filter items based on thresholds
        min_score = self.config['scoring']['thresholds']['minScore']
        max_risk = self.config['scoring']['thresholds']['maxRisk']
        
        eligible_items = [
            item for item in items
            if item.composite_score >= min_score and
            (item.risk_level != 'high' or max_risk >= 0.8)
        ]
        
        if not eligible_items:
            logger.info("No eligible items found, selecting top item regardless")
            return items[0] if items else None
        
        # Return highest scoring eligible item
        return eligible_items[0]
    
    def generate_backlog_markdown(self, items: List[ValueItem]) -> str:
        """Generate markdown backlog report."""
        now = datetime.now(timezone.utc)
        
        markdown = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Repository: FastVLM On-Device Kit
Maturity Level: Advanced (85%+)

## 🎯 Next Best Value Item
"""
        
        if items:
            next_item = items[0]
            markdown += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.0f} | **Tech Debt**: {next_item.technical_debt_score:.0f}
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Estimated Effort**: {next_item.estimated_effort_hours:.1f} hours
- **Risk Level**: {next_item.risk_level.title()}
- **Description**: {next_item.description}

"""
        
        markdown += f"""## 📋 Top {min(10, len(items))} Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Risk |
|------|-----|--------|---------|----------|------------|------|
"""
        
        for i, item in enumerate(items[:10], 1):
            category = item.category.replace('_', ' ').title()
            markdown += f"| {i} | {item.id.upper()} | {item.title[:40]}{'...' if len(item.title) > 40 else ''} | {item.composite_score:.1f} | {category} | {item.estimated_effort_hours:.1f} | {item.risk_level.title()} |\n"
        
        markdown += f"""

## 📈 Value Metrics
- **Total Items in Backlog**: {len(items)}
- **High Priority Items** (Score > 50): {len([i for i in items if i.composite_score > 50])}
- **Security Items**: {len([i for i in items if i.category == 'security'])}
- **Technical Debt Items**: {len([i for i in items if i.category == 'technical_debt'])}
- **Automation Opportunities**: {len([i for i in items if i.category == 'automation'])}

## 🔄 Discovery Sources
- **Static Analysis**: MyPy, Bandit, Safety
- **Git History**: TODO/FIXME analysis
- **Structure Analysis**: Missing components
- **Performance Analysis**: Large file detection
- **Dependency Analysis**: Outdated packages

## 📊 Category Breakdown
"""
        
        categories = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(items)) * 100
            markdown += f"- **{category.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        markdown += f"""

## 🏆 Recommended Next Actions
1. **Immediate**: Address highest-scoring security items
2. **Short-term**: Implement automation improvements
3. **Medium-term**: Reduce technical debt systematically
4. **Long-term**: Enhance performance and documentation

---
*Generated by Terragon Autonomous SDLC Engine*
*Next discovery cycle: {(now.timestamp() + 3600):.0f} (1 hour)*
"""
        
        return markdown
    
    async def save_metrics(self, items: List[ValueItem]) -> None:
        """Save value metrics to JSON file."""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'repository': {
                'name': 'FastVLM On-Device Kit',
                'maturity': self.config['repository']['maturity'],
                'path': str(self.repo_path)
            },
            'discovery': {
                'total_items': len(items),
                'avg_score': sum(item.composite_score for item in items) / len(items) if items else 0,
                'categories': {}
            },
            'top_items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'category': item.category,
                    'score': round(item.composite_score, 2),
                    'effort_hours': item.estimated_effort_hours,
                    'source': item.source
                }
                for item in items[:5]
            ]
        }
        
        # Category breakdown
        for item in items:
            category = item.category
            if category not in metrics['discovery']['categories']:
                metrics['discovery']['categories'][category] = {
                    'count': 0,
                    'avg_score': 0,
                    'total_effort': 0
                }
            
            metrics['discovery']['categories'][category]['count'] += 1
            metrics['discovery']['categories'][category]['total_effort'] += item.estimated_effort_hours
        
        # Calculate averages
        for category_data in metrics['discovery']['categories'].values():
            count = category_data['count']
            if count > 0:
                category_items = [i for i in items if i.category == category]
                category_data['avg_score'] = sum(i.composite_score for i in category_items) / count
        
        # Ensure directory exists
        self.metrics_path.parent.mkdir(exist_ok=True)
        
        # Save metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {self.metrics_path}")
    
    async def run_discovery_cycle(self) -> None:
        """Run a complete discovery and prioritization cycle."""
        logger.info("🚀 Starting autonomous value discovery cycle...")
        
        try:
            # Discover value items
            items = await self.discover_value_items()
            
            if not items:
                logger.info("No value items discovered, creating maintenance tasks...")
                items = self._generate_maintenance_tasks()
            
            # Select next best value item
            next_item = await self.select_next_best_value(items)
            
            if next_item:
                logger.info(f"🎯 Next best value: [{next_item.id}] {next_item.title} (Score: {next_item.composite_score:.1f})")
            
            # Generate and save backlog
            backlog_md = self.generate_backlog_markdown(items)
            with open(self.backlog_path, 'w') as f:
                f.write(backlog_md)
            
            # Save metrics
            await self.save_metrics(items)
            
            logger.info(f"✅ Discovery cycle complete. Found {len(items)} items, backlog updated.")
            
        except Exception as e:
            logger.error(f"❌ Discovery cycle failed: {e}")
            raise
    
    def _generate_maintenance_tasks(self) -> List[ValueItem]:
        """Generate maintenance tasks when no issues are found."""
        return [
            ValueItem(
                id="maintenance-deps",
                title="Review and update dependencies",
                description="Systematic review of all dependencies for updates and security patches",
                category="maintenance",
                estimated_effort_hours=2.0,
                source="maintenance_generator"
            ),
            ValueItem(
                id="maintenance-docs",
                title="Review and update documentation",
                description="Update API documentation and ensure all features are documented",
                category="documentation",
                estimated_effort_hours=3.0,
                source="maintenance_generator"
            ),
            ValueItem(
                id="maintenance-tests",
                title="Enhance test coverage",
                description="Add tests for edge cases and improve coverage metrics",
                category="code_quality",
                estimated_effort_hours=4.0,
                source="maintenance_generator"
            )
        ]


async def main():
    """Main entry point for autonomous value discovery."""
    engine = ValueDiscoveryEngine()
    
    try:
        await engine.run_discovery_cycle()
        print("🎉 Autonomous SDLC value discovery completed successfully!")
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)