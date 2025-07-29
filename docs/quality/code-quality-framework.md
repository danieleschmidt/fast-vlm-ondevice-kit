# Code Quality Framework

## Overview

Comprehensive code quality framework for Fast VLM On-Device Kit, implementing automated quality gates, metrics collection, and continuous quality improvement for MATURING SDLC environments.

## Required GitHub Actions Workflow

Create `.github/workflows/code-quality.yml`:

```yaml
name: Code Quality Analysis
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6AM

jobs:
  quality-analysis:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for SonarQube
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install Quality Tools
        run: |
          pip install -r requirements-dev.txt
          pip install radon xenon bandit jscpd coverage
          npm install -g jscpd  # For duplication detection
          
      - name: Run Quality Metrics Collection
        run: |
          python scripts/quality_metrics.py --fail-on-gate-failure
          
      - name: SonarQube Scan
        uses: sonarqube-quality-gate-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
          
      - name: CodeClimate Analysis
        uses: paambaati/codeclimate-action@v5.0.0
        env:
          CC_TEST_REPORTER_ID: ${{ secrets.CC_TEST_REPORTER_ID }}
        with:
          coverageCommand: coverage xml
          
      - name: Upload Quality Reports
        uses: actions/upload-artifact@v4
        with:
          name: quality-reports
          path: quality-reports/
          
      - name: Comment PR with Quality Results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const path = 'quality-reports/quality-report.md';
            
            if (fs.existsSync(path)) {
              const report = fs.readFileSync(path, 'utf8');
              
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: `## 📊 Code Quality Analysis\n\n${report}`
              });
            }
            
      - name: Quality Gate Status Check
        run: |
          if [ -f "quality-reports/quality-gates.json" ]; then
            python -c "
            import json
            with open('quality-reports/quality-gates.json') as f:
              gates = json.load(f)
            failed = [g for g in gates if g['status'] == 'FAIL']
            if failed:
              print(f'Quality gates failed: {len(failed)}')
              exit(1)
            else:
              print('All quality gates passed')
            "
          fi
```

## Quality Metrics Framework

### 1. Code Complexity Metrics

```python
# Complexity analysis configuration
complexity_rules = {
    'cyclomatic_complexity': {
        'threshold': 10,
        'measurement': 'average per function',
        'tool': 'radon'
    },
    'cognitive_complexity': {
        'threshold': 15,
        'measurement': 'average per function',
        'tool': 'xenon'
    },
    'nesting_depth': {
        'threshold': 4,
        'measurement': 'maximum levels',
        'tool': 'radon'
    },
    'maintainability_index': {
        'threshold': 60,
        'measurement': 'weighted average',
        'tool': 'radon'
    }
}
```

### 2. Test Coverage Analysis

```python
# Coverage requirements by component
coverage_targets = {
    'core_modules': {
        'line_coverage': 90,
        'branch_coverage': 85,
        'function_coverage': 95
    },
    'utility_modules': {
        'line_coverage': 80,
        'branch_coverage': 75,
        'function_coverage': 85
    },
    'integration_tests': {
        'line_coverage': 70,
        'branch_coverage': 65,
        'function_coverage': 80
    }
}
```

### 3. Code Duplication Detection

```yaml
# JSCPD configuration for duplication detection
duplication_config:
  threshold: 3  # minimum duplicate lines
  min-tokens: 50  # minimum tokens for detection
  reporters:
    - json
    - html
  ignore:
    - "**/__pycache__/**"
    - "**/node_modules/**"
    - "**/coverage/**"
  languages:
    - python
    - swift
    - dockerfile
```

## Quality Gates Implementation

### 1. Automated Quality Gates

```python
class QualityGate:
    def __init__(self, name: str, threshold: float, metric_type: str):
        self.name = name
        self.threshold = threshold
        self.metric_type = metric_type
    
    def evaluate(self, actual_value: float) -> QualityGateResult:
        if self.metric_type == "minimum":
            passed = actual_value >= self.threshold
        else:  # maximum
            passed = actual_value <= self.threshold
            
        return QualityGateResult(
            gate_name=self.name,
            status="PASS" if passed else "FAIL",
            actual_value=actual_value,
            threshold_value=self.threshold
        )

# Define quality gates
QUALITY_GATES = [
    QualityGate("test_coverage", 85.0, "minimum"),
    QualityGate("cyclomatic_complexity", 10.0, "maximum"),
    QualityGate("code_duplication", 3.0, "maximum"),
    QualityGate("maintainability_index", 60.0, "minimum"),
    QualityGate("security_hotspots", 0.0, "maximum")
]
```

### 2. Progressive Quality Improvement

```python
# Quality improvement tracking
class QualityTrendAnalyzer:
    def analyze_trends(self, historical_data: List[Dict]) -> Dict:
        """Analyze quality trends over time"""
        trends = {}
        
        for metric in ['coverage', 'complexity', 'duplication']:
            values = [d[metric] for d in historical_data]
            
            if len(values) >= 2:
                # Calculate trend direction
                recent_avg = sum(values[-3:]) / min(3, len(values))
                older_avg = sum(values[:-3]) / max(1, len(values) - 3)
                
                if metric == 'coverage':
                    # Higher is better for coverage
                    trend = "improving" if recent_avg > older_avg else "declining"
                else:
                    # Lower is better for complexity and duplication
                    trend = "improving" if recent_avg < older_avg else "declining"
                
                trends[metric] = {
                    'direction': trend,
                    'recent_average': recent_avg,
                    'change_percentage': ((recent_avg - older_avg) / older_avg) * 100
                }
        
        return trends
```

## Language-Specific Quality Rules

### 1. Python Quality Standards

```yaml
# Python-specific quality configuration
python_quality:
  linting:
    flake8:
      max_line_length: 88
      max_complexity: 10
      ignore: ["E203", "W503"]
      
    pylint:
      minimum_score: 8.0
      disable: ["too-few-public-methods"]
      
    mypy:
      strict_mode: true
      ignore_missing_imports: false
      
  formatting:
    black:
      line_length: 88
      target_version: ["py310"]
      
    isort:
      profile: "black"
      multi_line_output: 3
      
  security:
    bandit:
      skip_tests: ["B101"]  # Skip assert_used
      confidence_level: "medium"
```

### 2. Swift Quality Standards

```yaml
# Swift-specific quality configuration
swift_quality:
  linting:
    swiftlint:
      included:
        - ios/Sources
        - ios/Tests
      excluded:
        - ios/build
        - ios/.build
      rules:
        line_length:
          warning: 120
          error: 150
        function_body_length:
          warning: 50
          error: 100
        cyclomatic_complexity:
          warning: 10
          error: 20
          
  formatting:
    swiftformat:
      rules:
        - indent: 2
        - linebreaks: lf
        - semicolons: never
```

## Integration with Development Workflow

### 1. Pre-commit Quality Checks

```yaml
# Enhanced pre-commit configuration
repos:
  - repo: local
    hooks:
      - id: quality-check
        name: Quality Check
        entry: python scripts/quality_metrics.py
        language: system
        pass_filenames: false
        always_run: true
        
      - id: complexity-check
        name: Complexity Check
        entry: xenon --max-absolute B --max-modules A --max-average A src/
        language: system
        files: \.py$
        
      - id: duplication-check
        name: Duplication Check
        entry: jscpd src/ --threshold 3
        language: system
        files: \.py$
```

### 2. IDE Integration

```json
// VS Code quality settings
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.banditEnabled": true,
  
  "python.formatting.provider": "black",
  "python.sortImports.provider": "isort",
  
  "files.associations": {
    "*.yml": "yaml"
  },
  
  "yaml.schemas": {
    "./.github/code-quality-config.yml": "file://quality-schema.json"
  }
}
```

## Quality Reporting and Visualization

### 1. Quality Dashboard

```python
# Generate quality dashboard
class QualityDashboard:
    def generate_dashboard(self, metrics_history: List[Dict]) -> str:
        """Generate HTML quality dashboard"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Quality Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Fast VLM Quality Dashboard</h1>
            
            <div id="coverage-chart"></div>
            <div id="complexity-chart"></div>
            <div id="duplication-chart"></div>
            
            <script>
            // Coverage trend chart
            var coverageData = {coverage_data};
            Plotly.newPlot('coverage-chart', coverageData);
            
            // Complexity trend chart
            var complexityData = {complexity_data};
            Plotly.newPlot('complexity-chart', complexityData);
            
            // Duplication trend chart
            var duplicationData = {duplication_data};
            Plotly.newPlot('duplication-chart', duplicationData);
            </script>
        </body>
        </html>
        """
        
        # Generate chart data from metrics history
        coverage_data = self._generate_trend_chart(metrics_history, 'test_coverage')
        complexity_data = self._generate_trend_chart(metrics_history, 'cyclomatic_complexity')
        duplication_data = self._generate_trend_chart(metrics_history, 'code_duplication')
        
        return html_template.format(
            coverage_data=json.dumps(coverage_data),
            complexity_data=json.dumps(complexity_data),
            duplication_data=json.dumps(duplication_data)
        )
```

### 2. Quality Alerts and Notifications

```python
# Quality alert system
class QualityAlertSystem:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_quality_alert(self, gate_failures: List[QualityGateResult]):
        """Send alert for quality gate failures"""
        if not gate_failures:
            return
        
        message = {
            "text": "🚨 Code Quality Alert",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{len(gate_failures)} quality gates failed*"
                    }
                }
            ]
        }
        
        for failure in gate_failures:
            message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"❌ *{failure.gate_name}*: {failure.message}"
                }
            })
        
        requests.post(self.webhook_url, json=message)
```

## Technical Debt Management

### 1. Technical Debt Tracking

```python
# Technical debt quantification
class TechnicalDebtAnalyzer:
    def calculate_debt_hours(self, metrics: Dict) -> float:
        """Calculate technical debt in hours"""
        debt_factors = {
            'complexity_debt': metrics['cyclomatic_complexity'] * 0.5,  # hours per complexity point
            'coverage_debt': max(0, 85 - metrics['test_coverage']) * 0.25,  # hours per missing coverage %
            'duplication_debt': metrics['code_duplication'] * 1.0,  # hours per duplicate %
            'security_debt': metrics['security_issues']['high'] * 4.0 + metrics['security_issues']['medium'] * 1.0
        }
        
        return sum(debt_factors.values())
    
    def prioritize_debt_reduction(self, debt_analysis: Dict) -> List[str]:
        """Prioritize technical debt reduction tasks"""
        priorities = []
        
        if debt_analysis['security_debt'] > 0:
            priorities.append("Address security vulnerabilities (HIGH PRIORITY)")
        
        if debt_analysis['coverage_debt'] > 10:
            priorities.append("Increase test coverage")
        
        if debt_analysis['complexity_debt'] > 20:
            priorities.append("Refactor complex functions")
        
        if debt_analysis['duplication_debt'] > 5:
            priorities.append("Remove code duplication")
        
        return priorities
```

### 2. Refactoring Recommendations

```python
# Automated refactoring suggestions
class RefactoringRecommendations:
    def analyze_refactoring_opportunities(self, source_dir: Path) -> List[Dict]:
        """Identify refactoring opportunities"""
        recommendations = []
        
        for py_file in source_dir.rglob("*.py"):
            tree = ast.parse(py_file.read_text())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_function_complexity(node)
                    
                    if complexity > 15:
                        recommendations.append({
                            'type': 'high_complexity',
                            'file': str(py_file),
                            'function': node.name,
                            'line': node.lineno,
                            'complexity': complexity,
                            'suggestion': 'Break into smaller functions'
                        })
                
                elif isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    
                    if method_count > 20:
                        recommendations.append({
                            'type': 'large_class',
                            'file': str(py_file),
                            'class': node.name,
                            'line': node.lineno,
                            'method_count': method_count,
                            'suggestion': 'Consider splitting into multiple classes'
                        })
        
        return recommendations
```

## Quality Culture and Best Practices

### 1. Code Review Quality Checklist

```markdown
## Code Review Quality Checklist

### Functionality
- [ ] Code fulfills requirements
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Performance considerations addressed

### Code Quality
- [ ] Code is readable and self-documenting
- [ ] Functions are single-purpose and focused
- [ ] Variable and function names are descriptive
- [ ] Complex logic is commented

### Testing
- [ ] New code has appropriate tests
- [ ] Tests cover edge cases
- [ ] Test coverage meets requirements
- [ ] Tests are maintainable

### Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation is present
- [ ] Security best practices followed
- [ ] Dependencies are secure and up-to-date

### Documentation
- [ ] Public APIs are documented
- [ ] Complex algorithms are explained
- [ ] README and docs are updated
- [ ] Change log is updated
```

### 2. Quality Training and Guidelines

```python
# Quality training materials generator
def generate_quality_guidelines():
    guidelines = {
        'python_best_practices': [
            "Use type hints for function parameters and return values",
            "Follow PEP 8 style guidelines",
            "Write descriptive docstrings for all public functions",
            "Use list comprehensions judiciously",
            "Prefer composition over inheritance",
            "Handle exceptions at the appropriate level"
        ],
        'swift_best_practices': [
            "Use guard statements for early returns",
            "Prefer value types over reference types when appropriate",
            "Use meaningful variable and function names",
            "Follow Swift API design guidelines",
            "Use optionals safely with proper unwrapping",
            "Organize code with extensions and protocols"
        ],
        'general_principles': [
            "Write code that tells a story",
            "Optimize for readability over cleverness",
            "Test behavior, not implementation",
            "Refactor continuously to reduce technical debt",
            "Document decisions and trade-offs",
            "Automate repetitive quality checks"
        ]
    }
    
    return guidelines
```

## References

- [Clean Code by Robert Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [Refactoring by Martin Fowler](https://refactoring.com/)
- [SonarQube Quality Model](https://docs.sonarqube.org/latest/user-guide/metric-definitions/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)