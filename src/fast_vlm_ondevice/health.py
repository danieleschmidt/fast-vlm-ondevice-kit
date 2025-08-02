"""
Health check and system monitoring for FastVLM On-Device Kit.
Provides comprehensive health checks for dependencies, resources, and services.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class HealthStatus:
    """Health check status information."""
    status: str  # healthy, degraded, unhealthy
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class HealthChecker:
    """Comprehensive health checker for FastVLM components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self) -> HealthStatus:
        """Check availability and versions of critical dependencies."""
        try:
            details = {}
            issues = []
            
            # PyTorch check
            try:
                import torch
                details['pytorch'] = {
                    'available': True,
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                }
                
                # Check if PyTorch can create tensors
                test_tensor = torch.randn(10, 10)
                details['pytorch']['tensor_creation'] = True
                
            except Exception as e:
                issues.append(f"PyTorch: {str(e)}")
                details['pytorch'] = {'available': False, 'error': str(e)}
            
            # Core ML Tools check
            try:
                import coremltools
                details['coremltools'] = {
                    'available': True,
                    'version': coremltools.__version__
                }
                
                # Test basic Core ML functionality
                import coremltools.models.datatypes as dt
                details['coremltools']['datatypes_available'] = True
                
            except Exception as e:
                issues.append(f"Core ML Tools: {str(e)}")
                details['coremltools'] = {'available': False, 'error': str(e)}
            
            # Transformers check
            try:
                import transformers
                details['transformers'] = {
                    'available': True,
                    'version': transformers.__version__
                }
            except Exception as e:
                issues.append(f"Transformers: {str(e)}")
                details['transformers'] = {'available': False, 'error': str(e)}
            
            # PIL/Pillow check
            try:
                from PIL import Image
                details['pillow'] = {
                    'available': True,
                    'version': Image.__version__ if hasattr(Image, '__version__') else 'unknown'
                }
                
                # Test image creation
                test_image = Image.new('RGB', (100, 100))
                details['pillow']['image_creation'] = True
                
            except Exception as e:
                issues.append(f"Pillow: {str(e)}")
                details['pillow'] = {'available': False, 'error': str(e)}
            
            # NumPy check
            try:
                import numpy as np
                details['numpy'] = {
                    'available': True,
                    'version': np.__version__
                }
                
                # Test basic operations
                test_array = np.random.rand(10, 10)
                details['numpy']['array_creation'] = True
                
            except Exception as e:
                issues.append(f"NumPy: {str(e)}")
                details['numpy'] = {'available': False, 'error': str(e)}
            
            # Determine overall status
            if not issues:
                status = "healthy"
                message = "All dependencies are available and functional"
            elif len(issues) <= 1:
                status = "degraded"
                message = f"Some dependencies have issues: {'; '.join(issues)}"
            else:
                status = "unhealthy"
                message = f"Multiple dependency issues: {'; '.join(issues)}"
            
            return HealthStatus(status=status, message=message, details=details)
            
        except Exception as e:
            self.logger.error(f"Error checking dependencies: {e}")
            return HealthStatus(
                status="unhealthy",
                message=f"Failed to check dependencies: {str(e)}"
            )
    
    def check_system_resources(self) -> HealthStatus:
        """Check system resource availability and usage."""
        try:
            details = {}
            warnings = []
            
            # Memory check
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                details['memory'] = {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_percent': memory.percent,
                    'free_percent': round(100 - memory.percent, 1)
                }
                
                if memory.percent > 90:
                    warnings.append("Memory usage above 90%")
                elif memory.percent > 80:
                    warnings.append("Memory usage above 80%")
                
            except ImportError:
                warnings.append("psutil not available - cannot check memory")
                details['memory'] = {'available': False}
            
            # Disk space check
            try:
                import psutil
                disk = psutil.disk_usage('/')
                
                details['disk'] = {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_percent': round((disk.used / disk.total) * 100, 1)
                }
                
                used_percent = (disk.used / disk.total) * 100
                if used_percent > 90:
                    warnings.append("Disk usage above 90%")
                elif used_percent > 85:
                    warnings.append("Disk usage above 85%")
                    
            except (ImportError, OSError):
                warnings.append("Cannot check disk usage")
                details['disk'] = {'available': False}
            
            # CPU check
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                
                details['cpu'] = {
                    'usage_percent': cpu_percent,
                    'core_count': cpu_count,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
                
                if cpu_percent > 90:
                    warnings.append("CPU usage above 90%")
                elif cpu_percent > 80:
                    warnings.append("CPU usage above 80%")
                    
            except ImportError:
                warnings.append("Cannot check CPU usage")
                details['cpu'] = {'available': False}
            
            # Determine status
            if not warnings:
                status = "healthy"
                message = "System resources are within normal limits"
            elif len(warnings) <= 1:
                status = "degraded"
                message = f"Resource warnings: {'; '.join(warnings)}"
            else:
                status = "unhealthy"
                message = f"Multiple resource issues: {'; '.join(warnings)}"
            
            return HealthStatus(status=status, message=message, details=details)
            
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return HealthStatus(
                status="unhealthy",
                message=f"Failed to check system resources: {str(e)}"
            )
    
    def check_model_files(self, model_dir: str = "models") -> HealthStatus:
        """Check availability and integrity of model files."""
        try:
            import os
            from pathlib import Path
            
            details = {}
            issues = []
            
            model_path = Path(model_dir)
            
            if not model_path.exists():
                return HealthStatus(
                    status="degraded",
                    message=f"Model directory {model_dir} does not exist",
                    details={'model_directory': {'exists': False}}
                )
            
            # Check for model files
            model_files = list(model_path.glob("*.mlpackage"))
            checkpoint_files = list(model_path.glob("*.pth"))
            
            details['model_files'] = {
                'directory_exists': True,
                'mlpackage_count': len(model_files),
                'checkpoint_count': len(checkpoint_files),
                'total_size_mb': 0
            }
            
            # Calculate total size
            total_size = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            details['model_files']['total_size_mb'] = round(total_size / (1024**2), 2)
            
            # Check specific model variants
            expected_models = ['FastVLM-Tiny.mlpackage', 'FastVLM-Base.mlpackage']
            available_models = []
            
            for model_name in expected_models:
                model_file = model_path / model_name
                if model_file.exists():
                    available_models.append(model_name)
                    # Basic integrity check - verify it's a directory
                    if not model_file.is_dir():
                        issues.append(f"{model_name} is not a valid mlpackage directory")
                else:
                    issues.append(f"Missing model: {model_name}")
            
            details['model_files']['available_models'] = available_models
            
            # Determine status
            if not issues and len(available_models) >= 1:
                status = "healthy"
                message = f"Found {len(available_models)} model(s)"
            elif available_models:
                status = "degraded"
                message = f"Some models missing: {'; '.join(issues)}"
            else:
                status = "unhealthy"
                message = "No model files found"
            
            return HealthStatus(status=status, message=message, details=details)
            
        except Exception as e:
            self.logger.error(f"Error checking model files: {e}")
            return HealthStatus(
                status="unhealthy",
                message=f"Failed to check model files: {str(e)}"
            )
    
    def check_external_services(self) -> HealthStatus:
        """Check connectivity to external services if configured."""
        try:
            details = {}
            issues = []
            
            # This is a placeholder for external service checks
            # In a real implementation, you might check:
            # - Database connectivity
            # - API endpoints
            # - Cloud storage access
            # - Monitoring services
            
            details['external_services'] = {
                'checked': True,
                'note': 'FastVLM runs on-device with no external dependencies'
            }
            
            return HealthStatus(
                status="healthy",
                message="No external service dependencies",
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Error checking external services: {e}")
            return HealthStatus(
                status="unhealthy",
                message=f"Failed to check external services: {str(e)}"
            )
    
    def full_health_check(self, model_dir: str = "models") -> Dict[str, Any]:
        """Perform comprehensive health check of all components."""
        self.logger.info("Starting comprehensive health check")
        
        checks = {
            'dependencies': self.check_dependencies(),
            'system_resources': self.check_system_resources(),
            'model_files': self.check_model_files(model_dir),
            'external_services': self.check_external_services()
        }
        
        # Determine overall status
        statuses = [check.status for check in checks.values()]
        
        if all(status == "healthy" for status in statuses):
            overall_status = "healthy"
            overall_message = "All components are healthy"
        elif "unhealthy" in statuses:
            overall_status = "unhealthy"
            unhealthy_components = [name for name, check in checks.items() if check.status == "unhealthy"]
            overall_message = f"Unhealthy components: {', '.join(unhealthy_components)}"
        else:
            overall_status = "degraded"
            degraded_components = [name for name, check in checks.items() if check.status == "degraded"]
            overall_message = f"Degraded components: {', '.join(degraded_components)}"
        
        result = {
            'overall_status': overall_status,
            'overall_message': overall_message,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {name: asdict(check) for name, check in checks.items()}
        }
        
        self.logger.info(f"Health check completed: {overall_status}")
        return result
    
    def export_health_metrics(self, health_data: Dict[str, Any]) -> str:
        """Export health check results as Prometheus metrics."""
        metrics = []
        
        # Overall health metric
        status_value = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}
        overall_value = status_value.get(health_data['overall_status'], 0)
        metrics.append(f"fastvm_health_status {overall_value}")
        
        # Component health metrics
        for component, check_data in health_data['checks'].items():
            component_value = status_value.get(check_data['status'], 0)
            metrics.append(f'fastvm_component_health{{component="{component}"}} {component_value}')
        
        # System resource metrics
        if 'system_resources' in health_data['checks']:
            sys_details = health_data['checks']['system_resources'].get('details', {})
            
            if 'memory' in sys_details and sys_details['memory'].get('available', True):
                memory_percent = sys_details['memory']['used_percent']
                metrics.append(f"fastvm_memory_usage_percent {memory_percent}")
            
            if 'cpu' in sys_details and sys_details['cpu'].get('available', True):
                cpu_percent = sys_details['cpu']['usage_percent']
                metrics.append(f"fastvm_cpu_usage_percent {cpu_percent}")
            
            if 'disk' in sys_details and sys_details['disk'].get('available', True):
                disk_percent = sys_details['disk']['used_percent']
                metrics.append(f"fastvm_disk_usage_percent {disk_percent}")
        
        return "\n".join(metrics)


def create_health_endpoint(health_checker: Optional[HealthChecker] = None):
    """Create a simple HTTP health endpoint using built-in modules."""
    if health_checker is None:
        health_checker = HealthChecker()
    
    def health_handler():
        """Handle health check requests."""
        try:
            health_data = health_checker.full_health_check()
            
            # Return appropriate HTTP status code
            if health_data['overall_status'] == 'healthy':
                status_code = 200
            elif health_data['overall_status'] == 'degraded':
                status_code = 200  # Still serving traffic
            else:
                status_code = 503  # Service unavailable
            
            return {
                'status_code': status_code,
                'content_type': 'application/json',
                'body': json.dumps(health_data, indent=2)
            }
            
        except Exception as e:
            error_response = {
                'overall_status': 'unhealthy',
                'overall_message': f'Health check failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return {
                'status_code': 503,
                'content_type': 'application/json',
                'body': json.dumps(error_response, indent=2)
            }
    
    return health_handler


# Global health checker instance
_global_health_checker = HealthChecker()

def get_global_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _global_health_checker