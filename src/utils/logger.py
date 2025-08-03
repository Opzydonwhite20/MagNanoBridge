"""
Logging utilities for MagNanoBridge simulation
Provides structured logging with performance monitoring
"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json


def setup_logger(name: str = 'magnano', 
                level: str = 'INFO',
                log_file: Optional[str] = None,
                console_output: bool = True,
                max_file_size: int = 10,  # MB
                backup_count: int = 5) -> logging.Logger:
    """
    Setup logging configuration for MagNanoBridge
    
    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        console_output: Whether to output to console
        max_file_size: Maximum log file size in MB
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class PerformanceLogger:
    """Logger for performance monitoring and profiling"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
        self.counters = {}
        self.performance_data = []
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time"""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        self.performance_data.append({
            'timer': name,
            'elapsed_time': elapsed,
            'timestamp': time.time()
        })
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def log_performance(self, operation: str, elapsed_time: float, **kwargs):
        """Log performance information"""
        perf_info = {
            'operation': operation,
            'elapsed_time': elapsed_time,
            'timestamp': time.time(),
            **kwargs
        }
        
        self.performance_data.append(perf_info)
        
        if elapsed_time > 1.0:  # Log slow operations
            self.logger.warning(f"Slow operation '{operation}': {elapsed_time:.3f}s")
        else:
            self.logger.debug(f"Performance '{operation}': {elapsed_time:.6f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance data"""
        if not self.performance_data:
            return {}
        
        # Group by operation
        operations = {}
        for entry in self.performance_data:
            op_name = entry.get('operation', entry.get('timer', 'unknown'))
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(entry['elapsed_time'])
        
        # Calculate statistics
        summary = {}
        for op_name, times in operations.items():
            summary[op_name] = {
                'count': len(times),
                'total_time': sum(times),
                'mean_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        summary['counters'] = self.counters.copy()
        summary['total_entries'] = len(self.performance_data)
        
        return summary
    
    def save_performance_data(self, filepath: str):
        """Save performance data to file"""
        data = {
            'performance_entries': self.performance_data,
            'summary': self.get_performance_summary(),
            'export_time': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Performance data saved to {filepath}")


class SimulationLogger:
    """Specialized logger for simulation events"""
    
    def __init__(self, base_logger: logging.Logger):
        self.logger = base_logger
        self.performance = PerformanceLogger(base_logger)
        self.simulation_events = []
        self.start_time = time.time()
    
    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation start event"""
        event = {
            'event': 'simulation_start',
            'timestamp': time.time(),
            'particle_count': config.get('particles', {}).get('count', 0),
            'max_steps': config.get('simulation', {}).get('max_steps', 0),
            'time_step': config.get('simulation', {}).get('dt', 0),
            'control_type': config.get('focus', {}).get('control_type', 'unknown')
        }
        
        self.simulation_events.append(event)
        self.logger.info(
            f"Simulation started: {event['particle_count']} particles, "
            f"{event['max_steps']} max steps, dt={event['time_step']:.2e}s"
        )
    
    def log_simulation_step(self, step: int, time_val: float, efficiency: float, **kwargs):
        """Log simulation step information"""
        if step % 1000 == 0:  # Log every 1000 steps
            self.logger.info(
                f"Step {step}: t={time_val:.6f}s, efficiency={efficiency:.3f}, "
                f"extras={kwargs}"
            )
    
    def log_simulation_end(self, results: Dict[str, Any]):
        """Log simulation end event"""
        total_time = time.time() - self.start_time
        
        event = {
            'event': 'simulation_end',
            'timestamp': time.time(),
            'total_wall_time': total_time,
            'total_sim_time': results.get('total_time', 0),
            'total_steps': results.get('total_steps', 0),
            'final_efficiency': results.get('focus_efficiency', 0),
            'bridge_formed': results.get('bridge_formed', False)
        }
        
        self.simulation_events.append(event)
        self.logger.info(
            f"Simulation completed: {event['total_steps']} steps in {total_time:.2f}s, "
            f"efficiency={event['final_efficiency']:.3f}, bridge={event['bridge_formed']}"
        )
    
    def log_control_update(self, efficiency: float, adjustments: Dict[str, float]):
        """Log control system updates"""
        self.logger.debug(
            f"Control update: efficiency={efficiency:.3f}, "
            f"adjustments={adjustments}"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context"""
        self.logger.error(f"Error in {context}: {type(error).__name__}: {error}")
    
    def log_warning(self, message: str, **kwargs):
        """Log warnings with optional data"""
        extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message}"
        if extra_info:
            full_message += f" ({extra_info})"
        self.logger.warning(full_message)
    
    def log_physics_stats(self, kinetic_energy: float, potential_energy: float, 
                         temperature: float, **kwargs):
        """Log physics statistics"""
        self.logger.debug(
            f"Physics: KE={kinetic_energy:.2e}J, PE={potential_energy:.2e}J, "
            f"T={temperature:.1f}K, extras={kwargs}"
        )
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation log summary"""
        return {
            'events': self.simulation_events,
            'performance_summary': self.performance.get_performance_summary(),
            'total_wall_time': time.time() - self.start_time,
            'event_count': len(self.simulation_events)
        }
    
    def save_simulation_log(self, filepath: str):
        """Save complete simulation log"""
        log_data = {
            'simulation_summary': self.get_simulation_summary(),
            'logger_name': self.logger.name,
            'log_level': self.logger.level,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Simulation log saved to {filepath}")


def create_simulation_logger(log_dir: str = './logs',
                           level: str = 'INFO',
                           session_name: str = None) -> SimulationLogger:
    """
    Create a complete simulation logger setup
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        session_name: Optional session name for log files
        
    Returns:
        Configured SimulationLogger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate session name if not provided
    if session_name is None:
        import datetime
        session_name = datetime.datetime.now().strftime("magnano_%Y%m%d_%H%M%S")
    
    # Setup main logger
    log_file = log_path / f"{session_name}.log"
    base_logger = setup_logger(
        name='magnano_simulation',
        level=level,
        log_file=str(log_file),
        console_output=True
    )
    
    # Create simulation logger
    sim_logger = SimulationLogger(base_logger)
    
    base_logger.info(f"Simulation logger initialized: {log_file}")
    return sim_logger


class LogAnalyzer:
    """Analyzer for simulation log files"""
    
    @staticmethod
    def analyze_log_file(log_path: str) -> Dict[str, Any]:
        """
        Analyze a simulation log file
        
        Args:
            log_path: Path to log file
            
        Returns:
            Analysis results
        """
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Try to parse as text log
            return LogAnalyzer._analyze_text_log(log_path)
        
        # Analyze structured log
        analysis = {
            'total_wall_time': log_data.get('simulation_summary', {}).get('total_wall_time', 0),
            'event_count': log_data.get('simulation_summary', {}).get('event_count', 0),
            'performance_summary': log_data.get('simulation_summary', {}).get('performance_summary', {}),
            'log_level': log_data.get('log_level', 'unknown'),
            'export_timestamp': log_data.get('export_timestamp', 0)
        }
        
        # Extract simulation events
        events = log_data.get('simulation_summary', {}).get('events', [])
        analysis['events'] = {
            'start_events': [e for e in events if e.get('event') == 'simulation_start'],
            'end_events': [e for e in events if e.get('event') == 'simulation_end'],
            'total_events': len(events)
        }
        
        return analysis
    
    @staticmethod
    def _analyze_text_log(log_path: str) -> Dict[str, Any]:
        """Analyze text-based log file"""
        analysis = {
            'log_type': 'text',
            'line_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'debug_count': 0
        }
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    analysis['line_count'] += 1
                    
                    if 'ERROR' in line:
                        analysis['error_count'] += 1
                    elif 'WARNING' in line:
                        analysis['warning_count'] += 1
                    elif 'INFO' in line:
                        analysis['info_count'] += 1
                    elif 'DEBUG' in line:
                        analysis['debug_count'] += 1
        
        except FileNotFoundError:
            analysis['error'] = 'File not found'
        
        return analysis
    
    @staticmethod
    def compare_simulation_runs(log_paths: list) -> Dict[str, Any]:
        """
        Compare multiple simulation runs
        
        Args:
            log_paths: List of log file paths
            
        Returns:
            Comparison results
        """
        analyses = []
        for path in log_paths:
            analysis = LogAnalyzer.analyze_log_file(path)
            analysis['log_path'] = path
            analyses.append(analysis)
        
        comparison = {
            'run_count': len(analyses),
            'runs': analyses,
            'summary': {}
        }
        
        # Calculate summary statistics
        if analyses:
            wall_times = [a.get('total_wall_time', 0) for a in analyses]
            event_counts = [a.get('event_count', 0) for a in analyses]
            
            comparison['summary'] = {
                'mean_wall_time': sum(wall_times) / len(wall_times) if wall_times else 0,
                'min_wall_time': min(wall_times) if wall_times else 0,
                'max_wall_time': max(wall_times) if wall_times else 0,
                'mean_event_count': sum(event_counts) / len(event_counts) if event_counts else 0
            }
        
        return comparison


def monitor_memory_usage(logger: logging.Logger, interval: float = 60.0):
    """
    Monitor memory usage and log warnings
    
    Args:
        logger: Logger instance
        interval: Monitoring interval in seconds
    """
    import psutil
    import threading
    
    def memory_monitor():
        while True:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            if memory_percent > 90:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent > 75:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 50:
                logger.info(f"Memory usage: {memory_percent:.1f}%")
            
            time.sleep(interval)
    
    thread = threading.Thread(target=memory_monitor, daemon=True)
    thread.start()
    logger.info("Memory monitoring started")


# Global logger instance for easy access
_global_logger = None

def get_logger() -> logging.Logger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


def set_global_logger(logger: logging.Logger):
    """Set the global logger instance"""
    global _global_logger
    _global_logger = logger