"""
Configuration loader for MagNanoBridge simulation
Handles YAML/JSON configuration files and provides default settings
"""

import os
import json
import yaml
import numpy as np
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    """Handles loading and validation of simulation configurations"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file (.json or .yaml)
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        try:
            with open(config_path, 'r') as f:
                if ext.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif ext.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {ext}")
            
            # Validate and fill defaults
            config = ConfigLoader._validate_and_fill_defaults(config)
            
            logging.getLogger(__name__).info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading config: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            config_path: Output file path
        """
        _, ext = os.path.splitext(config_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if ext.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif ext.lower() == '.json':
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {ext}")
            
            logging.getLogger(__name__).info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error saving config: {e}")
            raise
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default simulation configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Simulation parameters
            'simulation': {
                'max_steps': 10000,
                'dt': 1e-6,  # 1 microsecond time step
                'log_interval': 100,
                'max_history': 1000,
                'stop_on_bridge': True,
                'boundary_box': {
                    'min': [-10e-3, -10e-3, -15e-3],  # 10mm x 10mm x 15mm box
                    'max': [10e-3, 10e-3, 5e-3],
                    'restitution': 0.8  # Coefficient of restitution for walls
                }
            },
            
            # Particle system parameters
            'particles': {
                'count': 100,
                'radius': 50e-9,  # 50 nm core radius
                'shell_thickness': 10e-9,  # 10 nm polymer shell
                'mass': None,  # Calculated automatically
                'magnetic_moment': 1e-18,  # A⋅m²
                'size_distribution': {
                    'enabled': False,
                    'std_dev': 5e-9  # Standard deviation for size distribution
                }
            },
            
            # Physics parameters
            'physics': {
                'enable_dipole_forces': True,
                'enable_brownian_motion': True,
                'temperature': 310.0,  # Body temperature (K)
                'fluid_viscosity': 1e-3,  # Water viscosity (Pa⋅s)
                'magnetic_susceptibility': 1.0,
                'cutoff_distance': 1e-3  # 1mm cutoff for long-range forces
            },
            
            # Magnetic field system
            'fields': {
                'coil_positions': [
                    [0, 0, 10e-3],     # Top center
                    [10e-3, 0, 5e-3],  # Top right
                    [-10e-3, 0, 5e-3], # Top left
                    [15e-3, 0, 0],     # Right side
                    [-15e-3, 0, 0],    # Left side
                    [0, 0, -10e-3]     # Bottom center
                ],
                'coil_currents': [1.0, 0.8, 0.8, 0.5, 0.5, -0.3],
                'coil_radii': [15e-3, 12e-3, 12e-3, 10e-3, 10e-3, 15e-3],
                'coil_turns': [100, 100, 100, 100, 100, 100],
                'max_current': 5.0  # Maximum current per coil (A)
            },
            
            # Focus control system
            'focus': {
                'control_enabled': True,
                'control_type': 'pid',  # 'pid', 'adaptive', or 'ai'
                'target_zone': {
                    'center': [0, 0, 0],  # Focus at origin
                    'shape': 'cylinder',
                    'parameters': {
                        'radius': 1e-3,  # 1mm radius
                        'height': 2e-3,  # 2mm height
                        'axis': [0, 0, 1]  # Z-axis aligned
                    }
                },
                'pid_gains': {
                    'kp': 1.0,   # Proportional gain
                    'ki': 0.1,   # Integral gain
                    'kd': 0.05   # Derivative gain
                },
                'control_frequency': 100.0  # Hz
            },
            
            # Visualization settings
            'visualization': {
                'enabled': True,
                'real_time': True,
                'particle_trails': True,
                'trail_length': 50,
                'field_visualization': True,
                'camera_position': [20e-3, 20e-3, 20e-3],
                'camera_target': [0, 0, 0],
                'background_color': '#1a1a1a',
                'particle_color': '#ff6b35',
                'shell_color': '#ffd23f',
                'field_color': '#3dd8ff'
            },
            
            # Output and data export
            'output': {
                'data_dir': './data/output',
                'export_interval': 10,  # Export every 10 steps
                'export_formats': ['csv', 'json'],
                'save_trajectories': True,
                'save_forces': True,
                'save_field_data': False,  # Large files
                'compression': True
            },
            
            # Performance optimization
            'performance': {
                'use_fast_multipole': False,  # Enable for >1000 particles
                'parallel_force_calculation': False,
                'gpu_acceleration': False,  # Future feature
                'memory_limit_mb': 1000
            },
            
            # Validation and testing
            'validation': {
                'check_energy_conservation': True,
                'energy_tolerance': 0.1,  # 10% energy drift tolerance
                'check_particle_overlap': True,
                'max_overlap_fraction': 0.1
            }
        }
    
    @staticmethod
    def _validate_and_fill_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and fill missing values with defaults
        
        Args:
            config: Input configuration
            
        Returns:
            Validated and complete configuration
        """
        default_config = ConfigLoader.get_default_config()
        
        # Recursively merge with defaults
        validated_config = ConfigLoader._deep_merge(default_config, config)
        
        # Perform validation checks
        ConfigLoader._validate_config(validated_config)
        
        return validated_config
    
    @staticmethod
    def _deep_merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence
        
        Args:
            default: Default configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """
        Validate configuration values
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        logger = logging.getLogger(__name__)
        
        # Validate simulation parameters
        sim_config = config['simulation']
        if sim_config['dt'] <= 0:
            raise ValueError("Time step must be positive")
        if sim_config['max_steps'] <= 0:
            raise ValueError("Maximum steps must be positive")
        
        # Validate particle parameters
        particle_config = config['particles']
        if particle_config['count'] <= 0:
            raise ValueError("Particle count must be positive")
        if particle_config['radius'] <= 0:
            raise ValueError("Particle radius must be positive")
        if particle_config['shell_thickness'] < 0:
            raise ValueError("Shell thickness must be non-negative")
        
        # Validate physics parameters
        physics_config = config['physics']
        if physics_config['temperature'] <= 0:
            raise ValueError("Temperature must be positive")
        if physics_config['fluid_viscosity'] <= 0:
            raise ValueError("Fluid viscosity must be positive")
        
        # Validate field configuration
        field_config = config['fields']
        n_coils = len(field_config['coil_positions'])
        
        for field_list in ['coil_currents', 'coil_radii', 'coil_turns']:
            if len(field_config[field_list]) != n_coils:
                logger.warning(f"Inconsistent {field_list} length, will use defaults")
        
        # Validate focus zone
        focus_config = config['focus']['target_zone']
        if focus_config['shape'] not in ['cylinder', 'sphere', 'box', 'mesh']:
            raise ValueError(f"Unsupported focus zone shape: {focus_config['shape']}")
        
        # Validate PID gains
        pid_gains = config['focus']['pid_gains']
        if any(gain < 0 for gain in pid_gains.values()):
            logger.warning("Negative PID gains detected - may cause instability")
        
        logger.info("Configuration validation completed")
    
    @staticmethod
    def create_config_template(output_path: str, format: str = 'yaml'):
        """
        Create a configuration template file
        
        Args:
            output_path: Output file path
            format: File format ('yaml' or 'json')
        """
        config = ConfigLoader.get_default_config()
        
        # Add comments for YAML
        if format.lower() == 'yaml':
            config['_comments'] = {
                'simulation': 'Core simulation parameters',
                'particles': 'Particle system configuration',
                'physics': 'Physical constants and models',
                'fields': 'Magnetic field system setup',
                'focus': 'Focus control and target zone',
                'visualization': 'Rendering and display options',
                'output': 'Data export and file management',
                'performance': 'Optimization settings',
                'validation': 'Quality checks and validation'
            }
        
        # Ensure correct file extension
        if not output_path.endswith(('.yaml', '.yml', '.json')):
            output_path += f'.{format}'
        
        ConfigLoader.save_config(config, output_path)
        
        logging.getLogger(__name__).info(f"Configuration template created: {output_path}")
    
    @staticmethod
    def validate_config_file(config_path: str) -> bool:
        """
        Validate a configuration file without loading
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            config = ConfigLoader.load_config(config_path)
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Config validation failed: {e}")
            return False
    
    @staticmethod
    def get_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two configurations and return differences
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary containing differences
        """
        differences = {}
        
        def _compare_recursive(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {'status': 'added', 'value': d2[key]}
                elif key not in d2:
                    differences[current_path] = {'status': 'removed', 'value': d1[key]}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    _compare_recursive(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {
                        'status': 'changed',
                        'old_value': d1[key],
                        'new_value': d2[key]
                    }
        
        _compare_recursive(config1, config2)
        return differences
    
    @staticmethod
    def optimize_config_for_hardware(config: Dict[str, Any], 
                                   available_memory_gb: float = 8.0,
                                   cpu_cores: int = 4) -> Dict[str, Any]:
        """
        Optimize configuration based on available hardware
        
        Args:
            config: Input configuration
            available_memory_gb: Available RAM in GB
            cpu_cores: Number of CPU cores
            
        Returns:
            Optimized configuration
        """
        optimized = config.copy()
        logger = logging.getLogger(__name__)
        
        # Estimate memory usage
        particle_count = config['particles']['count']
        estimated_memory_mb = particle_count * 0.001  # Rough estimate: 1KB per particle
        
        # Adjust particle count if memory is limited
        max_particles = int(available_memory_gb * 1000 / 0.001)
        if particle_count > max_particles:
            optimized['particles']['count'] = max_particles
            logger.warning(f"Reduced particle count to {max_particles} due to memory constraints")
        
        # Enable optimizations for large systems
        if particle_count > 1000:
            optimized['performance']['use_fast_multipole'] = True
            logger.info("Enabled Fast Multipole Method for large particle system")
        
        if cpu_cores > 2:
            optimized['performance']['parallel_force_calculation'] = True
            logger.info("Enabled parallel force calculation")
        
        # Adjust export frequency for performance
        if particle_count > 500:
            optimized['output']['export_interval'] = max(50, optimized['output']['export_interval'])
            optimized['output']['save_field_data'] = False  # Disable for large systems
        
        # Optimize time step based on system size
        if particle_count > 200:
            # Smaller time step for stability with many particles
            optimized['simulation']['dt'] = min(5e-7, optimized['simulation']['dt'])
        
        logger.info(f"Configuration optimized for {cpu_cores} cores, {available_memory_gb}GB RAM")
        return optimized
    
    @staticmethod
    def create_scenario_configs():
        """Create predefined scenario configurations"""
        scenarios = {
            'small_test': {
                'description': 'Small test system for debugging',
                'particles': {'count': 50},
                'simulation': {'max_steps': 1000, 'dt': 1e-6},
                'focus': {'target_zone': {'parameters': {'radius': 0.5e-3, 'height': 1e-3}}}
            },
            
            'medium_demo': {
                'description': 'Medium system for demonstration',
                'particles': {'count': 200},
                'simulation': {'max_steps': 5000, 'dt': 5e-7},
                'focus': {'target_zone': {'parameters': {'radius': 1e-3, 'height': 2e-3}}}
            },
            
            'large_simulation': {
                'description': 'Large system for research',
                'particles': {'count': 1000},
                'simulation': {'max_steps': 20000, 'dt': 2e-7},
                'performance': {'use_fast_multipole': True},
                'focus': {'target_zone': {'parameters': {'radius': 1.5e-3, 'height': 3e-3}}}
            },
            
            'high_precision': {
                'description': 'High precision simulation',
                'simulation': {'dt': 1e-7, 'max_steps': 50000},
                'physics': {'cutoff_distance': 2e-3},
                'validation': {'energy_tolerance': 0.01},
                'output': {'export_interval': 5}
            },
            
            'fast_preview': {
                'description': 'Fast preview for quick results',
                'particles': {'count': 30},
                'simulation': {'max_steps': 500, 'dt': 5e-6},
                'physics': {'enable_brownian_motion': False},
                'output': {'export_interval': 50}
            }
        }
        
        return scenarios
    
    @staticmethod
    def apply_scenario(base_config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
        """
        Apply a predefined scenario to a base configuration
        
        Args:
            base_config: Base configuration
            scenario_name: Name of scenario to apply
            
        Returns:
            Modified configuration
        """
        scenarios = ConfigLoader.create_scenario_configs()
        
        if scenario_name not in scenarios:
            available = list(scenarios.keys())
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
        
        scenario = scenarios[scenario_name]
        result = ConfigLoader._deep_merge(base_config, scenario)
        
        logging.getLogger(__name__).info(f"Applied scenario '{scenario_name}': {scenario['description']}")
        return result