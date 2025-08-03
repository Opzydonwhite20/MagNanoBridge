"""
Focus controller for guiding particles to target zone
Implements PID and AI-based control strategies
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class FocusZone:
    """Definition of the focus zone geometry"""
    center: np.ndarray         # Center position [x, y, z]
    shape: str                # 'cylinder', 'sphere', 'box', 'mesh'
    parameters: Dict[str, Any] # Shape-specific parameters
    
    def __post_init__(self):
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center, dtype=np.float64)


class FocusController:
    """
    Controller for guiding magnetic nanoparticles to a target focus zone
    Supports PID control and AI-based optimization
    """
    
    def __init__(self,
                 target_zone: Dict[str, Any],
                 control_type: str = 'pid',
                 pid_gains: Dict[str, float] = None,
                 control_frequency: float = 100.0):  # Hz
        """
        Initialize focus controller
        
        Args:
            target_zone: Dictionary defining the focus zone
            control_type: 'pid', 'adaptive', or 'ai'
            pid_gains: PID gains {'kp', 'ki', 'kd'}
            control_frequency: Control update frequency (Hz)
        """
        self.focus_zone = self._create_focus_zone(target_zone)
        self.control_type = control_type.lower()
        self.control_frequency = control_frequency
        
        # PID parameters
        if pid_gains is None:
            pid_gains = {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}
        self.pid_gains = pid_gains
        
        # Control state
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.last_update_time = 0.0
        
        # Performance tracking
        self.control_history = []
        self.efficiency_history = []
        
        # AI controller (if enabled)
        self.ai_controller = None
        if self.control_type == 'ai':
            self._initialize_ai_controller()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FocusController initialized: {self.control_type} control, "
                        f"zone: {self.focus_zone.shape}")
    
    def _create_focus_zone(self, zone_config: Dict[str, Any]) -> FocusZone:
        """Create focus zone from configuration"""
        center = np.array(zone_config.get('center', [0, 0, 0]))
        shape = zone_config.get('shape', 'cylinder')
        parameters = zone_config.get('parameters', {})
        
        # Validate and set default parameters based on shape
        if shape == 'cylinder':
            if 'radius' not in parameters:
                parameters['radius'] = 1e-3  # 1mm default
            if 'height' not in parameters:
                parameters['height'] = 2e-3  # 2mm default
            if 'axis' not in parameters:
                parameters['axis'] = np.array([0, 0, 1])  # z-axis default
        
        elif shape == 'sphere':
            if 'radius' not in parameters:
                parameters['radius'] = 1e-3  # 1mm default
        
        elif shape == 'box':
            if 'dimensions' not in parameters:
                parameters['dimensions'] = np.array([2e-3, 2e-3, 2e-3])  # 2mm cube
        
        elif shape == 'mesh':
            if 'vertices' not in parameters or 'faces' not in parameters:
                raise ValueError("Mesh focus zone requires 'vertices' and 'faces' parameters")
        
        return FocusZone(center=center, shape=shape, parameters=parameters)
    
    def _initialize_ai_controller(self):
        """Initialize AI-based controller (neural network)"""
        try:
            # Simple neural network for demonstration
            # In practice, would use more sophisticated AI/ML framework
            from .ai_controller import SimpleNeuralController
            self.ai_controller = SimpleNeuralController(
                input_dim=12,  # particle positions, velocities, field state
                output_dim=6,  # field adjustments for 6 coils
                hidden_dim=32
            )
            self.logger.info("AI controller initialized")
        except ImportError:
            self.logger.warning("AI controller not available, falling back to PID")
            self.control_type = 'pid'
    
    def calculate_focus_efficiency(self, 
                                 positions: np.ndarray,
                                 radii: np.ndarray) -> float:
        """
        Calculate focus efficiency (fraction of particles in focus zone)
        
        Args:
            positions: Particle positions (N, 3)
            radii: Particle radii (N,)
            
        Returns:
            Focus efficiency [0, 1]
        """
        if len(positions) == 0:
            return 0.0
        
        particles_in_zone = self.count_particles_in_zone(positions, radii)
        efficiency = particles_in_zone / len(positions)
        
        # Store in history
        self.efficiency_history.append(efficiency)
        if len(self.efficiency_history) > 1000:  # Limit history size
            self.efficiency_history.pop(0)
        
        return efficiency
    
    def count_particles_in_zone(self, positions: np.ndarray, radii: np.ndarray) -> int:
        """Count particles inside the focus zone"""
        count = 0
        
        for i, pos in enumerate(positions):
            if self._point_in_focus_zone(pos, radii[i]):
                count += 1
        
        return count
    
    def _point_in_focus_zone(self, point: np.ndarray, particle_radius: float = 0.0) -> bool:
        """Check if a point (with optional radius) is inside focus zone"""
        relative_pos = point - self.focus_zone.center
        
        if self.focus_zone.shape == 'cylinder':
            radius = self.focus_zone.parameters['radius']
            height = self.focus_zone.parameters['height']
            axis = self.focus_zone.parameters.get('axis', np.array([0, 0, 1]))
            
            # Project onto cylinder axis
            axis_normalized = axis / np.linalg.norm(axis)
            axial_distance = abs(np.dot(relative_pos, axis_normalized))
            
            # Radial distance from axis
            radial_component = relative_pos - np.dot(relative_pos, axis_normalized) * axis_normalized
            radial_distance = np.linalg.norm(radial_component)
            
            return (radial_distance <= radius - particle_radius and
                   axial_distance <= height / 2 - particle_radius)
        
        elif self.focus_zone.shape == 'sphere':
            radius = self.focus_zone.parameters['radius']
            distance = np.linalg.norm(relative_pos)
            return distance <= radius - particle_radius
        
        elif self.focus_zone.shape == 'box':
            dimensions = self.focus_zone.parameters['dimensions']
            half_dims = dimensions / 2 - particle_radius
            
            return (abs(relative_pos[0]) <= half_dims[0] and
                   abs(relative_pos[1]) <= half_dims[1] and
                   abs(relative_pos[2]) <= half_dims[2])
        
        elif self.focus_zone.shape == 'mesh':
            # For mesh, use point-in-polygon test (simplified)
            # This is a placeholder - real implementation would use
            # sophisticated point-in-mesh algorithms
            return self._point_in_mesh(point, particle_radius)
        
        return False
    
    def _point_in_mesh(self, point: np.ndarray, particle_radius: float) -> bool:
        """Check if point is inside mesh focus zone (placeholder)"""
        # Simplified implementation - just check distance to center
        distance = np.linalg.norm(point - self.focus_zone.center)
        estimated_radius = self.focus_zone.parameters.get('estimated_radius', 1e-3)
        return distance <= estimated_radius - particle_radius
    
    def check_bridge_formation(self, positions: np.ndarray, radii: np.ndarray) -> bool:
        """
        Check if particles have formed a sufficient bridge
        
        Args:
            positions: Particle positions (N, 3)
            radii: Particle radii (N,)
            
        Returns:
            True if bridge is formed
        """
        # Simple bridge criteria: sufficient density in focus zone
        particles_in_zone = self.count_particles_in_zone(positions, radii)
        focus_efficiency = particles_in_zone / len(positions) if len(positions) > 0 else 0
        
        # Bridge formation criteria
        min_efficiency = 0.6  # 60% of particles in zone
        min_particles = 10    # Minimum absolute number
        
        bridge_formed = (focus_efficiency >= min_efficiency and 
                        particles_in_zone >= min_particles)
        
        if bridge_formed:
            # Additional check: connectivity
            bridge_formed = self._check_particle_connectivity(positions, radii)
        
        return bridge_formed
    
    def _check_particle_connectivity(self, positions: np.ndarray, radii: np.ndarray) -> bool:
        """Check if particles in focus zone are sufficiently connected"""
        # Get particles in focus zone
        zone_particles = []
        zone_positions = []
        
        for i, pos in enumerate(positions):
            if self._point_in_focus_zone(pos, radii[i]):
                zone_particles.append(i)
                zone_positions.append(pos)
        
        if len(zone_positions) < 3:
            return False
        
        zone_positions = np.array(zone_positions)
        
        # Check connectivity using neighbor counting
        connected_count = 0
        connection_distance = 3.0  # particles within 3 radii are "connected"
        
        for i, pos1 in enumerate(zone_positions):
            neighbors = 0
            for j, pos2 in enumerate(zone_positions):
                if i != j:
                    distance = np.linalg.norm(pos1 - pos2)
                    particle_idx1 = zone_particles[i]
                    particle_idx2 = zone_particles[j]
                    max_connection_dist = connection_distance * (radii[particle_idx1] + radii[particle_idx2])
                    
                    if distance <= max_connection_dist:
                        neighbors += 1
            
            if neighbors >= 2:  # Each particle should have at least 2 neighbors
                connected_count += 1
        
        # Bridge is formed if most particles are well-connected
        connectivity_threshold = 0.7
        return connected_count / len(zone_positions) >= connectivity_threshold
    
    def update_control(self, 
                      current_efficiency: float,
                      particle_positions: np.ndarray,
                      dt: float) -> Dict[str, float]:
        """
        Update control based on current focus efficiency
        
        Args:
            current_efficiency: Current focus efficiency [0, 1]
            particle_positions: Current particle positions (N, 3)
            dt: Time step
            
        Returns:
            Dictionary of field adjustments for each coil
        """
        # Update timing
        current_time = self.last_update_time + dt
        self.last_update_time = current_time
        
        # Calculate control updates based on type
        if self.control_type == 'pid':
            field_adjustments = self._pid_control(current_efficiency, dt)
        elif self.control_type == 'adaptive':
            field_adjustments = self._adaptive_control(current_efficiency, particle_positions, dt)
        elif self.control_type == 'ai' and self.ai_controller is not None:
            field_adjustments = self._ai_control(current_efficiency, particle_positions)
        else:
            # Fallback to PID
            field_adjustments = self._pid_control(current_efficiency, dt)
        
        # Store control history
        control_state = {
            'time': current_time,
            'efficiency': current_efficiency,
            'adjustments': field_adjustments.copy(),
            'error': 1.0 - current_efficiency
        }
        self.control_history.append(control_state)
        
        # Limit history size
        if len(self.control_history) > 1000:
            self.control_history.pop(0)
        
        return field_adjustments
    
    def _pid_control(self, current_efficiency: float, dt: float) -> Dict[str, float]:
        """PID control implementation"""
        # Target efficiency
        target_efficiency = 0.8  # 80% target
        
        # Calculate error
        error = target_efficiency - current_efficiency
        
        # Proportional term
        proportional = self.pid_gains['kp'] * error
        
    def _pid_control(self, current_efficiency: float, dt: float) -> Dict[str, float]:
        """PID control implementation"""
        # Target efficiency
        target_efficiency = 0.8  # 80% target
        
        # Calculate error
        error = target_efficiency - current_efficiency
        
        # Proportional term
        proportional = self.pid_gains['kp'] * error
        
        # Integral term
        self.integral_error += error * dt
        integral = self.pid_gains['ki'] * self.integral_error
        
        # Derivative term
        derivative = self.pid_gains['kd'] * (error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = error
        
        # PID output
        pid_output = proportional + integral + derivative
        
        # Convert to field adjustments
        # Simple strategy: increase all coil currents proportionally
        base_adjustment = pid_output * 0.1  # Scale factor
        
        field_adjustments = {
            'coil_0': base_adjustment,
            'coil_1': base_adjustment * 0.8,
            'coil_2': base_adjustment * 0.8,
            'coil_3': base_adjustment * 0.5,
            'coil_4': base_adjustment * 0.5,
            'coil_5': -base_adjustment * 0.3  # Bottom coil opposite
        }
        
        return field_adjustments
    
    def _adaptive_control(self, 
                         current_efficiency: float,
                         particle_positions: np.ndarray,
                         dt: float) -> Dict[str, float]:
        """Adaptive control with zone-specific field adjustments"""
        target_efficiency = 0.8
        error = target_efficiency - current_efficiency
        
        # Analyze particle distribution
        zone_center = self.focus_zone.center
        particle_vectors = particle_positions - zone_center
        
        # Calculate center of mass of particles
        particle_com = np.mean(particle_positions, axis=0)
        com_offset = particle_com - zone_center
        
        # Adaptive gains based on error magnitude
        if abs(error) > 0.5:
            # Large error - aggressive control
            gain_multiplier = 2.0
        elif abs(error) > 0.2:
            # Medium error - normal control
            gain_multiplier = 1.0
        else:
            # Small error - gentle control
            gain_multiplier = 0.5
        
        # Directional field adjustments based on COM offset
        base_strength = error * gain_multiplier * 0.1
        
        # Coil-specific adjustments to pull particles toward zone
        field_adjustments = {}
        
        # Top coils (attract if particles are below)
        if com_offset[2] < 0:  # Particles below target
            field_adjustments['coil_0'] = base_strength * 1.5
            field_adjustments['coil_1'] = base_strength * 1.2
            field_adjustments['coil_2'] = base_strength * 1.2
        else:
            field_adjustments['coil_0'] = base_strength * 0.5
            field_adjustments['coil_1'] = base_strength * 0.5
            field_adjustments['coil_2'] = base_strength * 0.5
        
        # Side coils (adjust for lateral positioning)
        if com_offset[0] > 0:  # Particles to the right
            field_adjustments['coil_3'] = base_strength * 1.0  # Right coil
            field_adjustments['coil_4'] = base_strength * 0.5  # Left coil
        else:  # Particles to the left
            field_adjustments['coil_3'] = base_strength * 0.5
            field_adjustments['coil_4'] = base_strength * 1.0
        
        # Bottom coil (repel if particles are above)
        if com_offset[2] > 0:  # Particles above target
            field_adjustments['coil_5'] = -base_strength * 0.8
        else:
            field_adjustments['coil_5'] = base_strength * 0.2
        
        return field_adjustments
    
    def _ai_control(self, 
                   current_efficiency: float,
                   particle_positions: np.ndarray) -> Dict[str, float]:
        """AI-based control using neural network"""
        if self.ai_controller is None:
            return self._pid_control(current_efficiency, 0.01)
        
        # Prepare input features for neural network
        features = self._prepare_ai_features(current_efficiency, particle_positions)
        
        # Get control outputs from AI
        control_outputs = self.ai_controller.predict(features)
        
        # Convert to field adjustments
        field_adjustments = {
            f'coil_{i}': float(control_outputs[i]) for i in range(len(control_outputs))
        }
        
        return field_adjustments
    
    def _prepare_ai_features(self, 
                           current_efficiency: float,
                           particle_positions: np.ndarray) -> np.ndarray:
        """Prepare feature vector for AI controller"""
        zone_center = self.focus_zone.center
        
        # Basic features
        features = [current_efficiency]
        
        # Center of mass features
        if len(particle_positions) > 0:
            com = np.mean(particle_positions, axis=0)
            com_offset = com - zone_center
            features.extend(com_offset.tolist())
            
            # Distance statistics
            distances = np.linalg.norm(particle_positions - zone_center, axis=1)
            features.extend([
                np.mean(distances),
                np.std(distances),
                np.min(distances),
                np.max(distances)
            ])
        else:
            # No particles - default features
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Pad to fixed size
        while len(features) < 12:
            features.append(0.0)
        
        return np.array(features[:12])
    
    def get_focus_zone_data(self) -> Dict[str, Any]:
        """Get focus zone data for visualization"""
        return {
            'center': self.focus_zone.center.tolist(),
            'shape': self.focus_zone.shape,
            'parameters': self.focus_zone.parameters.copy(),
            'efficiency_history': self.efficiency_history[-100:],  # Last 100 points
            'target_efficiency': 0.8
        }
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """Get control performance statistics"""
        if not self.efficiency_history:
            return {}
        
        recent_efficiency = self.efficiency_history[-50:] if len(self.efficiency_history) >= 50 else self.efficiency_history
        
        return {
            'current_efficiency': self.efficiency_history[-1] if self.efficiency_history else 0,
            'mean_efficiency': np.mean(recent_efficiency),
            'std_efficiency': np.std(recent_efficiency),
            'max_efficiency': np.max(self.efficiency_history),
            'convergence_time': self._estimate_convergence_time(),
            'control_type': self.control_type,
            'pid_gains': self.pid_gains.copy(),
            'update_count': len(self.control_history)
        }
    
    def _estimate_convergence_time(self) -> Optional[float]:
        """Estimate time to convergence (80% efficiency)"""
        target = 0.8
        convergence_threshold = 0.75  # 75% efficiency considered "converged"
        
        for i, efficiency in enumerate(self.efficiency_history):
            if efficiency >= convergence_threshold:
                return i / self.control_frequency  # Convert to seconds
        
        return None  # Not yet converged
    
    def optimize_zone_parameters(self, 
                               particle_positions: np.ndarray,
                               particle_radii: np.ndarray) -> Dict[str, Any]:
        """
        Optimize focus zone parameters based on particle distribution
        
        Args:
            particle_positions: Current particle positions
            particle_radii: Particle radii
            
        Returns:
            Optimized zone parameters
        """
        if len(particle_positions) < 5:
            return self.focus_zone.parameters.copy()
        
        # Calculate particle distribution statistics
        com = np.mean(particle_positions, axis=0)
        distances = np.linalg.norm(particle_positions - com, axis=1)
        
        # Suggest optimal zone parameters
        optimal_params = self.focus_zone.parameters.copy()
        
        if self.focus_zone.shape == 'sphere':
            # Set radius to contain 80% of particles
            sorted_distances = np.sort(distances)
            optimal_radius = sorted_distances[int(0.8 * len(sorted_distances))]
            optimal_params['radius'] = max(optimal_radius, np.mean(particle_radii) * 3)
        
        elif self.focus_zone.shape == 'cylinder':
            # Optimize radius and height based on distribution
            radial_distances = np.linalg.norm(particle_positions[:, :2] - com[:2], axis=1)
            axial_distances = np.abs(particle_positions[:, 2] - com[2])
            
            optimal_params['radius'] = np.percentile(radial_distances, 80)
            optimal_params['height'] = 2 * np.percentile(axial_distances, 80)
        
        return optimal_params
    
    def reset(self):
        """Reset controller state"""
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.last_update_time = 0.0
        self.control_history.clear()
        self.efficiency_history.clear()
        
        if self.ai_controller is not None:
            self.ai_controller.reset()
        
        self.logger.info("Focus controller reset")
    
    def save_control_data(self, filepath: str):
        """Save control history and performance data"""
        import json
        
        data = {
            'control_type': self.control_type,
            'focus_zone': {
                'center': self.focus_zone.center.tolist(),
                'shape': self.focus_zone.shape,
                'parameters': self.focus_zone.parameters
            },
            'pid_gains': self.pid_gains,
            'control_history': self.control_history,
            'efficiency_history': self.efficiency_history,
            'statistics': self.get_control_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Control data saved to {filepath}")
    
    def load_control_data(self, filepath: str):
        """Load control history and performance data"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.control_type = data['control_type']
        self.pid_gains = data['pid_gains']
        self.control_history = data['control_history']
        self.efficiency_history = data['efficiency_history']
        
        # Reconstruct focus zone
        zone_data = data['focus_zone']
        self.focus_zone = FocusZone(
            center=np.array(zone_data['center']),
            shape=zone_data['shape'],
            parameters=zone_data['parameters']
        )
        
        self.logger.info(f"Control data loaded from {filepath}")
    
    def tune_pid_gains(self, 
                      particle_positions: np.ndarray,
                      particle_radii: np.ndarray,
                      simulation_steps: int = 100) -> Dict[str, float]:
        """
        Auto-tune PID gains using Ziegler-Nichols method
        
        Args:
            particle_positions: Current particle positions
            particle_radii: Particle radii
            simulation_steps: Number of steps for tuning
            
        Returns:
            Optimized PID gains
        """
        # This is a simplified auto-tuning algorithm
        # Real implementation would use more sophisticated methods
        
        self.logger.info("Starting PID auto-tuning")
        
        # Store original gains
        original_gains = self.pid_gains.copy()
        
        # Test different gain values
        test_gains = [
            {'kp': 0.5, 'ki': 0.05, 'kd': 0.02},
            {'kp': 1.0, 'ki': 0.1, 'kd': 0.05},
            {'kp': 2.0, 'ki': 0.2, 'kd': 0.1},
            {'kp': 1.5, 'ki': 0.15, 'kd': 0.08}
        ]
        
        best_gains = original_gains
        best_performance = 0.0
        
        for gains in test_gains:
            self.pid_gains = gains
            self.reset()
            
            # Simulate control performance
            efficiency_sum = 0.0
            for step in range(simulation_steps):
                current_efficiency = self.calculate_focus_efficiency(particle_positions, particle_radii)
                field_adjustments = self.update_control(current_efficiency, particle_positions, 0.01)
                efficiency_sum += current_efficiency
            
            avg_efficiency = efficiency_sum / simulation_steps
            
            if avg_efficiency > best_performance:
                best_performance = avg_efficiency
                best_gains = gains.copy()
        
        # Apply best gains
        self.pid_gains = best_gains
        self.reset()
        
        self.logger.info(f"PID auto-tuning completed. Best gains: {best_gains}")
        return best_gains
    
    def set_focus_zone(self, zone_config: Dict[str, Any]):
        """Update focus zone configuration"""
        self.focus_zone = self._create_focus_zone(zone_config)
        self.reset()
        self.logger.info(f"Focus zone updated: {self.focus_zone.shape}")
    
    def get_gradient_field_suggestion(self, 
                                    particle_positions: np.ndarray) -> Dict[str, Any]:
        """
        Suggest optimal field gradient configuration
        
        Args:
            particle_positions: Current particle positions
            
        Returns:
            Suggested field configuration
        """
        zone_center = self.focus_zone.center
        
        if len(particle_positions) == 0:
            return {'suggestion': 'uniform_attraction'}
        
        # Analyze particle distribution
        com = np.mean(particle_positions, axis=0)
        offset = com - zone_center
        spread = np.std(particle_positions, axis=0)
        
        suggestions = {}
        
        # Vertical positioning
        if offset[2] > 1e-3:  # Particles too high
            suggestions['vertical'] = 'increase_bottom_repulsion'
        elif offset[2] < -1e-3:  # Particles too low
            suggestions['vertical'] = 'increase_top_attraction'
        else:
            suggestions['vertical'] = 'maintain_vertical_balance'
        
        # Horizontal positioning
        if np.linalg.norm(offset[:2]) > 1e-3:  # Particles off-center
            suggestions['horizontal'] = 'apply_lateral_correction'
        else:
            suggestions['horizontal'] = 'maintain_centering'
        
        # Focusing
        if np.mean(spread) > 2e-3:  # Particles too spread out
            suggestions['focusing'] = 'increase_gradient_strength'
        else:
            suggestions['focusing'] = 'maintain_current_gradient'
        
        # Overall strategy
        efficiency = self.calculate_focus_efficiency(particle_positions, 
                                                   np.full(len(particle_positions), 50e-9))
        
        if efficiency < 0.3:
            suggestions['strategy'] = 'aggressive_collection'
        elif efficiency < 0.7:
            suggestions['strategy'] = 'moderate_focusing'
        else:
            suggestions['strategy'] = 'fine_tuning'
        
        return suggestions