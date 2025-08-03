"""
Core simulation engine for MagNanoBridge
Handles the main simulation loop, physics integration, and coordination
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .particles import ParticleSystem, Particle
from .fields import MagneticFieldSystem
from .forces import ForceCalculator
from .integrator import VelocityVerletIntegrator
from ..control.focus_controller import FocusController
from ..utils.data_export import DataExporter


@dataclass
class SimulationState:
    """Container for simulation state information"""
    time: float
    step: int
    particles: np.ndarray  # positions
    velocities: np.ndarray
    forces: np.ndarray
    magnetic_field: np.ndarray
    field_gradient: np.ndarray
    focus_efficiency: float
    bridge_formed: bool


class MagneticSimulation:
    """
    Main simulation class coordinating all physics components
    """
    
    def __init__(self, 
                 particle_system: ParticleSystem,
                 field_system: MagneticFieldSystem,
                 focus_controller: FocusController,
                 config: Dict[str, Any]):
        """
        Initialize the magnetic simulation
        
        Args:
            particle_system: ParticleSystem instance
            field_system: MagneticFieldSystem instance  
            focus_controller: FocusController instance
            config: Configuration dictionary
        """
        self.particle_system = particle_system
        self.field_system = field_system
        self.focus_controller = focus_controller
        self.config = config
        
        # Initialize physics components
        self.force_calculator = ForceCalculator(
            enable_dipole=config['physics']['enable_dipole_forces'],
            enable_brownian=config['physics']['enable_brownian_motion'],
            temperature=config['physics']['temperature'],
            fluid_viscosity=config['physics']['fluid_viscosity']
        )
        
        self.integrator = VelocityVerletIntegrator()
        
        # Simulation state
        self.current_time = 0.0
        self.current_step = 0
        self.is_running = False
        self.simulation_history = []
        
        # Performance tracking
        self.performance_stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'force_calc_time': 0.0,
            'integration_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("MagneticSimulation initialized")
    
    def get_current_state(self) -> SimulationState:
        """Get current simulation state"""
        # Calculate magnetic field at particle positions
        positions = self.particle_system.get_positions()
        B_field, B_gradient = self.field_system.calculate_field_and_gradient(positions)
        
        # Calculate focus efficiency
        focus_efficiency = self.focus_controller.calculate_focus_efficiency(
            positions, self.particle_system.get_radii()
        )
        
        # Check if bridge is formed
        bridge_formed = self.focus_controller.check_bridge_formation(
            positions, self.particle_system.get_radii()
        )
        
        return SimulationState(
            time=self.current_time,
            step=self.current_step,
            particles=positions.copy(),
            velocities=self.particle_system.get_velocities().copy(),
            forces=self.particle_system.get_forces().copy(),
            magnetic_field=B_field,
            field_gradient=B_gradient,
            focus_efficiency=focus_efficiency,
            bridge_formed=bridge_formed
        )
    
    def step(self, dt: float) -> SimulationState:
        """
        Perform a single simulation step
        
        Args:
            dt: Time step size
            
        Returns:
            Current simulation state
        """
        step_start_time = time.time()
        
        # Get current particle state
        positions = self.particle_system.get_positions()
        velocities = self.particle_system.get_velocities()
        
        # Calculate magnetic field and gradient
        B_field, B_gradient = self.field_system.calculate_field_and_gradient(positions)
        
        # Update field system based on focus controller
        if self.config['focus']['control_enabled']:
            focus_efficiency = self.focus_controller.calculate_focus_efficiency(
                positions, self.particle_system.get_radii()
            )
            
            field_adjustments = self.focus_controller.update_control(
                focus_efficiency, positions, dt
            )
            
            # Apply field adjustments
            self.field_system.update_currents(field_adjustments)
            
            # Recalculate field with new currents
            B_field, B_gradient = self.field_system.calculate_field_and_gradient(positions)
        
        # Calculate all forces
        force_start_time = time.time()
        
        total_forces = self.force_calculator.calculate_total_forces(
            particles=self.particle_system.particles,
            positions=positions,
            velocities=velocities,
            magnetic_field=B_field,
            field_gradient=B_gradient,
            dt=dt
        )
        
        force_calc_time = time.time() - force_start_time
        
        # Update particle forces
        self.particle_system.set_forces(total_forces)
        
        # Integrate motion
        integration_start_time = time.time()
        
        new_positions, new_velocities = self.integrator.integrate(
            positions=positions,
            velocities=velocities,
            forces=total_forces,
            masses=self.particle_system.get_masses(),
            dt=dt
        )
        
        integration_time = time.time() - integration_start_time
        
        # Update particle system
        self.particle_system.set_positions(new_positions)
        self.particle_system.set_velocities(new_velocities)
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Update simulation state
        self.current_time += dt
        self.current_step += 1
        
        # Update performance stats
        step_time = time.time() - step_start_time
        self._update_performance_stats(step_time, force_calc_time, integration_time)
        
        # Get current state
        current_state = self.get_current_state()
        
        # Log progress periodically
        if self.current_step % self.config['simulation']['log_interval'] == 0:
            self.logger.info(
                f"Step {self.current_step}: t={self.current_time:.3f}s, "
                f"focus_eff={current_state.focus_efficiency:.3f}, "
                f"bridge={current_state.bridge_formed}"
            )
        
        return current_state
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to prevent particles from escaping"""
        positions = self.particle_system.get_positions()
        velocities = self.particle_system.get_velocities()
        
        # Get boundary box from config
        boundary = self.config['simulation']['boundary_box']
        
        # Reflective boundaries
        for i in range(3):  # x, y, z
            # Lower boundary
            mask_low = positions[:, i] < boundary['min'][i]
            positions[mask_low, i] = boundary['min'][i]
            velocities[mask_low, i] = -velocities[mask_low, i] * boundary['restitution']
            
            # Upper boundary
            mask_high = positions[:, i] > boundary['max'][i]
            positions[mask_high, i] = boundary['max'][i]
            velocities[mask_high, i] = -velocities[mask_high, i] * boundary['restitution']
        
        # Update particle system
        self.particle_system.set_positions(positions)
        self.particle_system.set_velocities(velocities)
    
    def _update_performance_stats(self, step_time: float, force_time: float, 
                                 integration_time: float):
        """Update performance statistics"""
        self.performance_stats['total_steps'] += 1
        n = self.performance_stats['total_steps']
        
        # Running average
        self.performance_stats['avg_step_time'] = (
            (n - 1) * self.performance_stats['avg_step_time'] + step_time
        ) / n
        
        self.performance_stats['force_calc_time'] = (
            (n - 1) * self.performance_stats['force_calc_time'] + force_time
        ) / n
        
        self.performance_stats['integration_time'] = (
            (n - 1) * self.performance_stats['integration_time'] + integration_time
        ) / n
    
    def run_interactive(self, steps: int, dt: float, real_time: bool = True,
                       export_interval: int = 100, exporter: DataExporter = None) -> Dict[str, Any]:
        """
        Run simulation in interactive mode with real-time visualization
        
        Args:
            steps: Maximum number of steps
            dt: Time step size
            real_time: Whether to run in real-time
            export_interval: Steps between data exports
            exporter: Data exporter instance
            
        Returns:
            Final simulation results
        """
        self.logger.info(f"Starting interactive simulation: {steps} steps, dt={dt}")
        self.is_running = True
        
        try:
            for step_num in range(steps):
                if not self.is_running:
                    break
                
                step_start = time.time()
                
                # Perform simulation step
                state = self.step(dt)
                
                # Export data if needed
                if exporter and step_num % export_interval == 0:
                    exporter.export_state(state)
                
                # Store in history (limited buffer)
                self.simulation_history.append(state)
                if len(self.simulation_history) > self.config['simulation']['max_history']:
                    self.simulation_history.pop(0)
                
                # Real-time delay if requested
                if real_time:
                    target_step_time = dt
                    actual_step_time = time.time() - step_start
                    if actual_step_time < target_step_time:
                        time.sleep(target_step_time - actual_step_time)
                
                # Check for early termination conditions
                if state.bridge_formed and self.config['simulation']['stop_on_bridge']:
                    self.logger.info(f"Bridge formed at step {step_num}, stopping simulation")
                    break
            
            # Final export
            if exporter:
                final_state = self.get_current_state()
                exporter.export_state(final_state)
            
            return self._get_final_results()
            
        except KeyboardInterrupt:
            self.logger.info("Interactive simulation interrupted")
            self.is_running = False
            return self._get_final_results()
    
    def run_batch(self, steps: int, dt: float, export_interval: int = 100,
                  exporter: DataExporter = None) -> Dict[str, Any]:
        """
        Run simulation in batch mode (no visualization, optimized for speed)
        
        Args:
            steps: Number of steps to run
            dt: Time step size
            export_interval: Steps between data exports
            exporter: Data exporter instance
            
        Returns:
            Simulation results dictionary
        """
        self.logger.info(f"Starting batch simulation: {steps} steps, dt={dt}")
        
        start_time = time.time()
        
        for step_num in range(steps):
            # Perform simulation step
            state = self.step(dt)
            
            # Export data if needed
            if exporter and step_num % export_interval == 0:
                exporter.export_state(state)
            
            # Check for early termination
            if state.bridge_formed and self.config['simulation']['stop_on_bridge']:
                self.logger.info(f"Bridge formed at step {step_num}")
                break
        
        # Final export
        if exporter:
            final_state = self.get_current_state()
            exporter.export_state(final_state)
        
        total_time = time.time() - start_time
        self.logger.info(f"Batch simulation completed in {total_time:.2f}s")
        
        return self._get_final_results()
    
    def _get_final_results(self) -> Dict[str, Any]:
        """Generate final simulation results"""
        final_state = self.get_current_state()
        
        return {
            'total_time': self.current_time,
            'total_steps': self.current_step,
            'focus_efficiency': final_state.focus_efficiency,
            'bridge_formed': final_state.bridge_formed,
            'bridge_time': self.current_time if final_state.bridge_formed else None,
            'final_positions': final_state.particles.tolist(),
            'performance_stats': self.performance_stats.copy(),
            'particle_count': len(self.particle_system.particles),
            'config_used': self.config.copy()
        }
    
    def stop(self):
        """Stop the simulation"""
        self.is_running = False
        self.logger.info("Simulation stop requested")
    
    def reset(self):
        """Reset simulation to initial state"""
        self.current_time = 0.0
        self.current_step = 0
        self.simulation_history.clear()
        self.particle_system.reset()
        self.field_system.reset()
        self.focus_controller.reset()
        self.logger.info("Simulation reset")
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization"""
        state = self.get_current_state()
        
        return {
            'time': state.time,
            'step': state.step,
            'particles': {
                'positions': state.particles.tolist(),
                'velocities': state.velocities.tolist(),
                'radii': self.particle_system.get_radii().tolist(),
                'shell_thickness': [p.shell_thickness for p in self.particle_system.particles],
                'ids': [p.id for p in self.particle_system.particles]
            },
            'magnetic_field': {
                'field_vectors': state.magnetic_field.tolist(),
                'field_magnitude': np.linalg.norm(state.magnetic_field, axis=1).tolist()
            },
            'focus_zone': self.focus_controller.get_focus_zone_data(),
            'metrics': {
                'focus_efficiency': state.focus_efficiency,
                'bridge_formed': state.bridge_formed,
                'particle_count_in_zone': self.focus_controller.count_particles_in_zone(
                    state.particles, self.particle_system.get_radii()
                )
            },
            'coils': self.field_system.get_coil_data()
        }