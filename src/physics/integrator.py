"""
Numerical integration methods for particle motion
Implements stable integrators for the magnetic nanoparticle simulation
"""

import numpy as np
import logging
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class Integrator(ABC):
    """Abstract base class for numerical integrators"""
    
    @abstractmethod
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step
        
        Args:
            positions: Current positions (N, 3)
            velocities: Current velocities (N, 3)
            forces: Current forces (N, 3)
            masses: Particle masses (N,)
            dt: Time step
            
        Returns:
            Tuple of (new_positions, new_velocities)
        """
        pass


class EulerIntegrator(Integrator):
    """
    Simple Euler integrator (first-order)
    Not recommended for production use due to instability
    """
    
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Forward Euler integration"""
        accelerations = forces / masses.reshape(-1, 1)
        
        new_positions = positions + velocities * dt
        new_velocities = velocities + accelerations * dt
        
        return new_positions, new_velocities


class VelocityVerletIntegrator(Integrator):
    """
    Velocity Verlet integrator (second-order, symplectic)
    Recommended for molecular dynamics simulations
    """
    
    def __init__(self):
        self.previous_accelerations = None
        self.logger = logging.getLogger(__name__)
    
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Velocity Verlet integration algorithm
        
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
        """
        current_accelerations = forces / masses.reshape(-1, 1)
        
        # Update positions
        new_positions = (positions + 
                        velocities * dt + 
                        0.5 * current_accelerations * dt**2)
        
        # Update velocities
        if self.previous_accelerations is not None:
            new_velocities = (velocities + 
                            0.5 * (self.previous_accelerations + current_accelerations) * dt)
        else:
            # First step: use current acceleration only
            new_velocities = velocities + current_accelerations * dt
        
        # Store accelerations for next step
        self.previous_accelerations = current_accelerations.copy()
        
        return new_positions, new_velocities
    
    def reset(self):
        """Reset integrator state"""
        self.previous_accelerations = None


class RungeKutta4Integrator(Integrator):
    """
    Fourth-order Runge-Kutta integrator
    Higher accuracy but more computationally expensive
    """
    
    def __init__(self, force_calculator=None):
        """
        Initialize RK4 integrator
        
        Args:
            force_calculator: Force calculator for intermediate steps
        """
        self.force_calculator = force_calculator
        self.logger = logging.getLogger(__name__)
    
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fourth-order Runge-Kutta integration
        Note: This implementation assumes constant forces during the step
        For full RK4 with force evaluation, need access to force calculator
        """
        # Convert to accelerations
        accelerations = forces / masses.reshape(-1, 1)
        
        # RK4 coefficients for position and velocity
        # k1
        k1_pos = velocities
        k1_vel = accelerations
        
        # k2  
        k2_pos = velocities + 0.5 * k1_vel * dt
        k2_vel = accelerations  # Assuming constant acceleration
        
        # k3
        k3_pos = velocities + 0.5 * k2_vel * dt  
        k3_vel = accelerations
        
        # k4
        k4_pos = velocities + k3_vel * dt
        k4_vel = accelerations
        
        # Final update
        new_positions = positions + (dt / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        new_velocities = velocities + (dt / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        
        return new_positions, new_velocities


class AdaptiveIntegrator(Integrator):
    """
    Adaptive time-stepping integrator using embedded Runge-Kutta method
    Automatically adjusts time step for optimal accuracy/efficiency
    """
    
    def __init__(self, 
                 tolerance: float = 1e-6,
                 min_dt: float = 1e-8,
                 max_dt: float = 1e-3,
                 safety_factor: float = 0.9):
        """
        Initialize adaptive integrator
        
        Args:
            tolerance: Error tolerance for adaptive stepping
            min_dt: Minimum allowed time step
            max_dt: Maximum allowed time step
            safety_factor: Safety factor for step size adjustment
        """
        self.tolerance = tolerance
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.safety_factor = safety_factor
        self.current_dt = None
        self.logger = logging.getLogger(__name__)
    
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive integration using Runge-Kutta-Fehlberg method
        """
        if self.current_dt is None:
            self.current_dt = dt
        
        # Attempt integration with current time step
        h = min(self.current_dt, dt)
        
        # RKF45 coefficients (simplified version)
        accelerations = forces / masses.reshape(-1, 1)
        
        # Fourth-order estimate
        pos_4th, vel_4th = self._rk4_step(positions, velocities, accelerations, h)
        
        # Fifth-order estimate (simplified)
        pos_5th, vel_5th = self._rk5_step(positions, velocities, accelerations, h)
        
        # Estimate error
        pos_error = np.max(np.abs(pos_5th - pos_4th))
        vel_error = np.max(np.abs(vel_5th - vel_4th))
        error = max(pos_error, vel_error)
        
        # Adjust time step for next iteration
        if error > 0:
            factor = self.safety_factor * (self.tolerance / error) ** 0.2
            self.current_dt = np.clip(h * factor, self.min_dt, self.max_dt)
        
        # Use fifth-order result
        return pos_5th, vel_5th
    
    def _rk4_step(self, positions, velocities, accelerations, h):
        """Fourth-order Runge-Kutta step"""
        k1_pos = velocities
        k1_vel = accelerations
        
        k2_pos = velocities + 0.5 * k1_vel * h
        k2_vel = accelerations
        
        k3_pos = velocities + 0.5 * k2_vel * h
        k3_vel = accelerations
        
        k4_pos = velocities + k3_vel * h
        k4_vel = accelerations
        
        new_pos = positions + (h / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        new_vel = velocities + (h / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        
        return new_pos, new_vel
    
    def _rk5_step(self, positions, velocities, accelerations, h):
        """Fifth-order step (simplified)"""
        # This is a simplified fifth-order step
        # For full RKF45, need complete coefficient tables
        k1_pos = velocities
        k1_vel = accelerations
        
        k2_pos = velocities + 0.25 * k1_vel * h
        k2_vel = accelerations
        
        k3_pos = velocities + 0.375 * k2_vel * h
        k3_vel = accelerations
        
        k4_pos = velocities + (12.0/13.0) * k3_vel * h
        k4_vel = accelerations
        
        k5_pos = velocities + k4_vel * h
        k5_vel = accelerations
        
        new_pos = positions + h * (16.0/135.0 * k1_pos + 6656.0/12825.0 * k3_pos + 
                                  28561.0/56430.0 * k4_pos - 9.0/50.0 * k5_pos)
        new_vel = velocities + h * (16.0/135.0 * k1_vel + 6656.0/12825.0 * k3_vel + 
                                  28561.0/56430.0 * k4_vel - 9.0/50.0 * k5_vel)
        
        return new_pos, new_vel


class LeapfrogIntegrator(Integrator):
    """
    Leapfrog integrator (symplectic, second-order)
    Good for conservative systems with energy conservation
    """
    
    def __init__(self):
        self.half_step_velocities = None
        self.first_step = True
        self.logger = logging.getLogger(__name__)
    
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog integration algorithm
        
        v(t+dt/2) = v(t-dt/2) + a(t)*dt
        x(t+dt) = x(t) + v(t+dt/2)*dt
        """
        accelerations = forces / masses.reshape(-1, 1)
        
        if self.first_step:
            # Initialize half-step velocities
            self.half_step_velocities = velocities - 0.5 * accelerations * dt
            self.first_step = False
        
        # Update half-step velocities
        new_half_step_velocities = self.half_step_velocities + accelerations * dt
        
        # Update positions using half-step velocities
        new_positions = positions + new_half_step_velocities * dt
        
        # Calculate full-step velocities for output
        new_velocities = new_half_step_velocities + 0.5 * accelerations * dt
        
        # Store half-step velocities for next iteration
        self.half_step_velocities = new_half_step_velocities
        
        return new_positions, new_velocities
    
    def reset(self):
        """Reset integrator state"""
        self.half_step_velocities = None
        self.first_step = True


class BerendsenThermostat:
    """
    Berendsen thermostat for temperature control
    Can be combined with any integrator
    """
    
    def __init__(self, target_temperature: float, coupling_time: float = 1e-12):
        """
        Initialize Berendsen thermostat
        
        Args:
            target_temperature: Target temperature (K)
            coupling_time: Coupling time constant (s)
        """
        self.target_temperature = target_temperature
        self.coupling_time = coupling_time
        self.kb = 1.380649e-23  # Boltzmann constant
        
    def apply_thermostat(self, 
                        velocities: np.ndarray,
                        masses: np.ndarray,
                        dt: float) -> np.ndarray:
        """
        Apply Berendsen thermostat to velocities
        
        Args:
            velocities: Current velocities (N, 3)
            masses: Particle masses (N,)
            dt: Time step
            
        Returns:
            Thermostatted velocities
        """
        # Calculate current kinetic energy
        kinetic_energy = 0.5 * np.sum(masses.reshape(-1, 1) * velocities**2)
        
        # Calculate current temperature (3N degrees of freedom)
        n_particles = len(masses)
        current_temperature = (2 * kinetic_energy) / (3 * n_particles * self.kb)
        
        if current_temperature <= 0:
            return velocities
        
        # Berendsen scaling factor
        scaling_factor = np.sqrt(1 + (dt / self.coupling_time) * 
                               (self.target_temperature / current_temperature - 1))
        
        return velocities * scaling_factor


class IntegratorFactory:
    """Factory class for creating integrators"""
    
    @staticmethod
    def create_integrator(integrator_type: str, **kwargs) -> Integrator:
        """
        Create integrator instance
        
        Args:
            integrator_type: Type of integrator ('verlet', 'rk4', 'adaptive', 'leapfrog', 'euler')
            **kwargs: Additional arguments for integrator
            
        Returns:
            Integrator instance
        """
        integrator_type = integrator_type.lower()
        
        if integrator_type == 'verlet' or integrator_type == 'velocity_verlet':
            return VelocityVerletIntegrator()
        elif integrator_type == 'rk4' or integrator_type == 'runge_kutta':
            return RungeKutta4Integrator(**kwargs)
        elif integrator_type == 'adaptive':
            return AdaptiveIntegrator(**kwargs)
        elif integrator_type == 'leapfrog':
            return LeapfrogIntegrator()
        elif integrator_type == 'euler':
            return EulerIntegrator()
        else:
            raise ValueError(f"Unknown integrator type: {integrator_type}")
    
    @staticmethod
    def get_available_integrators() -> list:
        """Get list of available integrator types"""
        return ['verlet', 'rk4', 'adaptive', 'leapfrog', 'euler']


class IntegratorWithThermostat:
    """
    Wrapper class combining an integrator with a thermostat
    """
    
    def __init__(self, 
                 integrator: Integrator,
                 thermostat: BerendsenThermostat = None):
        """
        Initialize integrator with optional thermostat
        
        Args:
            integrator: Base integrator
            thermostat: Optional thermostat for temperature control
        """
        self.integrator = integrator
        self.thermostat = thermostat
        self.logger = logging.getLogger(__name__)
    
    def integrate(self, 
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 forces: np.ndarray,
                 masses: np.ndarray,
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform integration with optional thermostating
        """
        # Apply thermostat before integration
        if self.thermostat is not None:
            velocities = self.thermostat.apply_thermostat(velocities, masses, dt)
        
        # Perform integration
        new_positions, new_velocities = self.integrator.integrate(
            positions, velocities, forces, masses, dt
        )
        
        # Apply thermostat after integration
        if self.thermostat is not None:
            new_velocities = self.thermostat.apply_thermostat(new_velocities, masses, dt)
        
        return new_positions, new_velocities
    
    def reset(self):
        """Reset integrator state"""
        if hasattr(self.integrator, 'reset'):
            self.integrator.reset()


def calculate_system_energy(positions: np.ndarray,
                           velocities: np.ndarray,
                           masses: np.ndarray,
                           forces: np.ndarray) -> dict:
    """
    Calculate total system energy for monitoring conservation
    
    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        forces: Current forces (N, 3)
        
    Returns:
        Dictionary with energy components
    """
    # Kinetic energy
    kinetic_energy = 0.5 * np.sum(masses.reshape(-1, 1) * velocities**2)
    
    # Potential energy (rough estimate from forces)
    # This is approximate - true potential energy calculation needs force calculator
    potential_energy = -np.sum(forces * positions)  # Very rough approximation
    
    total_energy = kinetic_energy + potential_energy
    
    return {
        'kinetic': kinetic_energy,
        'potential': potential_energy,
        'total': total_energy,
        'temperature_estimate': (2 * kinetic_energy) / (3 * len(masses) * 1.380649e-23)
    }