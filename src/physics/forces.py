"""
Force calculation module for MagNanoBridge
Implements magnetic forces, dipole interactions, drag, and collisions
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .particles import Particle


@dataclass
class ForceComponents:
    """Container for different force components acting on particles"""
    magnetic: np.ndarray      # Magnetic gradient force
    dipole: np.ndarray       # Dipole-dipole interactions
    drag: np.ndarray         # Fluid drag force
    collision: np.ndarray    # Collision forces
    brownian: np.ndarray     # Brownian motion force
    total: np.ndarray        # Sum of all forces


class ForceCalculator:
    """
    Calculates all forces acting on magnetic nanoparticles
    """
    
    # Physical constants
    KB = 1.380649e-23  # Boltzmann constant (J/K)
    MU_0 = 4e-7 * np.pi  # Permeability of free space (H/m)
    
    def __init__(self, 
                 enable_dipole: bool = True,
                 enable_brownian: bool = True,
                 temperature: float = 310.0,  # Body temperature (K)
                 fluid_viscosity: float = 1e-3):  # Water viscosity (Pa⋅s)
        """
        Initialize force calculator
        
        Args:
            enable_dipole: Enable dipole-dipole interactions
            enable_brownian: Enable Brownian motion
            temperature: Temperature for thermal effects (K)
            fluid_viscosity: Fluid viscosity (Pa⋅s)
        """
        self.enable_dipole = enable_dipole
        self.enable_brownian = enable_brownian
        self.temperature = temperature
        self.fluid_viscosity = fluid_viscosity
        
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization parameters
        self.dipole_cutoff_distance = 1e-3  # 1mm cutoff for dipole interactions
        self.use_fast_multipole = False  # Enable for >1000 particles
        
        self.logger.info(f"ForceCalculator initialized: dipole={enable_dipole}, "
                        f"brownian={enable_brownian}, T={temperature}K")
    
    def calculate_total_forces(self, 
                             particles: List[Particle],
                             positions: np.ndarray,
                             velocities: np.ndarray,
                             magnetic_field: np.ndarray,
                             field_gradient: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Calculate total forces on all particles
        
        Args:
            particles: List of Particle objects
            positions: Particle positions (N, 3)
            velocities: Particle velocities (N, 3)  
            magnetic_field: Magnetic field at each position (N, 3)
            field_gradient: Field gradient tensor at each position (N, 3, 3)
            dt: Time step for stochastic forces
            
        Returns:
            Total forces on all particles (N, 3)
        """
        n_particles = len(particles)
        
        # Initialize force components
        force_components = ForceComponents(
            magnetic=np.zeros((n_particles, 3)),
            dipole=np.zeros((n_particles, 3)),
            drag=np.zeros((n_particles, 3)),
            collision=np.zeros((n_particles, 3)),
            brownian=np.zeros((n_particles, 3)),
            total=np.zeros((n_particles, 3))
        )
        
        # 1. Magnetic gradient force
        force_components.magnetic = self._calculate_magnetic_gradient_force(
            particles, magnetic_field, field_gradient
        )
        
        # 2. Dipole-dipole interactions
        if self.enable_dipole:
            force_components.dipole = self._calculate_dipole_forces(
                particles, positions
            )
        
        # 3. Fluid drag force
        force_components.drag = self._calculate_drag_force(
            particles, velocities
        )
        
        # 4. Collision forces
        force_components.collision = self._calculate_collision_forces(
            particles, positions
        )
        
        # 5. Brownian motion
        if self.enable_brownian:
            force_components.brownian = self._calculate_brownian_force(
                particles, dt
            )
        
        # Sum all forces
        force_components.total = (force_components.magnetic + 
                                force_components.dipole +
                                force_components.drag +
                                force_components.collision +
                                force_components.brownian)
        
        return force_components.total
    
    def _calculate_magnetic_gradient_force(self, 
                                         particles: List[Particle],
                                         magnetic_field: np.ndarray,
                                         field_gradient: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic gradient force: F = ∇(m⋅B)
        
        Args:
            particles: List of particles
            magnetic_field: B field at each particle (N, 3)
            field_gradient: ∇B tensor at each particle (N, 3, 3)
            
        Returns:
            Magnetic forces (N, 3)
        """
        n_particles = len(particles)
        magnetic_forces = np.zeros((n_particles, 3))
        
        for i, particle in enumerate(particles):
            # Magnetic moment vector
            m = particle.magnetic_moment
            
            # Force = ∇(m⋅B) = (m⋅∇)B
            # where ∇B is the gradient tensor [∂Bj/∂xi]
            
            # For each force component: Fi = Σj mj * ∂Bj/∂xi
            for force_idx in range(3):  # x, y, z components of force
                force_component = 0.0
                for moment_idx in range(3):  # x, y, z components of moment
                    force_component += m[moment_idx] * field_gradient[i, moment_idx, force_idx]
                magnetic_forces[i, force_idx] = force_component
        
        return magnetic_forces
    
    def _calculate_dipole_forces(self, 
                               particles: List[Particle],
                               positions: np.ndarray) -> np.ndarray:
        """
        Calculate dipole-dipole interaction forces
        
        Args:
            particles: List of particles
            positions: Particle positions (N, 3)
            
        Returns:
            Dipole forces (N, 3)
        """
        n_particles = len(particles)
        dipole_forces = np.zeros((n_particles, 3))
        
        if self.use_fast_multipole and n_particles > 1000:
            # Use Fast Multipole Method for large systems
            return self._calculate_dipole_forces_fmm(particles, positions)
        
        # Direct N² calculation for smaller systems
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # Vector from particle j to particle i
                r_vec = positions[i] - positions[j]
                r_mag = np.linalg.norm(r_vec)
                
                # Skip if particles are too close (avoid singularity)
                if r_mag < 2 * (particles[i].total_radius + particles[j].total_radius):
                    continue
                
                # Skip if beyond cutoff distance
                if r_mag > self.dipole_cutoff_distance:
                    continue
                
                # Unit vector
                r_hat = r_vec / r_mag
                
                # Magnetic moments
                m_i = particles[i].magnetic_moment
                m_j = particles[j].magnetic_moment
                
                # Dipole-dipole force calculation
                # F_ij = (3μ₀/4π) * (1/r⁴) * [3(m_i⋅r̂)(m_j⋅r̂)r̂ - m_i(m_j⋅r̂) - m_j(m_i⋅r̂) - (m_i⋅m_j)r̂ + 5(m_i⋅r̂)(m_j⋅r̂)r̂]
                
                # Dot products
                mi_dot_r = np.dot(m_i, r_hat)
                mj_dot_r = np.dot(m_j, r_hat)
                mi_dot_mj = np.dot(m_i, m_j)
                
                # Force prefactor
                prefactor = (3 * self.MU_0) / (4 * np.pi * r_mag**4)
                
                # Force components
                force_ij = prefactor * (
                    3 * mi_dot_r * mj_dot_r * r_hat -
                    m_i * mj_dot_r -
                    m_j * mi_dot_r -
                    mi_dot_mj * r_hat +
                    5 * mi_dot_r * mj_dot_r * r_hat
                )
                
                # Apply Newton's third law
                dipole_forces[i] += force_ij
                dipole_forces[j] -= force_ij
        
        return dipole_forces
    
    def _calculate_dipole_forces_fmm(self, 
                                   particles: List[Particle],
                                   positions: np.ndarray) -> np.ndarray:
        """
        Fast Multipole Method for dipole-dipole forces (simplified implementation)
        
        Args:
            particles: List of particles
            positions: Particle positions (N, 3)
            
        Returns:
            Dipole forces (N, 3)
        """
        # This is a simplified FMM implementation
        # For production use, consider using specialized libraries like FMM3D
        
        n_particles = len(particles)
        dipole_forces = np.zeros((n_particles, 3))
        
        # For now, use hierarchical grouping with distance cutoffs
        # Group particles into spatial cells
        cell_size = self.dipole_cutoff_distance / 2
        
        # Find bounding box
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        
        # Create grid
        grid_dims = np.ceil((max_pos - min_pos) / cell_size).astype(int) + 1
        
        # Assign particles to cells
        particle_cells = {}
        for i, pos in enumerate(positions):
            cell_idx = tuple(((pos - min_pos) / cell_size).astype(int))
            if cell_idx not in particle_cells:
                particle_cells[cell_idx] = []
            particle_cells[cell_idx].append(i)
        
        # Calculate forces only between nearby cells
        for cell_idx, particle_list in particle_cells.items():
            # Check neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_idx = (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz)
                        
                        if neighbor_idx in particle_cells:
                            neighbor_list = particle_cells[neighbor_idx]
                            
                            # Calculate forces between particles in these cells
                            for i in particle_list:
                                for j in neighbor_list:
                                    if i >= j:  # Avoid double counting
                                        continue
                                    
                                    # Calculate dipole force (same as direct method)
                                    r_vec = positions[i] - positions[j]
                                    r_mag = np.linalg.norm(r_vec)
                                    
                                    if (r_mag < 2 * (particles[i].total_radius + particles[j].total_radius) or
                                        r_mag > self.dipole_cutoff_distance):
                                        continue
                                    
                                    r_hat = r_vec / r_mag
                                    m_i = particles[i].magnetic_moment
                                    m_j = particles[j].magnetic_moment
                                    
                                    mi_dot_r = np.dot(m_i, r_hat)
                                    mj_dot_r = np.dot(m_j, r_hat)
                                    mi_dot_mj = np.dot(m_i, m_j)
                                    
                                    prefactor = (3 * self.MU_0) / (4 * np.pi * r_mag**4)
                                    
                                    force_ij = prefactor * (
                                        3 * mi_dot_r * mj_dot_r * r_hat -
                                        m_i * mj_dot_r -
                                        m_j * mi_dot_r -
                                        mi_dot_mj * r_hat +
                                        5 * mi_dot_r * mj_dot_r * r_hat
                                    )
                                    
                                    dipole_forces[i] += force_ij
                                    dipole_forces[j] -= force_ij
        
        return dipole_forces
    
    def _calculate_drag_force(self, 
                            particles: List[Particle],
                            velocities: np.ndarray) -> np.ndarray:
        """
        Calculate fluid drag force using Stokes' law
        
        Args:
            particles: List of particles
            velocities: Particle velocities (N, 3)
            
        Returns:
            Drag forces (N, 3)
        """
        n_particles = len(particles)
        drag_forces = np.zeros((n_particles, 3))
        
        for i, particle in enumerate(particles):
            # Stokes drag: F_drag = -6πηRv
            # where η is viscosity, R is particle radius, v is velocity
            drag_coefficient = 6 * np.pi * self.fluid_viscosity * particle.total_radius
            drag_forces[i] = -drag_coefficient * velocities[i]
        
        return drag_forces
    
    def _calculate_collision_forces(self, 
                                  particles: List[Particle],
                                  positions: np.ndarray) -> np.ndarray:
        """
        Calculate collision forces using Lennard-Jones potential
        
        Args:
            particles: List of particles
            positions: Particle positions (N, 3)
            
        Returns:
            Collision forces (N, 3)
        """
        n_particles = len(particles)
        collision_forces = np.zeros((n_particles, 3))
        
        # Lennard-Jones parameters
        epsilon = 1e-20  # Energy scale (J)
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                r_vec = positions[i] - positions[j]
                r_mag = np.linalg.norm(r_vec)
                
                # Contact distance (sum of total radii)
                sigma = particles[i].total_radius + particles[j].total_radius
                
                # Only calculate if particles are close
                if r_mag > 2 * sigma:
                    continue
                
                # Avoid singularity
                if r_mag < 0.1 * sigma:
                    r_mag = 0.1 * sigma
                
                # Unit vector
                r_hat = r_vec / r_mag
                
                # Lennard-Jones force: F = 24ε/r * [(σ/r)⁶ - 2(σ/r)¹²]
                sigma_over_r = sigma / r_mag
                sigma6 = sigma_over_r**6
                sigma12 = sigma6**2
                
                force_magnitude = 24 * epsilon / r_mag * (sigma6 - 2 * sigma12)
                force_vec = force_magnitude * r_hat
                
                # Apply Newton's third law
                collision_forces[i] += force_vec
                collision_forces[j] -= force_vec
        
        return collision_forces
    
    def _calculate_brownian_force(self, 
                                particles: List[Particle],
                                dt: float) -> np.ndarray:
        """
        Calculate Brownian force using Langevin dynamics
        
        Args:
            particles: List of particles
            dt: Time step
            
        Returns:
            Brownian forces (N, 3)
        """
        n_particles = len(particles)
        brownian_forces = np.zeros((n_particles, 3))
        
        for i, particle in enumerate(particles):
            # Drag coefficient
            gamma = 6 * np.pi * self.fluid_viscosity * particle.total_radius
            
            # Thermal force amplitude
            # From fluctuation-dissipation theorem: <F²> = 2γkBT/dt
            force_amplitude = np.sqrt(2 * gamma * self.KB * self.temperature / dt)
            
            # Random force components
            random_force = np.random.normal(0, force_amplitude, 3)
            brownian_forces[i] = random_force
        
        return brownian_forces
    
    def calculate_force_components(self, 
                                 particles: List[Particle],
                                 positions: np.ndarray,
                                 velocities: np.ndarray,
                                 magnetic_field: np.ndarray,
                                 field_gradient: np.ndarray,
                                 dt: float) -> ForceComponents:
        """
        Calculate individual force components (for analysis)
        
        Returns:
            ForceComponents object with all force types
        """
        n_particles = len(particles)
        
        force_components = ForceComponents(
            magnetic=self._calculate_magnetic_gradient_force(particles, magnetic_field, field_gradient),
            dipole=self._calculate_dipole_forces(particles, positions) if self.enable_dipole else np.zeros((n_particles, 3)),
            drag=self._calculate_drag_force(particles, velocities),
            collision=self._calculate_collision_forces(particles, positions),
            brownian=self._calculate_brownian_force(particles, dt) if self.enable_brownian else np.zeros((n_particles, 3)),
            total=np.zeros((n_particles, 3))
        )
        
        force_components.total = (force_components.magnetic + 
                                force_components.dipole +
                                force_components.drag +
                                force_components.collision +
                                force_components.brownian)
        
        return force_components
    
    def get_force_statistics(self, forces: np.ndarray) -> Dict[str, Any]:
        """
        Calculate force statistics for analysis
        
        Args:
            forces: Force array (N, 3)
            
        Returns:
            Dictionary with force statistics
        """
        force_magnitudes = np.linalg.norm(forces, axis=1)
        
        return {
            'mean_force_magnitude': np.mean(force_magnitudes),
            'max_force_magnitude': np.max(force_magnitudes),
            'std_force_magnitude': np.std(force_magnitudes),
            'total_force_vector': np.sum(forces, axis=0).tolist(),
            'force_components': {
                'x': {'mean': np.mean(forces[:, 0]), 'std': np.std(forces[:, 0])},
                'y': {'mean': np.mean(forces[:, 1]), 'std': np.std(forces[:, 1])},
                'z': {'mean': np.mean(forces[:, 2]), 'std': np.std(forces[:, 2])}
            }
        }
    
    def enable_fast_multipole_method(self, enable: bool = True):
        """Enable or disable Fast Multipole Method for dipole calculations"""
        self.use_fast_multipole = enable
        self.logger.info(f"Fast Multipole Method: {'enabled' if enable else 'disabled'}")
    
    def set_cutoff_distance(self, distance: float):
        """Set cutoff distance for dipole interactions"""
        self.dipole_cutoff_distance = distance
        self.logger.info(f"Dipole cutoff distance set to {distance*1e3:.2f} mm")
    
    def update_parameters(self, **kwargs):
        """Update force calculation parameters"""
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        if 'fluid_viscosity' in kwargs:
            self.fluid_viscosity = kwargs['fluid_viscosity']
        if 'enable_dipole' in kwargs:
            self.enable_dipole = kwargs['enable_dipole']
        if 'enable_brownian' in kwargs:
            self.enable_brownian = kwargs['enable_brownian']
        
        self.logger.info(f"Force parameters updated: T={self.temperature}K, "
                        f"η={self.fluid_viscosity:.2e} Pa⋅s")