"""
Particle system implementation for MagNanoBridge
Handles individual particles and collections with physical properties
"""

import numpy as np
import uuid
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Particle:
    """
    Individual magnetic nanoparticle with physical properties
    """
    id: str
    position: np.ndarray  # 3D position [x, y, z]
    velocity: np.ndarray  # 3D velocity [vx, vy, vz]
    force: np.ndarray     # 3D force [fx, fy, fz]
    
    # Physical properties
    radius: float         # Core radius (m)
    mass: float          # Total mass (kg)
    shell_thickness: float # Plastic shell thickness (m)
    magnetic_moment: np.ndarray  # Magnetic dipole moment vector (A⋅m²)
    
    # Material properties
    density: float = 7800.0  # Iron oxide density (kg/m³)
    shell_density: float = 1200.0  # Polymer shell density (kg/m³)
    
    def __post_init__(self):
        """Initialize derived properties"""
        # Ensure arrays are numpy arrays
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)
        if not isinstance(self.force, np.ndarray):
            self.force = np.array(self.force, dtype=np.float64)
        if not isinstance(self.magnetic_moment, np.ndarray):
            self.magnetic_moment = np.array(self.magnetic_moment, dtype=np.float64)
    
    @property
    def total_radius(self) -> float:
        """Total radius including shell"""
        return self.radius + self.shell_thickness
    
    @property
    def core_volume(self) -> float:
        """Volume of magnetic core"""
        return (4.0 / 3.0) * np.pi * self.radius**3
    
    @property
    def shell_volume(self) -> float:
        """Volume of plastic shell"""
        total_vol = (4.0 / 3.0) * np.pi * self.total_radius**3
        return total_vol - self.core_volume
    
    @property
    def moment_magnitude(self) -> float:
        """Magnitude of magnetic moment"""
        return np.linalg.norm(self.magnetic_moment)
    
    def update_magnetic_moment(self, external_field: np.ndarray, susceptibility: float = 1.0):
        """
        Update magnetic moment based on external field
        
        Args:
            external_field: External magnetic field vector (T)
            susceptibility: Magnetic susceptibility
        """
        if np.linalg.norm(external_field) > 0:
            # Align moment with field (simplified model)
            field_direction = external_field / np.linalg.norm(external_field)
            moment_magnitude = self.moment_magnitude
            self.magnetic_moment = moment_magnitude * field_direction * susceptibility
    
    def get_dict(self) -> Dict[str, Any]:
        """Convert particle to dictionary for serialization"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'force': self.force.tolist(),
            'radius': self.radius,
            'mass': self.mass,
            'shell_thickness': self.shell_thickness,
            'magnetic_moment': self.magnetic_moment.tolist(),
            'total_radius': self.total_radius,
            'core_volume': self.core_volume,
            'shell_volume': self.shell_volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Particle':
        """Create particle from dictionary"""
        return cls(
            id=data['id'],
            position=np.array(data['position']),
            velocity=np.array(data['velocity']),
            force=np.array(data['force']),
            radius=data['radius'],
            mass=data['mass'],
            shell_thickness=data['shell_thickness'],
            magnetic_moment=np.array(data['magnetic_moment'])
        )


class ParticleSystem:
    """
    Collection of particles with system-level operations
    """
    
    def __init__(self, 
                 num_particles: int = 100,
                 particle_radius: float = 50e-9,  # 50 nm
                 particle_mass: float = None,
                 shell_thickness: float = 10e-9,  # 10 nm
                 magnetic_moment: float = 1e-18):  # A⋅m²
        """
        Initialize particle system
        
        Args:
            num_particles: Number of particles to create
            particle_radius: Core radius of particles (m)
            particle_mass: Mass per particle (kg), calculated if None
            shell_thickness: Thickness of plastic shell (m)
            magnetic_moment: Magnetic moment magnitude (A⋅m²)
        """
        self.num_particles = num_particles
        self.particle_radius = particle_radius
        self.shell_thickness = shell_thickness
        self.magnetic_moment_magnitude = magnetic_moment
        
        # Calculate mass if not provided
        if particle_mass is None:
            core_volume = (4.0 / 3.0) * np.pi * particle_radius**3
            total_radius = particle_radius + shell_thickness
            total_volume = (4.0 / 3.0) * np.pi * total_radius**3
            shell_volume = total_volume - core_volume
            
            # Iron oxide core + polymer shell
            particle_mass = (core_volume * 7800.0 + shell_volume * 1200.0)
        
        self.particle_mass = particle_mass
        self.particles: List[Particle] = []
        
        # Initial configuration
        self.initial_positions = None
        self.initial_velocities = None
        
        # Create particles
        self._initialize_particles()
    
    def _initialize_particles(self):
        """Create initial particle configuration"""
        self.particles = []
        
        # Generate initial positions (random distribution in a box)
        positions = self._generate_initial_positions()
        velocities = self._generate_initial_velocities()
        
        for i in range(self.num_particles):
            # Random magnetic moment direction (initially random)
            moment_direction = np.random.randn(3)
            moment_direction /= np.linalg.norm(moment_direction)
            magnetic_moment = self.magnetic_moment_magnitude * moment_direction
            
            particle = Particle(
                id=str(uuid.uuid4()),
                position=positions[i],
                velocity=velocities[i],
                force=np.zeros(3),
                radius=self.particle_radius,
                mass=self.particle_mass,
                shell_thickness=self.shell_thickness,
                magnetic_moment=magnetic_moment
            )
            
            self.particles.append(particle)
        
        # Store initial state for reset
        self.initial_positions = positions.copy()
        self.initial_velocities = velocities.copy()
    
    def _generate_initial_positions(self) -> np.ndarray:
        """Generate initial particle positions"""
        # Start particles in a small region away from focus zone
        box_size = np.array([2e-3, 2e-3, 2e-3])  # 2mm cube
        offset = np.array([0, 0, -5e-3])  # 5mm below focus zone
        
        positions = np.random.uniform(-box_size/2, box_size/2, (self.num_particles, 3))
        positions += offset
        
        # Ensure no overlaps
        min_distance = 2.5 * (self.particle_radius + self.shell_thickness)
        
        for i in range(self.num_particles):
            for j in range(i):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_distance:
                    # Move particle i away from j
                    direction = positions[i] - positions[j]
                    if np.linalg.norm(direction) > 0:
                        direction /= np.linalg.norm(direction)
                        positions[i] = positions[j] + direction * min_distance
        
        return positions
    
    def _generate_initial_velocities(self) -> np.ndarray:
        """Generate initial particle velocities"""
        # Small random thermal velocities
        thermal_velocity = 1e-6  # m/s
        velocities = np.random.normal(0, thermal_velocity, (self.num_particles, 3))
        return velocities
    
    def get_positions(self) -> np.ndarray:
        """Get all particle positions as numpy array"""
        return np.array([p.position for p in self.particles])
    
    def get_velocities(self) -> np.ndarray:
        """Get all particle velocities as numpy array"""
        return np.array([p.velocity for p in self.particles])
    
    def get_forces(self) -> np.ndarray:
        """Get all particle forces as numpy array"""
        return np.array([p.force for p in self.particles])
    
    def get_masses(self) -> np.ndarray:
        """Get all particle masses as numpy array"""
        return np.array([p.mass for p in self.particles])
    
    def get_radii(self) -> np.ndarray:
        """Get all particle core radii as numpy array"""
        return np.array([p.radius for p in self.particles])
    
    def get_total_radii(self) -> np.ndarray:
        """Get all particle total radii (including shell) as numpy array"""
        return np.array([p.total_radius for p in self.particles])
    
    def get_magnetic_moments(self) -> np.ndarray:
        """Get all particle magnetic moments as numpy array"""
        return np.array([p.magnetic_moment for p in self.particles])
    
    def set_positions(self, positions: np.ndarray):
        """Set all particle positions from numpy array"""
        for i, particle in enumerate(self.particles):
            particle.position = positions[i].copy()
    
    def set_velocities(self, velocities: np.ndarray):
        """Set all particle velocities from numpy array"""
        for i, particle in enumerate(self.particles):
            particle.velocity = velocities[i].copy()
    
    def set_forces(self, forces: np.ndarray):
        """Set all particle forces from numpy array"""
        for i, particle in enumerate(self.particles):
            particle.force = forces[i].copy()
    
    def update_magnetic_moments(self, external_field: np.ndarray, susceptibility: float = 1.0):
        """
        Update all particle magnetic moments based on external field
        
        Args:
            external_field: External field at each particle position (N, 3)
            susceptibility: Magnetic susceptibility
        """
        for i, particle in enumerate(self.particles):
            particle.update_magnetic_moment(external_field[i], susceptibility)
    
    def get_particle_by_id(self, particle_id: str) -> Optional[Particle]:
        """Get particle by ID"""
        for particle in self.particles:
            if particle.id == particle_id:
                return particle
        return None
    
    def get_particles_in_region(self, center: np.ndarray, radius: float) -> List[Particle]:
        """Get particles within a spherical region"""
        particles_in_region = []
        for particle in self.particles:
            distance = np.linalg.norm(particle.position - center)
            if distance <= radius:
                particles_in_region.append(particle)
        return particles_in_region
    
    def calculate_center_of_mass(self) -> np.ndarray:
        """Calculate center of mass of all particles"""
        total_mass = 0.0
        com = np.zeros(3)
        
        for particle in self.particles:
            total_mass += particle.mass
            com += particle.mass * particle.position
        
        if total_mass > 0:
            com /= total_mass
        
        return com
    
    def calculate_total_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of the system"""
        total_ke = 0.0
        for particle in self.particles:
            v_squared = np.dot(particle.velocity, particle.velocity)
            total_ke += 0.5 * particle.mass * v_squared
        return total_ke
    
    def calculate_total_magnetic_moment(self) -> np.ndarray:
        """Calculate total magnetic moment of the system"""
        total_moment = np.zeros(3)
        for particle in self.particles:
            total_moment += particle.magnetic_moment
        return total_moment
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        positions = self.get_positions()
        velocities = self.get_velocities()
        
        return {
            'num_particles': len(self.particles),
            'center_of_mass': self.calculate_center_of_mass().tolist(),
            'total_kinetic_energy': self.calculate_total_kinetic_energy(),
            'total_magnetic_moment': self.calculate_total_magnetic_moment().tolist(),
            'position_stats': {
                'mean': np.mean(positions, axis=0).tolist(),
                'std': np.std(positions, axis=0).tolist(),
                'min': np.min(positions, axis=0).tolist(),
                'max': np.max(positions, axis=0).tolist()
            },
            'velocity_stats': {
                'mean': np.mean(velocities, axis=0).tolist(),
                'std': np.std(velocities, axis=0).tolist(),
                'mean_speed': np.mean(np.linalg.norm(velocities, axis=1)),
                'max_speed': np.max(np.linalg.norm(velocities, axis=1))
            },
            'particle_properties': {
                'core_radius': self.particle_radius,
                'shell_thickness': self.shell_thickness,
                'total_radius': self.particle_radius + self.shell_thickness,
                'mass': self.particle_mass,
                'magnetic_moment_magnitude': self.magnetic_moment_magnitude
            }
        }
    
    def export_particle_data(self) -> List[Dict[str, Any]]:
        """Export all particle data as list of dictionaries"""
        return [particle.get_dict() for particle in self.particles]
    
    def load_particle_data(self, particle_data: List[Dict[str, Any]]):
        """Load particle data from list of dictionaries"""
        self.particles = [Particle.from_dict(data) for data in particle_data]
        self.num_particles = len(self.particles)
    
    def reset(self):
        """Reset particles to initial configuration"""
        if self.initial_positions is not None and self.initial_velocities is not None:
            self.set_positions(self.initial_positions.copy())
            self.set_velocities(self.initial_velocities.copy())
            
            # Reset forces
            self.set_forces(np.zeros((self.num_particles, 3)))
            
            # Reset magnetic moments to random orientations
            for particle in self.particles:
                moment_direction = np.random.randn(3)
                moment_direction /= np.linalg.norm(moment_direction)
                particle.magnetic_moment = self.magnetic_moment_magnitude * moment_direction
    
    def add_particle(self, position: np.ndarray, velocity: np.ndarray = None, 
                    magnetic_moment: np.ndarray = None) -> str:
        """
        Add a new particle to the system
        
        Args:
            position: 3D position
            velocity: 3D velocity (default: zero)
            magnetic_moment: 3D magnetic moment (default: random)
            
        Returns:
            Particle ID
        """
        if velocity is None:
            velocity = np.zeros(3)
        
        if magnetic_moment is None:
            moment_direction = np.random.randn(3)
            moment_direction /= np.linalg.norm(moment_direction)
            magnetic_moment = self.magnetic_moment_magnitude * moment_direction
        
        particle = Particle(
            id=str(uuid.uuid4()),
            position=position.copy(),
            velocity=velocity.copy(),
            force=np.zeros(3),
            radius=self.particle_radius,
            mass=self.particle_mass,
            shell_thickness=self.shell_thickness,
            magnetic_moment=magnetic_moment.copy()
        )
        
        self.particles.append(particle)
        self.num_particles += 1
        
        return particle.id
    
    def remove_particle(self, particle_id: str) -> bool:
        """
        Remove a particle from the system
        
        Args:
            particle_id: ID of particle to remove
            
        Returns:
            True if particle was found and removed
        """
        for i, particle in enumerate(self.particles):
            if particle.id == particle_id:
                self.particles.pop(i)
                self.num_particles -= 1
                return True
        return False
    
    def check_collisions(self) -> List[Tuple[int, int, float]]:
        """
        Check for particle collisions
        
        Returns:
            List of (particle1_index, particle2_index, overlap_distance) tuples
        """
        collisions = []
        positions = self.get_positions()
        total_radii = self.get_total_radii()
        
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                distance = np.linalg.norm(positions[i] - positions[j])
                contact_distance = total_radii[i] + total_radii[j]
                
                if distance < contact_distance:
                    overlap = contact_distance - distance
                    collisions.append((i, j, overlap))
        
        return collisions
    
    def apply_size_distribution(self, mean_radius: float, std_radius: float):
        """
        Apply a size distribution to particles
        
        Args:
            mean_radius: Mean core radius
            std_radius: Standard deviation of radius
        """
        radii = np.random.normal(mean_radius, std_radius, self.num_particles)
        radii = np.clip(radii, mean_radius * 0.5, mean_radius * 2.0)  # Reasonable bounds
        
        for i, particle in enumerate(self.particles):
            old_radius = particle.radius
            new_radius = radii[i]
            
            # Update radius
            particle.radius = new_radius
            
            # Update mass proportionally (assuming same density)
            volume_ratio = (new_radius / old_radius) ** 3
            particle.mass *= volume_ratio
            
            # Update magnetic moment proportionally
            particle.magnetic_moment *= volume_ratio
    
    def __len__(self) -> int:
        """Return number of particles"""
        return len(self.particles)
    
    def __iter__(self):
        """Make system iterable"""
        return iter(self.particles)
    
    def __getitem__(self, index: int) -> Particle:
        """Allow indexing"""
        return self.particles[index]