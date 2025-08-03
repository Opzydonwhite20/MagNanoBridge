"""
Magnetic field system implementation using Biot-Savart law
Handles multiple coils and realistic field calculations
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class MagneticCoil:
    """
    Represents a circular magnetic coil with physical properties
    """
    id: str
    position: np.ndarray  # Center position [x, y, z] (m)
    radius: float         # Coil radius (m)
    current: float        # Current through coil (A)
    turns: int           # Number of turns
    normal: np.ndarray   # Unit normal vector (coil axis)
    resistance: float = 1.0  # Coil resistance (Ohm)
    max_current: float = 10.0  # Maximum current (A)
    
    def __post_init__(self):
        """Initialize derived properties"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.normal, np.ndarray):
            self.normal = np.array(self.normal, dtype=np.float64)
        
        # Normalize the normal vector
        self.normal = self.normal / np.linalg.norm(self.normal)
    
    @property
    def magnetic_dipole_moment(self) -> float:
        """Calculate magnetic dipole moment of the coil"""
        return self.current * self.turns * np.pi * self.radius**2
    
    def set_current(self, current: float):
        """Set coil current with safety limits"""
        self.current = np.clip(current, -self.max_current, self.max_current)
    
    def get_dict(self) -> Dict[str, Any]:
        """Convert coil to dictionary for serialization"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'radius': self.radius,
            'current': self.current,
            'turns': self.turns,
            'normal': self.normal.tolist(),
            'resistance': self.resistance,
            'max_current': self.max_current,
            'magnetic_dipole_moment': self.magnetic_dipole_moment
        }


class MagneticFieldSystem:
    """
    System of magnetic coils generating fields using Biot-Savart law
    """
    
    # Physical constants
    MU_0 = 4e-7 * np.pi  # Permeability of free space (H/m)
    
    def __init__(self, 
                 coil_positions: List[List[float]] = None,
                 coil_currents: List[float] = None,
                 coil_radii: List[float] = None,
                 coil_turns: List[int] = None):
        """
        Initialize magnetic field system
        
        Args:
            coil_positions: List of [x, y, z] positions for coils
            coil_currents: List of currents for each coil
            coil_radii: List of radii for each coil
            coil_turns: List of number of turns for each coil
        """
        self.coils: List[MagneticCoil] = []
        self.logger = logging.getLogger(__name__)
        
        # Create default configuration if none provided
        if coil_positions is None:
            self._create_default_coil_configuration()
        else:
            self._create_coils_from_parameters(
                coil_positions, coil_currents, coil_radii, coil_turns
            )
        
        self.logger.info(f"Initialized magnetic field system with {len(self.coils)} coils")
    
    def _create_default_coil_configuration(self):
        """Create a default set of coils for bone fracture guidance"""
        # Configuration: 6 coils arranged around the focus zone
        # Focus zone at origin, coils positioned to create gradient
        
        coil_configs = [
            # Top coils (above fracture)
            {'pos': [0, 0, 10e-3], 'radius': 15e-3, 'current': 1.0, 'normal': [0, 0, -1]},
            {'pos': [10e-3, 0, 5e-3], 'radius': 12e-3, 'current': 0.8, 'normal': [-0.7, 0, -0.7]},
            {'pos': [-10e-3, 0, 5e-3], 'radius': 12e-3, 'current': 0.8, 'normal': [0.7, 0, -0.7]},
            
            # Side coils (horizontal gradient)
            {'pos': [15e-3, 0, 0], 'radius': 10e-3, 'current': 0.5, 'normal': [-1, 0, 0]},
            {'pos': [-15e-3, 0, 0], 'radius': 10e-3, 'current': 0.5, 'normal': [1, 0, 0]},
            
            # Bottom coil (below fracture)
            {'pos': [0, 0, -10e-3], 'radius': 15e-3, 'current': -0.3, 'normal': [0, 0, 1]}
        ]
        
        for i, config in enumerate(coil_configs):
            coil = MagneticCoil(
                id=f"coil_{i}",
                position=np.array(config['pos']),
                radius=config['radius'],
                current=config['current'],
                turns=100,  # 100 turns per coil
                normal=np.array(config['normal']),
                max_current=5.0
            )
            self.coils.append(coil)
    
    def _create_coils_from_parameters(self, positions, currents, radii, turns):
        """Create coils from parameter lists"""
        n_coils = len(positions)
        
        # Fill defaults if lists are incomplete
        if currents is None:
            currents = [1.0] * n_coils
        if radii is None:
            radii = [10e-3] * n_coils  # 10mm default
        if turns is None:
            turns = [100] * n_coils
        
        for i in range(n_coils):
            # Default normal pointing toward origin
            pos = np.array(positions[i])
            if np.linalg.norm(pos) > 0:
                normal = -pos / np.linalg.norm(pos)
            else:
                normal = np.array([0, 0, 1])
            
            coil = MagneticCoil(
                id=f"coil_{i}",
                position=pos,
                radius=radii[i] if i < len(radii) else 10e-3,
                current=currents[i] if i < len(currents) else 1.0,
                turns=turns[i] if i < len(turns) else 100,
                normal=normal
            )
            self.coils.append(coil)
    
    def calculate_field_at_point(self, point: np.ndarray, coil: MagneticCoil) -> np.ndarray:
        """
        Calculate magnetic field at a point due to a single coil using Biot-Savart law
        
        Args:
            point: 3D position where to calculate field
            coil: MagneticCoil object
            
        Returns:
            Magnetic field vector [Bx, By, Bz] in Tesla
        """
        # Vector from coil center to field point
        r_vec = point - coil.position
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e-10:  # Avoid singularity at coil center
            return np.zeros(3)
        
        # For circular coil, we use the analytical solution
        # Transform to coil coordinate system where coil is in xy-plane
        
        # Create coordinate system with coil normal as z-axis
        z_axis = coil.normal
        
        # Choose arbitrary x-axis perpendicular to normal
        if abs(z_axis[2]) < 0.9:
            x_axis = np.array([0, 0, 1])
        else:
            x_axis = np.array([1, 0, 0])
        
        # Make x_axis perpendicular to z_axis
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # y_axis completes the right-handed system
        y_axis = np.cross(z_axis, x_axis)
        
        # Transform point to coil coordinate system
        r_local = np.array([
            np.dot(r_vec, x_axis),
            np.dot(r_vec, y_axis),
            np.dot(r_vec, z_axis)
        ])
        
        rho = np.sqrt(r_local[0]**2 + r_local[1]**2)  # Radial distance
        z = r_local[2]  # Axial distance
        
        # Magnetic field calculation for circular coil
        if rho < 1e-10:  # On axis
            # Axial field formula
            factor = (self.MU_0 * coil.current * coil.turns * coil.radius**2) / 2
            denominator = (coil.radius**2 + z**2)**(3/2)
            B_z = factor / denominator
            B_field_local = np.array([0, 0, B_z])
        else:
            # Off-axis field using elliptic integrals (simplified approximation)
            # This is a simplified model - for high accuracy, use complete elliptic integrals
            r_total = np.sqrt(rho**2 + z**2)
            
            # Approximate field using dipole model for far field
            if r_total > 3 * coil.radius:
                # Magnetic dipole approximation
                m = coil.magnetic_dipole_moment * z_axis
                
                # Dipole field formula
                r_hat = r_vec / r_mag
                B_field = (self.MU_0 / (4 * np.pi * r_mag**3)) * (
                    3 * np.dot(m, r_hat) * r_hat - m
                )
                return B_field
            else:
                # Near field - use simplified Biot-Savart integration
                alpha = coil.radius / r_total
                beta = z / r_total
                
                # Simplified field components
                B_rho = (self.MU_0 * coil.current * coil.turns * alpha) / (2 * r_total) * beta
                B_z = (self.MU_0 * coil.current * coil.turns * alpha**2) / (2 * r_total)
                
                # Convert back to Cartesian
                if rho > 0:
                    cos_phi = r_local[0] / rho
                    sin_phi = r_local[1] / rho
                    B_field_local = np.array([
                        B_rho * cos_phi,
                        B_rho * sin_phi,
                        B_z
                    ])
                else:
                    B_field_local = np.array([0, 0, B_z])
        
        # Transform back to global coordinate system
        B_field = (B_field_local[0] * x_axis + 
                  B_field_local[1] * y_axis + 
                  B_field_local[2] * z_axis)
        
        return B_field
    
    def calculate_field_gradient_at_point(self, point: np.ndarray, coil: MagneticCoil) -> np.ndarray:
        """
        Calculate magnetic field gradient at a point due to a single coil
        
        Args:
            point: 3D position where to calculate gradient
            coil: MagneticCoil object
            
        Returns:
            Field gradient tensor (3x3 matrix) [dBi/dxj]
        """
        # Use finite differences for gradient calculation
        delta = 1e-6  # Small displacement for numerical derivative
        gradient = np.zeros((3, 3))
        
        for i in range(3):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += delta
            point_minus[i] -= delta
            
            B_plus = self.calculate_field_at_point(point_plus, coil)
            B_minus = self.calculate_field_at_point(point_minus, coil)
            
            gradient[:, i] = (B_plus - B_minus) / (2 * delta)
        
        return gradient
    
    def calculate_field_and_gradient(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetic field and gradient at multiple points
        
        Args:
            points: Array of 3D positions (N, 3)
            
        Returns:
            Tuple of (fields, gradients) where:
            - fields: (N, 3) array of field vectors
            - gradients: (N, 3, 3) array of gradient tensors
        """
        n_points = points.shape[0]
        total_fields = np.zeros((n_points, 3))
        total_gradients = np.zeros((n_points, 3, 3))
        
        # Sum contributions from all coils
        for coil in self.coils:
            for i, point in enumerate(points):
                field = self.calculate_field_at_point(point, coil)
                gradient = self.calculate_field_gradient_at_point(point, coil)
                
                total_fields[i] += field
                total_gradients[i] += gradient
        
        return total_fields, total_gradients
    
    def update_currents(self, current_adjustments: Dict[str, float]):
        """
        Update coil currents
        
        Args:
            current_adjustments: Dictionary mapping coil_id to current adjustment
        """
        for coil in self.coils:
            if coil.id in current_adjustments:
                new_current = coil.current + current_adjustments[coil.id]
                coil.set_current(new_current)
    
    def set_currents(self, currents: Dict[str, float]):
        """
        Set absolute coil currents
        
        Args:
            currents: Dictionary mapping coil_id to absolute current
        """
        for coil in self.coils:
            if coil.id in currents:
                coil.set_current(currents[coil.id])
    
    def get_coil_data(self) -> List[Dict[str, Any]]:
        """Get all coil data for visualization"""
        return [coil.get_dict() for coil in self.coils]
    
    def get_field_energy(self) -> float:
        """Calculate total magnetic field energy"""
        # Simple approximation based on coil currents
        total_energy = 0.0
        for coil in self.coils:
            # Energy = 0.5 * L * I^2 (approximate inductance)
            inductance = self.MU_0 * coil.turns**2 * coil.radius  # Rough approximation
            total_energy += 0.5 * inductance * coil.current**2
        return total_energy
    
    def calculate_field_at_focus_zone(self, focus_center: np.ndarray, focus_radius: float) -> Dict[str, Any]:
        """
        Calculate field properties at the focus zone
        
        Args:
            focus_center: Center of focus zone
            focus_radius: Radius of focus zone
            
        Returns:
            Dictionary with field statistics
        """
        # Sample points in focus zone
        n_samples = 20
        sample_points = []
        
        # Create spherical sampling
        for i in range(n_samples):
            for j in range(n_samples):
                theta = np.pi * i / (n_samples - 1)
                phi = 2 * np.pi * j / n_samples
                
                r = focus_radius * np.random.uniform(0.1, 1.0)  # Random radius
                point = focus_center + r * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                sample_points.append(point)
        
        sample_points = np.array(sample_points)
        fields, gradients = self.calculate_field_and_gradient(sample_points)
        
        # Calculate statistics
        field_magnitudes = np.linalg.norm(fields, axis=1)
        gradient_norms = np.linalg.norm(gradients.reshape(len(gradients), -1), axis=1)
        
        return {
            'mean_field_magnitude': np.mean(field_magnitudes),
            'std_field_magnitude': np.std(field_magnitudes),
            'max_field_magnitude': np.max(field_magnitudes),
            'min_field_magnitude': np.min(field_magnitudes),
            'mean_gradient_norm': np.mean(gradient_norms),
            'max_gradient_norm': np.max(gradient_norms),
            'field_uniformity': 1.0 - np.std(field_magnitudes) / np.mean(field_magnitudes),
            'sample_points': sample_points.tolist(),
            'field_vectors': fields.tolist()
        }
    
    def reset(self):
        """Reset all coils to default currents"""
        for coil in self.coils:
            # Reset to a reasonable default current
            coil.current = 1.0
    
    def add_coil(self, position: np.ndarray, radius: float, current: float = 0.0,
                turns: int = 100, normal: np.ndarray = None) -> str:
        """
        Add a new coil to the system
        
        Args:
            position: 3D position of coil center
            radius: Coil radius
            current: Initial current
            turns: Number of turns
            normal: Coil normal vector (default: toward origin)
            
        Returns:
            Coil ID
        """
        if normal is None:
            if np.linalg.norm(position) > 0:
                normal = -position / np.linalg.norm(position)
            else:
                normal = np.array([0, 0, 1])
        
        coil_id = f"coil_{len(self.coils)}"
        coil = MagneticCoil(
            id=coil_id,
            position=position.copy(),
            radius=radius,
            current=current,
            turns=turns,
            normal=normal.copy()
        )
        
        self.coils.append(coil)
        self.logger.info(f"Added coil {coil_id} at position {position}")
        
        return coil_id
    
    def remove_coil(self, coil_id: str) -> bool:
        """
        Remove a coil from the system
        
        Args:
            coil_id: ID of coil to remove
            
        Returns:
            True if coil was found and removed
        """
        for i, coil in enumerate(self.coils):
            if coil.id == coil_id:
                self.coils.pop(i)
                self.logger.info(f"Removed coil {coil_id}")
                return True
        return False
    
    def get_coil_by_id(self, coil_id: str) -> Optional[MagneticCoil]:
        """Get coil by ID"""
        for coil in self.coils:
            if coil.id == coil_id:
                return coil
        return None
    
    def optimize_currents_for_focus(self, focus_center: np.ndarray, focus_radius: float,
                                   target_field_strength: float = 0.1) -> Dict[str, float]:
        """
        Optimize coil currents to achieve desired field at focus zone
        
        Args:
            focus_center: Center of focus zone
            focus_radius: Radius of focus zone
            target_field_strength: Target field magnitude (T)
            
        Returns:
            Dictionary of optimized currents
        """
        from scipy.optimize import minimize
        
        def objective(currents):
            # Set currents
            for i, coil in enumerate(self.coils):
                coil.current = currents[i]
            
            # Calculate field at focus center
            field, _ = self.calculate_field_and_gradient(focus_center.reshape(1, -1))
            field_magnitude = np.linalg.norm(field[0])
            
            # Objective: minimize difference from target
            return (field_magnitude - target_field_strength)**2
        
        # Initial guess
        initial_currents = [coil.current for coil in self.coils]
        
        # Bounds for currents
        bounds = [(-coil.max_current, coil.max_current) for coil in self.coils]
        
        # Optimize
        result = minimize(objective, initial_currents, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimized_currents = {}
            for i, coil in enumerate(self.coils):
                coil.current = result.x[i]
                optimized_currents[coil.id] = result.x[i]
            
            self.logger.info(f"Current optimization successful. Target: {target_field_strength:.3f}T")
            return optimized_currents
        else:
            self.logger.warning("Current optimization failed")
            return {coil.id: coil.current for coil in self.coils}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        total_power = sum(coil.current**2 * coil.resistance for coil in self.coils)
        total_moment = sum(abs(coil.magnetic_dipole_moment) for coil in self.coils)
        
        return {
            'num_coils': len(self.coils),
            'total_power': total_power,
            'total_magnetic_moment': total_moment,
            'field_energy': self.get_field_energy(),
            'coil_currents': {coil.id: coil.current for coil in self.coils},
            'coil_positions': {coil.id: coil.position.tolist() for coil in self.coils},
            'max_field_strength_estimate': self._estimate_max_field_strength()
        }
    
    def _estimate_max_field_strength(self) -> float:
        """Estimate maximum field strength in the system"""
        max_field = 0.0
        
        for coil in self.coils:
            # Field at coil surface (approximate)
            field_at_surface = (self.MU_0 * abs(coil.current) * coil.turns) / (2 * coil.radius)
            max_field = max(max_field, field_at_surface)
        
        return max_field
    
    def export_field_configuration(self) -> Dict[str, Any]:
        """Export complete field configuration"""
        return {
            'coils': [coil.get_dict() for coil in self.coils],
            'system_stats': self.get_system_stats(),
            'field_type': 'realistic_biotivart',
            'created_timestamp': time.time()
        }
    
    def load_field_configuration(self, config: Dict[str, Any]):
        """Load field configuration from dictionary"""
        self.coils = []
        
        for coil_data in config['coils']:
            coil = MagneticCoil(
                id=coil_data['id'],
                position=np.array(coil_data['position']),
                radius=coil_data['radius'],
                current=coil_data['current'],
                turns=coil_data['turns'],
                normal=np.array(coil_data['normal']),
                resistance=coil_data.get('resistance', 1.0),
                max_current=coil_data.get('max_current', 10.0)
            )
            self.coils.append(coil)
        
        self.logger.info(f"Loaded field configuration with {len(self.coils)} coils")
    
    def visualize_field_lines(self, bounds: Dict[str, Tuple[float, float]], 
                             resolution: int = 20) -> Dict[str, Any]:
        """
        Generate field line data for visualization
        
        Args:
            bounds: Dictionary with 'x', 'y', 'z' bounds as (min, max) tuples
            resolution: Number of sample points per dimension
            
        Returns:
            Dictionary with field line data
        """
        # Create sampling grid
        x = np.linspace(bounds['x'][0], bounds['x'][1], resolution)
        y = np.linspace(bounds['y'][0], bounds['y'][1], resolution)
        z = np.linspace(bounds['z'][0], bounds['z'][1], resolution)
        
        # Sample on a few planes for visualization
        field_data = {}
        
        # XY plane at z=0
        xy_points = []
        for xi in x:
            for yi in y:
                xy_points.append([xi, yi, 0])
        
        xy_points = np.array(xy_points)
        xy_fields, _ = self.calculate_field_and_gradient(xy_points)
        
        field_data['xy_plane'] = {
            'positions': xy_points.tolist(),
            'fields': xy_fields.tolist(),
            'field_magnitudes': np.linalg.norm(xy_fields, axis=1).tolist()
        }
        
        # XZ plane at y=0
        xz_points = []
        for xi in x:
            for zi in z:
                xz_points.append([xi, 0, zi])
        
        xz_points = np.array(xz_points)
        xz_fields, _ = self.calculate_field_and_gradient(xz_points)
        
        field_data['xz_plane'] = {
            'positions': xz_points.tolist(),
            'fields': xz_fields.tolist(),
            'field_magnitudes': np.linalg.norm(xz_fields, axis=1).tolist()
        }
        
        return field_data
    
    def __len__(self) -> int:
        """Return number of coils"""
        return len(self.coils)
    
    def __iter__(self):
        """Make system iterable"""
        return iter(self.coils)
    
    def __getitem__(self, index: int) -> MagneticCoil:
        """Allow indexing"""
        return self.coils[index]