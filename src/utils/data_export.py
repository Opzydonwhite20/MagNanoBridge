"""
Data export utilities for MagNanoBridge simulation
Handles CSV, JSON, and HDF5 export formats with compression
"""

import os
import json
import csv
import gzip
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime


class DataExporter:
    """
    Handles export of simulation data in multiple formats
    """
    
    def __init__(self, 
                 output_dir: str = './data/output',
                 compress: bool = True,
                 export_formats: List[str] = None):
        """
        Initialize data exporter
        
        Args:
            output_dir: Output directory for data files
            compress: Whether to compress output files
            export_formats: List of formats to export ('csv', 'json', 'hdf5')
        """
        self.output_dir = Path(output_dir)
        self.compress = compress
        
        if export_formats is None:
            export_formats = ['csv', 'json']
        self.export_formats = export_formats
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data buffers
        self.trajectory_data = []
        self.force_data = []
        self.field_data = []
        self.simulation_metadata = {}
        
        # File handles
        self.csv_writers = {}
        self.csv_files = {}
        
        # Create timestamped subdirectory
        self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DataExporter initialized: {self.session_dir}")
        
        # Initialize CSV files if needed
        if 'csv' in self.export_formats:
            self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV files and writers"""
        csv_files = {
            'trajectories': 'particle_trajectories.csv',
            'forces': 'particle_forces.csv',
            'fields': 'magnetic_fields.csv',
            'summary': 'simulation_summary.csv'
        }
        
        for data_type, filename in csv_files.items():
            filepath = self.session_dir / filename
            
            if self.compress:
                file_handle = gzip.open(f"{filepath}.gz", 'wt', newline='')
            else:
                file_handle = open(filepath, 'w', newline='')
            
            self.csv_files[data_type] = file_handle
            
            # Initialize CSV writers with headers
            if data_type == 'trajectories':
                writer = csv.writer(file_handle)
                writer.writerow([
                    'time', 'step', 'particle_id', 'x', 'y', 'z', 
                    'vx', 'vy', 'vz', 'radius', 'mass'
                ])
                self.csv_writers[data_type] = writer
            
            elif data_type == 'forces':
                writer = csv.writer(file_handle)
                writer.writerow([
                    'time', 'step', 'particle_id', 
                    'fx_total', 'fy_total', 'fz_total',
                    'fx_magnetic', 'fy_magnetic', 'fz_magnetic',
                    'fx_dipole', 'fy_dipole', 'fz_dipole',
                    'fx_drag', 'fy_drag', 'fz_drag',
                    'fx_collision', 'fy_collision', 'fz_collision',
                    'fx_brownian', 'fy_brownian', 'fz_brownian'
                ])
                self.csv_writers[data_type] = writer
            
            elif data_type == 'fields':
                writer = csv.writer(file_handle)
                writer.writerow([
                    'time', 'step', 'x', 'y', 'z',
                    'bx', 'by', 'bz', 'b_magnitude',
                    'dbx_dx', 'dbx_dy', 'dbx_dz',
                    'dby_dx', 'dby_dy', 'dby_dz',
                    'dbz_dx', 'dbz_dy', 'dbz_dz'
                ])
                self.csv_writers[data_type] = writer
            
            elif data_type == 'summary':
                writer = csv.writer(file_handle)
                writer.writerow([
                    'time', 'step', 'total_kinetic_energy', 'total_potential_energy',
                    'focus_efficiency', 'bridge_formed', 'particles_in_zone',
                    'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z',
                    'mean_speed', 'max_speed', 'temperature_estimate'
                ])
                self.csv_writers[data_type] = writer
    
    def export_state(self, simulation_state):
        """
        Export complete simulation state
        
        Args:
            simulation_state: SimulationState object from simulation
        """
        # Export to different formats
        if 'csv' in self.export_formats:
            self._export_csv(simulation_state)
        
        # Store in buffers for batch export
        self._store_in_buffers(simulation_state)
    
    def _export_csv(self, state):
        """Export current state to CSV files"""
        time_val = state.time
        step_val = state.step
        
        # Export trajectories
        if 'trajectories' in self.csv_writers:
            writer = self.csv_writers['trajectories']
            for i in range(len(state.particles)):
                particle_id = f"p_{i}"
                pos = state.particles[i]
                vel = state.velocities[i]
                
                # Assuming uniform particles for now - would need particle system reference
                radius = 50e-9  # Default
                mass = 1e-15    # Default
                
                writer.writerow([
                    time_val, step_val, particle_id,
                    pos[0], pos[1], pos[2],
                    vel[0], vel[1], vel[2],
                    radius, mass
                ])
        
        # Export summary data
        if 'summary' in self.csv_writers:
            writer = self.csv_writers['summary']
            
            # Calculate summary statistics
            kinetic_energy = 0.5 * np.sum(state.velocities**2) * 1e-15  # Approximate
            potential_energy = 0.0  # Would need force calculator reference
            
            com = np.mean(state.particles, axis=0)
            speeds = np.linalg.norm(state.velocities, axis=1)
            mean_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            
            # Temperature estimate from kinetic energy
            kb = 1.380649e-23
            n_particles = len(state.particles)
            temp_estimate = (2 * kinetic_energy) / (3 * n_particles * kb) if n_particles > 0 else 0
            
            particles_in_zone = 0  # Would need focus controller reference
            
            writer.writerow([
                time_val, step_val, kinetic_energy, potential_energy,
                state.focus_efficiency, state.bridge_formed, particles_in_zone,
                com[0], com[1], com[2],
                mean_speed, max_speed, temp_estimate
            ])
    
    def _store_in_buffers(self, state):
        """Store state data in memory buffers for batch export"""
        # Trajectory data
        trajectory_entry = {
            'time': state.time,
            'step': state.step,
            'positions': state.particles.tolist(),
            'velocities': state.velocities.tolist(),
            'forces': state.forces.tolist()
        }
        self.trajectory_data.append(trajectory_entry)
        
        # Field data (sample a few points)
        if hasattr(state, 'magnetic_field') and len(state.magnetic_field) > 0:
            field_entry = {
                'time': state.time,
                'step': state.step,
                'field_vectors': state.magnetic_field.tolist(),
                'sample_positions': state.particles[:10].tolist()  # First 10 particles
            }
            self.field_data.append(field_entry)
    
    def export_final_data(self, simulation_results: Dict[str, Any]):
        """
        Export final simulation results and metadata
        
        Args:
            simulation_results: Final results dictionary from simulation
        """
        # Export JSON summary
        if 'json' in self.export_formats:
            self._export_json_summary(simulation_results)
        
        # Export HDF5 if available
        if 'hdf5' in self.export_formats:
            try:
                self._export_hdf5()
            except ImportError:
                self.logger.warning("HDF5 export requested but h5py not available")
        
        # Export metadata
        self._export_metadata(simulation_results)
    
    def _export_json_summary(self, results: Dict[str, Any]):
        """Export JSON summary with all trajectory data"""
        json_data = {
            'simulation_results': results,
            'trajectory_data': self.trajectory_data,
            'metadata': self.simulation_metadata,
            'export_info': {
                'export_time': datetime.now().isoformat(),
                'total_data_points': len(self.trajectory_data),
                'compression_used': self.compress
            }
        }
        
        filepath = self.session_dir / 'simulation_data.json'
        
        if self.compress:
            with gzip.open(f"{filepath}.gz", 'wt') as f:
                json.dump(json_data, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        self.logger.info(f"JSON data exported: {filepath}")
    
    def _export_hdf5(self):
        """Export data in HDF5 format for large datasets"""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 export")
        
        filepath = self.session_dir / 'simulation_data.h5'
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            traj_group = f.create_group('trajectories')
            field_group = f.create_group('fields')
            meta_group = f.create_group('metadata')
            
            # Export trajectory data
            if self.trajectory_data:
                times = [entry['time'] for entry in self.trajectory_data]
                steps = [entry['step'] for entry in self.trajectory_data]
                positions = np.array([entry['positions'] for entry in self.trajectory_data])
                velocities = np.array([entry['velocities'] for entry in self.trajectory_data])
                forces = np.array([entry['forces'] for entry in self.trajectory_data])
                
                traj_group.create_dataset('times', data=times)
                traj_group.create_dataset('steps', data=steps)
                traj_group.create_dataset('positions', data=positions, compression='gzip')
                traj_group.create_dataset('velocities', data=velocities, compression='gzip')
                traj_group.create_dataset('forces', data=forces, compression='gzip')
            
            # Export field data
            if self.field_data:
                field_times = [entry['time'] for entry in self.field_data]
                field_vectors = np.array([entry['field_vectors'] for entry in self.field_data])
                
                field_group.create_dataset('times', data=field_times)
                field_group.create_dataset('field_vectors', data=field_vectors, compression='gzip')
            
            # Export metadata
            for key, value in self.simulation_metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.create_dataset(key, data=str(value))
        
        self.logger.info(f"HDF5 data exported: {filepath}")
    
    def _export_metadata(self, results: Dict[str, Any]):
        """Export simulation metadata and configuration"""
        metadata = {
            'simulation_info': {
                'total_time': results.get('total_time', 0),
                'total_steps': results.get('total_steps', 0),
                'final_efficiency': results.get('focus_efficiency', 0),
                'bridge_formed': results.get('bridge_formed', False),
                'particle_count': results.get('particle_count', 0)
            },
            'performance_stats': results.get('performance_stats', {}),
            'configuration': results.get('config_used', {}),
            'export_summary': {
                'data_points_exported': len(self.trajectory_data),
                'field_samples': len(self.field_data),
                'export_formats': self.export_formats,
                'compression_used': self.compress,
                'session_directory': str(self.session_dir)
            }
        }
        
        # Save as both JSON and text
        metadata_json = self.session_dir / 'metadata.json'
        with open(metadata_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Human-readable summary
        summary_text = self.session_dir / 'summary.txt'
        with open(summary_text, 'w') as f:
            f.write("MagNanoBridge Simulation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session Directory: {self.session_dir}\n\n")
            
            sim_info = metadata['simulation_info']
            f.write("Simulation Results:\n")
            f.write(f"  Total Time: {sim_info['total_time']:.6f} seconds\n")
            f.write(f"  Total Steps: {sim_info['total_steps']}\n")
            f.write(f"  Final Focus Efficiency: {sim_info['final_efficiency']:.3f}\n")
            f.write(f"  Bridge Formed: {sim_info['bridge_formed']}\n")
            f.write(f"  Particle Count: {sim_info['particle_count']}\n\n")
            
            perf_stats = metadata['performance_stats']
            if perf_stats:
                f.write("Performance Statistics:\n")
                f.write(f"  Average Step Time: {perf_stats.get('avg_step_time', 0):.6f} seconds\n")
                f.write(f"  Force Calculation Time: {perf_stats.get('force_calc_time', 0):.6f} seconds\n")
                f.write(f"  Integration Time: {perf_stats.get('integration_time', 0):.6f} seconds\n\n")
            
            export_info = metadata['export_summary']
            f.write("Export Information:\n")
            f.write(f"  Data Points: {export_info['data_points_exported']}\n")
            f.write(f"  Field Samples: {export_info['field_samples']}\n")
            f.write(f"  Formats: {', '.join(export_info['export_formats'])}\n")
            f.write(f"  Compression: {export_info['compression_used']}\n")
        
        self.logger.info("Metadata and summary exported")
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set simulation metadata for export"""
        self.simulation_metadata.update(metadata)
    
    def export_particle_analysis(self, 
                                positions: np.ndarray,
                                velocities: np.ndarray,
                                forces: np.ndarray,
                                time_points: np.ndarray):
        """
        Export detailed particle analysis
        
        Args:
            positions: Particle positions over time (T, N, 3)
            velocities: Particle velocities over time (T, N, 3)  
            forces: Particle forces over time (T, N, 3)
            time_points: Time values (T,)
        """
        analysis_dir = self.session_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        n_times, n_particles, _ = positions.shape
        
        # Calculate particle statistics
        speeds = np.linalg.norm(velocities, axis=2)
        force_magnitudes = np.linalg.norm(forces, axis=2)
        
        # Export individual particle trajectories
        for particle_id in range(min(n_particles, 10)):  # Limit to first 10 particles
            particle_data = {
                'time': time_points.tolist(),
                'position_x': positions[:, particle_id, 0].tolist(),
                'position_y': positions[:, particle_id, 1].tolist(),
                'position_z': positions[:, particle_id, 2].tolist(),
                'velocity_x': velocities[:, particle_id, 0].tolist(),
                'velocity_y': velocities[:, particle_id, 1].tolist(),
                'velocity_z': velocities[:, particle_id, 2].tolist(),
                'speed': speeds[:, particle_id].tolist(),
                'force_magnitude': force_magnitudes[:, particle_id].tolist()
            }
            
            filepath = analysis_dir / f'particle_{particle_id}_trajectory.json'
            with open(filepath, 'w') as f:
                json.dump(particle_data, f, indent=2)
        
        # Export system-wide statistics
        system_stats = {
            'time': time_points.tolist(),
            'center_of_mass': np.mean(positions, axis=1).tolist(),
            'mean_speed': np.mean(speeds, axis=1).tolist(),
            'max_speed': np.max(speeds, axis=1).tolist(),
            'std_speed': np.std(speeds, axis=1).tolist(),
            'mean_force': np.mean(force_magnitudes, axis=1).tolist(),
            'max_force': np.max(force_magnitudes, axis=1).tolist(),
            'kinetic_energy': (0.5 * 1e-15 * np.sum(speeds**2, axis=1)).tolist()  # Approximate
        }
        
        filepath = analysis_dir / 'system_statistics.json'
        with open(filepath, 'w') as f:
            json.dump(system_stats, f, indent=2)
        
        self.logger.info(f"Particle analysis exported to {analysis_dir}")
    
    def finalize(self):
        """Close all file handles and finalize export"""
        # Close CSV files
        for file_handle in self.csv_files.values():
            file_handle.close()
        
        # Create export summary
        self._create_export_summary()
        
        self.logger.info(f"Data export finalized: {self.session_dir}")
    
    def _create_export_summary(self):
        """Create a summary of all exported files"""
        summary = {
            'session_directory': str(self.session_dir),
            'export_time': datetime.now().isoformat(),
            'files_created': [],
            'total_size_mb': 0
        }
        
        # List all created files
        for filepath in self.session_dir.rglob('*'):
            if filepath.is_file():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                summary['files_created'].append({
                    'filename': filepath.name,
                    'relative_path': str(filepath.relative_to(self.session_dir)),
                    'size_mb': round(size_mb, 3)
                })
                summary['total_size_mb'] += size_mb
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 3)
        summary['file_count'] = len(summary['files_created'])
        
        # Save summary
        summary_path = self.session_dir / 'export_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_export_path(self) -> Path:
        """Get the current export session directory"""
        return self.session_dir
    
    def cleanup_old_exports(self, keep_days: int = 7):
        """
        Clean up old export directories
        
        Args:
            keep_days: Number of days to keep exports
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (keep_days * 24 * 3600)
        
        deleted_count = 0
        for export_dir in self.output_dir.iterdir():
            if export_dir.is_dir() and export_dir != self.session_dir:
                try:
                    # Check if directory name follows timestamp format
                    datetime.strptime(export_dir.name, "%Y%m%d_%H%M%S")
                    
                    # Check modification time
                    if export_dir.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(export_dir)
                        deleted_count += 1
                        
                except (ValueError, OSError):
                    # Skip directories that don't match format or can't be accessed
                    continue
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old export directories")
    
    @staticmethod
    def load_simulation_data(data_path: str) -> Dict[str, Any]:
        """
        Load previously exported simulation data
        
        Args:
            data_path: Path to exported data directory or file
            
        Returns:
            Loaded simulation data
        """
        data_path = Path(data_path)
        
        if data_path.is_dir():
            # Load from directory
            json_file = data_path / 'simulation_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    return json.load(f)
            
            # Try compressed version
            json_gz = data_path / 'simulation_data.json.gz'
            if json_gz.exists():
                with gzip.open(json_gz, 'rt') as f:
                    return json.load(f)
            
            raise FileNotFoundError("No simulation data file found in directory")
        
        else:
            # Load single file
            if data_path.suffix == '.gz':
                with gzip.open(data_path, 'rt') as f:
                    return json.load(f)
            else:
                with open(data_path, 'r') as f:
                    return json.load(f)
    
    @staticmethod
    def convert_format(input_path: str, output_path: str, target_format: str):
        """
        Convert exported data between formats
        
        Args:
            input_path: Input file path
            output_path: Output file path
            target_format: Target format ('csv', 'json', 'hdf5')
        """
        # Load data
        data = DataExporter.load_simulation_data(input_path)
        
        # Create new exporter
        output_dir = os.path.dirname(output_path)
        exporter = DataExporter(output_dir, export_formats=[target_format])
        
        # Set metadata
        exporter.set_metadata(data.get('metadata', {}))
        
        # Export in new format
        if 'trajectory_data' in data:
            exporter.trajectory_data = data['trajectory_data']
        
        exporter.export_final_data(data.get('simulation_results', {}))
        exporter.finalize()
        
        logging.getLogger(__name__).info(f"Converted {input_path} to {target_format} format")