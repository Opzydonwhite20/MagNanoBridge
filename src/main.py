"""
MagNanoBridge: Main simulation entry point
Magnetically guided nanoparticle simulation for bone fracture bridging

"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.physics.simulation import MagneticSimulation
from src.physics.particles import ParticleSystem
from src.physics.fields import MagneticFieldSystem
from src.control.focus_controller import FocusController
from src.utils.config_loader import ConfigLoader
from src.utils.data_export import DataExporter
from src.utils.logger import setup_logger


def setup_simulation(config_path: str = None) -> MagneticSimulation:
    """
    Initialize simulation with configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured MagneticSimulation instance
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = ConfigLoader.load_config(config_path)
    else:
        config = ConfigLoader.get_default_config()
    
    # Initialize particle system
    particle_system = ParticleSystem(
        num_particles=config['particles']['count'],
        particle_radius=config['particles']['radius'],
        particle_mass=config['particles']['mass'],
        shell_thickness=config['particles']['shell_thickness'],
        magnetic_moment=config['particles']['magnetic_moment']
    )
    
    # Initialize magnetic field system
    field_system = MagneticFieldSystem(
        coil_positions=config['fields']['coil_positions'],
        coil_currents=config['fields']['coil_currents'],
        coil_radii=config['fields']['coil_radii']
    )
    
    # Initialize focus controller
    focus_controller = FocusController(
        target_zone=config['focus']['target_zone'],
        control_type=config['focus']['control_type'],
        pid_gains=config['focus']['pid_gains']
    )
    
    # Create simulation
    simulation = MagneticSimulation(
        particle_system=particle_system,
        field_system=field_system,
        focus_controller=focus_controller,
        config=config
    )
    
    return simulation


def run_interactive_mode(simulation: MagneticSimulation, config: Dict[str, Any]):
    """Run simulation in interactive mode with visualization"""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive simulation mode")
    
    # Initialize data exporter
    exporter = DataExporter(output_dir=config['output']['data_dir'])
    
    try:
        # Start simulation loop
        simulation.run_interactive(
            steps=config['simulation']['max_steps'],
            dt=config['simulation']['dt'],
            real_time=config['visualization']['real_time'],
            export_interval=config['output']['export_interval'],
            exporter=exporter
        )
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise
    finally:
        # Export final data
        exporter.finalize()


def run_batch_mode(simulation: MagneticSimulation, config: Dict[str, Any]):
    """Run simulation in batch mode for optimization"""
    logger = logging.getLogger(__name__)
    logger.info("Starting batch simulation mode")
    
    # Initialize data exporter
    exporter = DataExporter(output_dir=config['output']['data_dir'])
    
    try:
        # Run batch simulation
        results = simulation.run_batch(
            steps=config['simulation']['max_steps'],
            dt=config['simulation']['dt'],
            export_interval=config['output']['export_interval'],
            exporter=exporter
        )
        
        # Log results
        logger.info(f"Batch simulation completed. Focus efficiency: {results['focus_efficiency']:.3f}")
        logger.info(f"Bridge formation time: {results['bridge_time']:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch simulation error: {e}")
        raise
    finally:
        # Export final data
        exporter.finalize()


def main():
    """Main entry point for MagNanoBridge simulation"""
    
    parser = argparse.ArgumentParser(description='MagNanoBridge: Magnetic Nanoparticle Simulation')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--mode', '-m', choices=['interactive', 'batch'], default='interactive',
                       help='Simulation mode')
    parser.add_argument('--output', '-o', type=str, default='./data/output',
                       help='Output directory for results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Enable visualization (interactive mode only)')
    parser.add_argument('--no-visualize', action='store_true', default=False,
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing MagNanoBridge simulation")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config or 'default'}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Setup simulation
        simulation = setup_simulation(args.config)
        
        # Update output directory in config
        simulation.config['output']['data_dir'] = args.output
        
        # Handle visualization settings
        if args.no_visualize:
            simulation.config['visualization']['enabled'] = False
        elif args.visualize and args.mode == 'interactive':
            simulation.config['visualization']['enabled'] = True
        
        # Run simulation based on mode
        if args.mode == 'interactive':
            run_interactive_mode(simulation, simulation.config)
        elif args.mode == 'batch':
            results = run_batch_mode(simulation, simulation.config)
            
            # Save batch results
            results_path = os.path.join(args.output, 'batch_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch results saved to: {results_path}")
        
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()