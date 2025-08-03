# MagNanoBridge: Magnetic Nanoparticle Simulation

A physically realistic 3D simulation of magnetically steerable nanoparticles forming temporary bridges across bone fractures. This project combines advanced physics modeling with real-time visualization to study magnetic guidance of therapeutic nanoparticles.

## 🎯 Overview

MagNanoBridge simulates the behavior of magnetic nanoparticles in biological environments, focusing on their ability to form structured bridges across bone fractures under the influence of external magnetic fields. The simulation includes:

- **Realistic Physics**: Magnetic gradient forces, dipole-dipole interactions, fluid dynamics, and Brownian motion
- **Advanced Control**: PID and AI-based magnetic field control systems
- **3D Visualization**: Real-time particle tracking, force vectors, and magnetic field visualization
- **Scientific Output**: Comprehensive data export and analysis capabilities

##  Architecture

```
MagNanoBridge/
├── src/                    # Core Python simulation engine
│   ├── main.py            # Main entry point
│   ├── physics/           # Physics modules
│   │   ├── simulation.py  # Main simulation loop
│   │   ├── particles.py   # Particle system
│   │   ├── fields.py      # Magnetic field calculations
│   │   ├── forces.py      # Force calculations
│   │   └── integrator.py  # Numerical integration
│   ├── control/           # Control systems
│   │   └── focus_controller.py  # PID/AI control
│   ├── utils/             # Utilities
│   │   ├── config_loader.py     # Configuration management
│   │   ├── data_export.py       # Data export utilities
│   │   └── logger.py            # Logging system
│   └── config/            # Configuration files
├── visualization/         # Web-based 3D visualization
│   ├── index.html         # Main interface
│   ├── threejs/           # Three.js components
│   │   ├── simulation-renderer.js     # Main renderer
│   │   ├── particle-system-3d.js     # Particle visualization
│   │   ├── magnetic-field-viz.js     # Field visualization
│   │   ├── controls-manager.js       # UI controls
│   │   └── data-interface.js         # Backend communication
│   └── static/            # Assets and styles
├── notebooks/             # Jupyter analysis notebooks
├── data/                  # Simulation results
├── tests/                 # Unit and integration tests
├── doc/                   # Documentation
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with scientific computing libraries
- Modern web browser with WebGL support
- Google Colab account (for cloud execution)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/magnanobridge.git
   cd magnanobridge
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Google Colab:**
   ```python
   !git clone https://github.com/your-repo/magnanobridge.git
   %cd magnanobridge
   !pip install -r requirements.txt
   ```

### Basic Usage

1. **Run with default settings:**
   ```bash
   python src/main.py
   ```

2. **Interactive mode with visualization:**
   ```bash
   python src/main.py --mode interactive --visualize
   ```

3. **Batch mode for optimization:**
   ```bash
   python src/main.py --mode batch --config config/high_precision.yaml
   ```

4. **Open visualization:**
   Navigate to `visualization/index.html` in your browser or serve with:
   ```bash
   python -m http.server 8000
   # Then open http://localhost:8000/visualization/
   ```

## ⚙️ Configuration

### YAML Configuration Example

```yaml
# config/example.yaml
simulation:
  max_steps: 10000
  dt: 1e-6  # 1 microsecond time step
  boundary_box:
    min: [-10e-3, -10e-3, -15e-3]  # 10mm box
    max: [10e-3, 10e-3, 5e-3]

particles:
  count: 100
  radius: 50e-9  # 50 nm core
  shell_thickness: 10e-9  # 10 nm polymer shell
  magnetic_moment: 1e-18  # A⋅m²

physics:
  enable_dipole_forces: true
  enable_brownian_motion: true
  temperature: 310.0  # Body temperature (K)
  fluid_viscosity: 1e-3  # Water viscosity

focus:
  target_zone:
    center: [0, 0, 0]
    shape: "cylinder"
    parameters:
      radius: 1e-3  # 1mm
      height: 2e-3  # 2mm
  control_type: "pid"
  pid_gains:
    kp: 1.0
    ki: 0.1
    kd: 0.05
```

### Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  -c, --config PATH     Configuration file path
  -m, --mode MODE       Simulation mode (interactive/batch)
  -o, --output PATH     Output directory
  --log-level LEVEL     Logging level (DEBUG/INFO/WARNING/ERROR)
  --visualize           Enable visualization
  --no-visualize        Disable visualization
```

## 🎮 Visualization Controls

### Keyboard Shortcuts
- **Space**: Play/Pause simulation
- **R**: Reset simulation
- **S**: Single step
- **1/2/3**: Camera views (top/side/3D)
- **T**: Toggle particle trails
- **F**: Toggle force vectors
- **M**: Toggle magnetic field visualization
- **Z**: Toggle focus zone
- **+/-**: Adjust simulation speed
- **Ctrl+E**: Export data
- **Ctrl+S**: Save configuration

### Mouse Controls
- **Left Click + Drag**: Rotate camera
- **Mouse Wheel**: Zoom in/out
- **Click on Particle**: Select and highlight
- **Right Click**: Context menu

## 🔬 Physics Model

### Particle Properties
- **Core**: Magnetic nanoparticle (typically iron oxide)
- **Shell**: Biocompatible polymer coating
- **Size Range**: 20-200 nm total diameter
- **Magnetic Moment**: ~10⁻¹⁸ A⋅m²

### Forces Implemented
1. **Magnetic Gradient Force**: F = ∇(m⋅B)
2. **Dipole-Dipole Interactions**: Particle-particle magnetic forces
3. **Stokes Drag**: Fluid resistance
4. **Lennard-Jones Collisions**: Soft-sphere particle collisions
5. **Brownian Motion**: Thermal fluctuations

### Magnetic Field System
- **Biot-Savart Law**: Realistic field calculations
- **Multiple Coils**: 3D field gradients
- **Real-time Control**: PID and AI-based optimization

## 📊 Data Analysis

### Output Formats
- **CSV**: Time series data for analysis
- **JSON**: Structured simulation results
- **HDF5**: Large dataset storage
- **Images**: Visualization screenshots

### Key Metrics
- **Focus Efficiency**: Fraction of particles in target zone
- **Bridge Formation Time**: Time to achieve therapeutic density
- **Energy Consumption**: Magnetic field power requirements
- **Particle Distribution**: Spatial and temporal analysis

### Example Analysis

```python
# Load simulation results
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data
with open('data/simulation_results.json', 'r') as f:
    results = json.load(f)

# Analyze focus efficiency over time
df = pd.DataFrame(results['trajectory_data'])
df['efficiency'] = df['metrics'].apply(lambda x: x['focus_efficiency'])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['efficiency'])
plt.xlabel('Time (s)')
plt.ylabel('Focus Efficiency')
plt.title('Particle Focusing Performance')
plt.show()
```

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Run Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### Performance Benchmarks
```bash
python tests/benchmark.py --particles 1000 --steps 5000
```

## 🎯 Use Cases

### Research Applications
- **Magnetic Drug Delivery**: Optimize targeting strategies
- **Hyperthermia Treatment**: Heat generation studies
- **Tissue Engineering**: Scaffold formation analysis

### Educational Use
- **Physics Demonstration**: Magnetic force visualization
- **Computational Methods**: Numerical simulation techniques
- **Biomedical Engineering**: Medical device design

### Industrial Applications
- **Device Design**: Magnetic actuator optimization
- **Quality Control**: Particle behavior validation
- **Process Optimization**: Manufacturing parameters

## 🛠️ Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style
- **Python**: PEP 8 compliance
- **JavaScript**: ESLint configuration
- **Documentation**: NumPy docstring format

### Performance Optimization
- **Large Systems**: Enable Fast Multipole Method
- **GPU Acceleration**: CUDA support (future)
- **Parallel Processing**: Multi-core force calculations

## 📋 Requirements

### Python Dependencies
```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
pyyaml>=5.4.0
h5py>=3.1.0
pytest>=6.0.0
jupyter>=1.0.0
```

### System Requirements
- **RAM**: 4GB minimum, 16GB recommended for large simulations
- **CPU**: Multi-core processor recommended
- **GPU**: Optional, for future acceleration features
- **Storage**: 1GB for software, additional space for data output

## 🚨 Troubleshooting

### Common Issues

**"WebGL not supported" error:**
- Update your browser to the latest version
- Enable hardware acceleration in browser settings
- Try a different browser (Chrome, Firefox, Safari)

**Python import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

**Simulation runs slowly:**
- Reduce particle count in configuration
- Disable Brownian motion for faster computation
- Use batch mode instead of interactive mode

**Out of memory errors:**
- Reduce `max_history` in configuration
- Decrease export frequency
- Use compression for data output

### Getting Help

1. **Documentation**: Check the `doc/` directory
2. **Issues**: Submit bug reports on GitHub
3. **Discussions**: Join the community forum
4. **Email**: Contact maintainers directly

## 📜 License

This project is proprietary software. All rights reserved.

**Important**: This software is for research and educational use only. Commercial use requires explicit permission from the copyright holder.

## 🙏 Acknowledgments

- **Three.js Community**: 3D visualization framework
- **NumPy/SciPy Teams**: Scientific computing libraries
- **Research Collaborators**: Physics model validation
- **Beta Testers**: User interface feedback

## 🔗 References

1. Magnetic Drug Delivery Systems - *Nature Reviews Drug Discovery*
2. Nanoparticle Physics and Biomedical Applications - *Advanced Materials*
3. Computational Methods in Biotechnology - *Annual Review of Bioengineering*
4. Three.js Documentation - https://threejs.org/docs/

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintainer**: MagNanoBridge Development Team  
**Status**: Active Development