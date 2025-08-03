/**
 * Data Interface for MagNanoBridge visualization
 * Handles communication with Python simulation backend
 */

class DataInterface {
    constructor(app) {
        this.app = app;
        
        // Connection settings
        this.wsUrl = 'ws://localhost:8765'; // WebSocket server URL
        this.httpUrl = 'http://localhost:8000'; // HTTP API URL
        
        // Connection state
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000; // ms
        
        // Data buffers
        this.lastSimulationData = null;
        this.dataHistory = [];
        this.maxHistorySize = 1000;
        
        // Callbacks
        this.onDataReceived = null;
        this.onConnectionStateChanged = null;
        
        // Simulation state
        this.simulationRunning = false;
        this.simulationSpeed = 1.0;
        
        // Mock data for development/testing
        this.useMockData = false;
        this.mockDataInterval = null;
    }

    async initialize() {
        console.log('Initializing DataInterface...');
        
        // Try to connect to real backend
        try {
            await this.connect();
        } catch (error) {
            console.warn('Failed to connect to backend, enabling mock data mode:', error);
            this.enableMockData();
        }
        
        console.log('DataInterface initialized');
    }

    async connect() {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(this.wsUrl);
                
                this.websocket.onopen = (event) => {
                    console.log('WebSocket connected');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    
                    if (this.onConnectionStateChanged) {
                        this.onConnectionStateChanged(true);
                    }
                    
                    resolve();
                };
                
                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleIncomingData(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
                
                this.websocket.onclose = (event) => {
                    console.log('WebSocket disconnected');
                    this.isConnected = false;
                    
                    if (this.onConnectionStateChanged) {
                        this.onConnectionStateChanged(false);
                    }
                    
                    // Attempt reconnection
                    if (this.reconnectAttempts < this.maxReconnectAttempts) {
                        setTimeout(() => {
                            this.reconnectAttempts++;
                            console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                            this.connect().catch(() => {
                                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                                    console.log('Max reconnection attempts reached, enabling mock data');
                                    this.enableMockData();
                                }
                            });
                        }, this.reconnectDelay);
                    }
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };
                
                // Timeout for connection
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.websocket.close();
                        reject(new Error('Connection timeout'));
                    }
                }, 5000);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    handleIncomingData(data) {
        // Store in history
        this.dataHistory.push({
            timestamp: Date.now(),
            data: data
        });
        
        // Limit history size
        if (this.dataHistory.length > this.maxHistorySize) {
            this.dataHistory.shift();
        }
        
        // Update last data
        this.lastSimulationData = data;
        
        // Notify app
        if (this.onDataReceived) {
            this.onDataReceived(data);
        }
    }

    enableMockData() {
        this.useMockData = true;
        console.log('Mock data mode enabled');
        
        // Start mock data generation
        this.startMockDataGeneration();
    }

    startMockDataGeneration() {
        if (this.mockDataInterval) {
            clearInterval(this.mockDataInterval);
        }
        
        let step = 0;
        const particleCount = 100;
        
        this.mockDataInterval = setInterval(() => {
            const mockData = this.generateMockData(step, particleCount);
            this.handleIncomingData(mockData);
            step++;
        }, 50); // 20 FPS mock data
    }

    generateMockData(step, particleCount) {
        const time = step * 0.05; // 50ms per step
        
        // Generate particle positions with some motion
        const positions = [];
        const velocities = [];
        const forces = [];
        
        for (let i = 0; i < particleCount; i++) {
            // Spiral motion toward center
            const angle = time * 0.1 + i * 0.1;
            const radius = 0.005 * (1 + 0.3 * Math.sin(time * 0.5 + i * 0.2));
            const height = 0.002 * Math.sin(time * 0.3 + i * 0.15);
            
            positions.push([
                radius * Math.cos(angle),
                height,
                radius * Math.sin(angle)
            ]);
            
            // Velocity toward center with some randomness
            velocities.push([
                -0.001 * Math.cos(angle) + 0.0002 * (Math.random() - 0.5),
                0.0001 * Math.sin(time + i),
                -0.001 * Math.sin(angle) + 0.0002 * (Math.random() - 0.5)
            ]);
            
            // Mock forces
            forces.push([
                1e-12 * (Math.random() - 0.5),
                1e-12 * (Math.random() - 0.5),
                1e-12 * (Math.random() - 0.5)
            ]);
        }
        
        // Calculate focus efficiency (particles within cylinder)
        const focusRadius = 0.001;
        const focusHeight = 0.002;
        let particlesInZone = 0;
        
        positions.forEach(pos => {
            const radialDist = Math.sqrt(pos[0]**2 + pos[2]**2);
            const axialDist = Math.abs(pos[1]);
            
            if (radialDist <= focusRadius && axialDist <= focusHeight/2) {
                particlesInZone++;
            }
        });
        
        const focusEfficiency = particlesInZone / particleCount;
        
        // Mock magnetic field data
        const fieldSamples = 20;
        const fieldVectors = [];
        const samplePositions = [];
        
        for (let i = 0; i < fieldSamples; i++) {
            const x = (Math.random() - 0.5) * 0.01;
            const y = (Math.random() - 0.5) * 0.01;
            const z = (Math.random() - 0.5) * 0.01;
            
            samplePositions.push([x, y, z]);
            
            // Mock field pointing toward center
            const fieldStrength = 0.1; // Tesla
            const distance = Math.sqrt(x**2 + y**2 + z**2);
            
            if (distance > 0) {
                fieldVectors.push([
                    -x / distance * fieldStrength * (1 + 0.2 * Math.sin(time)),
                    -y / distance * fieldStrength * (1 + 0.2 * Math.sin(time + 1)),
                    -z / distance * fieldStrength * (1 + 0.2 * Math.sin(time + 2))
                ]);
            } else {
                fieldVectors.push([0, 0, 0]);
            }
        }
        
        // Mock coil data
        const coils = [
            {
                id: 'coil_0',
                position: [0, 0, 0.01],
                radius: 0.015,
                current: 1.0 + 0.2 * Math.sin(time * 0.5),
                max_current: 5.0,
                normal: [0, 0, -1]
            },
            {
                id: 'coil_1',
                position: [0.01, 0, 0.005],
                radius: 0.012,
                current: 0.8 + 0.1 * Math.cos(time * 0.7),
                max_current: 5.0,
                normal: [-0.7, 0, -0.7]
            },
            {
                id: 'coil_2',
                position: [-0.01, 0, 0.005],
                radius: 0.012,
                current: 0.8 + 0.1 * Math.sin(time * 0.6),
                max_current: 5.0,
                normal: [0.7, 0, -0.7]
            }
        ];
        
        return {
            time: time,
            step: step,
            particles: {
                positions: positions,
                velocities: velocities,
                forces: forces,
                radii: new Array(particleCount).fill(50e-9),
                shell_thickness: new Array(particleCount).fill(10e-9),
                ids: Array.from({length: particleCount}, (_, i) => `particle_${i}`)
            },
            magnetic_field: {
                field_vectors: fieldVectors,
                sample_positions: samplePositions,
                field_magnitudes: fieldVectors.map(v => Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2))
            },
            focus_zone: {
                center: [0, 0, 0],
                shape: 'cylinder',
                parameters: {
                    radius: focusRadius,
                    height: focusHeight,
                    axis: [0, 0, 1]
                },
                efficiency_history: [focusEfficiency],
                target_efficiency: 0.8
            },
            metrics: {
                focus_efficiency: focusEfficiency,
                bridge_formed: focusEfficiency > 0.7,
                particle_count_in_zone: particlesInZone
            },
            coils: coils
        };
    }

    async sendCommand(command, data = {}) {
        const message = {
            command: command,
            data: data,
            timestamp: Date.now()
        };
        
        if (this.isConnected && this.websocket) {
            this.websocket.send(JSON.stringify(message));
        } else if (this.useMockData) {
            // Handle mock commands
            this.handleMockCommand(command, data);
        } else {
            throw new Error('Not connected to simulation backend');
        }
    }

    handleMockCommand(command, data) {
        console.log(`Mock command: ${command}`, data);
        
        switch (command) {
            case 'start_simulation':
                this.simulationRunning = true;
                console.log('Mock simulation started');
                break;
                
            case 'pause_simulation':
                this.simulationRunning = false;
                console.log('Mock simulation paused');
                break;
                
            case 'reset_simulation':
                this.simulationRunning = false;
                console.log('Mock simulation reset');
                break;
                
            case 'set_speed':
                this.simulationSpeed = data.speed || 1.0;
                console.log(`Mock simulation speed: ${this.simulationSpeed}x`);
                break;
                
            case 'update_coil_current':
                console.log(`Mock coil ${data.coil_index} current: ${data.current}A`);
                break;
                
            case 'set_control_enabled':
                console.log(`Mock control enabled: ${data.enabled}`);
                break;
                
            case 'set_target_efficiency':
                console.log(`Mock target efficiency: ${data.efficiency}`);
                break;
        }
    }

    // Simulation control methods
    async startSimulation() {
        await this.sendCommand('start_simulation');
    }

    async pauseSimulation() {
        await this.sendCommand('pause_simulation');
    }

    async resetSimulation() {
        await this.sendCommand('reset_simulation');
    }

    async stepSimulation() {
        await this.sendCommand('step_simulation');
    }

    async setSpeed(speed) {
        await this.sendCommand('set_speed', { speed: speed });
    }

    async updateCoilCurrent(coilIndex, current) {
        await this.sendCommand('update_coil_current', {
            coil_index: coilIndex,
            current: current
        });
    }

    async setControlEnabled(enabled) {
        await this.sendCommand('set_control_enabled', { enabled: enabled });
    }

    async setTargetEfficiency(efficiency) {
        await this.sendCommand('set_target_efficiency', { efficiency: efficiency });
    }

    // Data export methods
    async exportData() {
        try {
            if (this.useMockData) {
                this.exportMockData();
                return;
            }
            
            const response = await fetch(`${this.httpUrl}/export`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: 'json',
                    include_history: true
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `magnano_export_${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
            } else {
                throw new Error('Export failed');
            }
            
        } catch (error) {
            console.error('Export error:', error);
            this.app.showError('Export failed: ' + error.message);
        }
    }

    exportMockData() {
        const exportData = {
            simulation_data: this.dataHistory,
            export_time: new Date().toISOString(),
            data_type: 'mock_data',
            total_samples: this.dataHistory.length
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `magnano_mock_export_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        console.log('Mock data exported');
    }

    async saveConfiguration() {
        try {
            const config = {
                visualization: this.app.particleSystem?.exportConfiguration() || {},
                field_viz: this.app.fieldVisualizer?.exportConfiguration() || {},
                renderer: this.app.renderer?.exportScene() || {},
                export_time: new Date().toISOString()
            };
            
            if (this.useMockData) {
                // Save locally for mock data
                const blob = new Blob([JSON.stringify(config, null, 2)], {
                    type: 'application/json'
                });
                
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `magnano_config_${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
                
                console.log('Configuration saved locally');
                return;
            }
            
            const response = await fetch(`${this.httpUrl}/save_config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                throw new Error('Save failed');
            }
            
            console.log('Configuration saved to backend');
            
        } catch (error) {
            console.error('Save configuration error:', error);
            this.app.showError('Save failed: ' + error.message);
        }
    }

    async loadConfiguration() {
        try {
            // Create file input
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            
            input.onchange = async (event) => {
                const file = event.target.files[0];
                if (!file) return;
                
                try {
                    const text = await file.text();
                    const config = JSON.parse(text);
                    
                    // Apply configuration
                    if (config.visualization && this.app.particleSystem) {
                        this.app.particleSystem.loadConfiguration(config.visualization);
                    }
                    
                    if (config.field_viz && this.app.fieldVisualizer) {
                        this.app.fieldVisualizer.loadConfiguration(config.field_viz);
                    }
                    
                    console.log('Configuration loaded successfully');
                    
                } catch (error) {
                    console.error('Load configuration error:', error);
                    this.app.showError('Load failed: ' + error.message);
                }
            };
            
            input.click();
            
        } catch (error) {
            console.error('Load configuration error:', error);
            this.app.showError('Load failed: ' + error.message);
        }
    }

    // Data query methods
    getLatestData() {
        return this.lastSimulationData;
    }

    getDataHistory(maxSamples = 100) {
        const history = this.dataHistory.slice(-maxSamples);
        return history.map(entry => entry.data);
    }

    getConnectionStatus() {
        return {
            connected: this.isConnected,
            mockMode: this.useMockData,
            reconnectAttempts: this.reconnectAttempts,
            dataPoints: this.dataHistory.length
        };
    }

    // Statistics and analysis
    calculateDataRate() {
        if (this.dataHistory.length < 2) return 0;
        
        const recent = this.dataHistory.slice(-10);
        const timeSpan = recent[recent.length - 1].timestamp - recent[0].timestamp;
        
        return (recent.length - 1) / (timeSpan / 1000); // Hz
    }

    getDataStatistics() {
        if (this.dataHistory.length === 0) return null;
        
        const latest = this.lastSimulationData;
        
        return {
            dataRate: this.calculateDataRate(),
            totalSamples: this.dataHistory.length,
            simulationTime: latest?.time || 0,
            simulationSteps: latest?.step || 0,
            particleCount: latest?.particles?.positions?.length || 0,
            focusEfficiency: latest?.metrics?.focus_efficiency || 0,
            bridgeFormed: latest?.metrics?.bridge_formed || false
        };
    }

    // Performance monitoring
    startPerformanceMonitoring() {
        setInterval(() => {
            const stats = this.getDataStatistics();
            if (stats) {
                console.log('Data Interface Stats:', {
                    dataRate: `${stats.dataRate.toFixed(1)} Hz`,
                    samples: stats.totalSamples,
                    simTime: `${stats.simulationTime.toFixed(3)}s`,
                    efficiency: `${(stats.focusEfficiency * 100).toFixed(1)}%`
                });
            }
        }, 10000); // Every 10 seconds
    }

    // Cleanup
    dispose() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        if (this.mockDataInterval) {
            clearInterval(this.mockDataInterval);
        }
        
        this.dataHistory = [];
        this.lastSimulationData = null;
        
        console.log('DataInterface disposed');
    }
}