/**
 * Controls Manager for MagNanoBridge visualization
 * Handles UI interactions and visualization controls
 */

class ControlsManager {
    constructor(app) {
        this.app = app;
        
        // Control elements
        this.controls = {};
        this.shortcuts = new Map();
        
        // State
        this.isInitialized = false;
        this.controlsLocked = false;
        
        // Settings
        this.settings = {
            enableKeyboardShortcuts: true,
            enableMouseInteraction: true,
            autoSaveSettings: true,
            settingsKey: 'magnano_settings'
        };
    }

    async initialize() {
        console.log('Initializing ControlsManager...');
        
        this.setupControlReferences();
        this.setupKeyboardShortcuts();
        this.setupMouseInteraction();
        this.loadSettings();
        
        this.isInitialized = true;
        console.log('ControlsManager initialized');
    }

    setupControlReferences() {
        // Simulation controls
        this.controls.playPause = document.getElementById('play-pause-btn');
        this.controls.reset = document.getElementById('reset-btn');
        this.controls.step = document.getElementById('step-btn');
        this.controls.speedSlider = document.getElementById('speed-slider');
        this.controls.speedValue = document.getElementById('speed-value');
        
        // Visualization toggles
        this.controls.showTrails = document.getElementById('show-trails');
        this.controls.showForces = document.getElementById('show-forces');
        this.controls.showField = document.getElementById('show-field');
        this.controls.showFocusZone = document.getElementById('show-focus-zone');
        
        // Camera controls
        this.controls.cameraTop = document.getElementById('camera-top');
        this.controls.cameraSide = document.getElementById('camera-side');
        this.controls.camera3D = document.getElementById('camera-3d');
        this.controls.autoRotate = document.getElementById('auto-rotate');
        
        // Coil controls
        this.controls.coils = [];
        for (let i = 0; i < 6; i++) {
            const slider = document.getElementById(`coil-${i}`);
            const value = document.getElementById(`coil-${i}-value`);
            if (slider && value) {
                this.controls.coils.push({ slider, value });
            }
        }
        
        // Focus control
        this.controls.enableControl = document.getElementById('enable-control');
        this.controls.targetEfficiency = document.getElementById('target-efficiency');
        this.controls.targetEfficiencyValue = document.getElementById('target-efficiency-value');
        
        // Data controls
        this.controls.exportData = document.getElementById('export-data');
        this.controls.saveConfig = document.getElementById('save-config');
        this.controls.loadConfig = document.getElementById('load-config');
    }

    setupKeyboardShortcuts() {
        if (!this.settings.enableKeyboardShortcuts) return;
        
        // Define keyboard shortcuts
        this.shortcuts.set('Space', () => this.app.toggleSimulation());
        this.shortcuts.set('KeyR', () => this.app.resetSimulation());
        this.shortcuts.set('KeyS', () => this.app.stepSimulation());
        
        // Camera shortcuts
        this.shortcuts.set('Digit1', () => this.app.renderer.setCameraView('top'));
        this.shortcuts.set('Digit2', () => this.app.renderer.setCameraView('side'));
        this.shortcuts.set('Digit3', () => this.app.renderer.setCameraView('3d'));
        
        // Visualization shortcuts
        this.shortcuts.set('KeyT', () => this.toggleTrails());
        this.shortcuts.set('KeyF', () => this.toggleForces());
        this.shortcuts.set('KeyM', () => this.toggleMagneticField());
        this.shortcuts.set('KeyZ', () => this.toggleFocusZone());
        
        // Speed control
        this.shortcuts.set('Equal', () => this.adjustSpeed(0.1)); // +
        this.shortcuts.set('Minus', () => this.adjustSpeed(-0.1)); // -
        
        // Export shortcuts
        this.shortcuts.set('KeyE', (event) => {
            if (event.ctrlKey) {
                event.preventDefault();
                this.app.exportData();
            }
        });
        
        this.shortcuts.set('KeyO', (event) => {
            if (event.ctrlKey) {
                event.preventDefault();
                this.app.loadConfiguration();
            }
        });
        
        this.shortcuts.set('KeyS', (event) => {
            if (event.ctrlKey) {
                event.preventDefault();
                this.app.saveConfiguration();
            }
        });
        
        // Add event listeners
        document.addEventListener('keydown', (event) => {
            if (this.controlsLocked) return;
            
            const shortcut = this.shortcuts.get(event.code);
            if (shortcut) {
                shortcut(event);
            }
        });
        
        console.log('Keyboard shortcuts enabled');
    }

    setupMouseInteraction() {
        if (!this.settings.enableMouseInteraction) return;
        
        const canvas = this.app.renderer?.renderer?.domElement;
        if (!canvas) return;
        
        // Mouse picking for particle selection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        canvas.addEventListener('click', (event) => {
            if (this.controlsLocked) return;
            
            // Calculate mouse position in normalized device coordinates
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            // Cast ray from camera
            raycaster.setFromCamera(mouse, this.app.renderer.camera);
            
            // Check for intersections with particles
            const particleGroup = this.app.particleSystem.particleGroup;
            if (particleGroup) {
                const intersects = raycaster.intersectObjects(particleGroup.children);
                
                if (intersects.length > 0) {
                    const selectedObject = intersects[0].object;
                    const particleId = selectedObject.userData.id;
                    
                    this.selectParticle(particleId);
                }
            }
        });
        
        // Context menu for advanced options
        canvas.addEventListener('contextmenu', (event) => {
            event.preventDefault();
            this.showContextMenu(event.clientX, event.clientY);
        });
        
        console.log('Mouse interaction enabled');
    }

    selectParticle(particleId) {
        // Highlight selected particle
        if (this.selectedParticle) {
            this.app.particleSystem.highlightParticle(this.selectedParticle, false);
        }
        
        this.selectedParticle = particleId;
        this.app.particleSystem.highlightParticle(particleId, true);
        
        // Show particle info
        this.showParticleInfo(particleId);
        
        console.log(`Selected particle: ${particleId}`);
    }

    showParticleInfo(particleId) {
        const particle = this.app.particleSystem.getParticleById(particleId);
        if (!particle) return;
        
        // Create info popup
        const popup = document.createElement('div');
        popup.id = 'particle-info-popup';
        popup.style.cssText = `
            position: fixed;
            top: 20px;
            right: 280px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #3dd8ff;
            z-index: 1000;
            min-width: 200px;
            font-size: 12px;
        `;
        
        popup.innerHTML = `
            <h4 style="margin: 0 0 10px 0; color: #3dd8ff;">Particle ${particleId}</h4>
            <div>Position: ${particle.position.map(p => p.toExponential(2)).join(', ')}</div>
            <div>Radius: ${(particle.radius * 1e9).toFixed(1)} nm</div>
            <div>Shell: ${(particle.shellThickness * 1e9).toFixed(1)} nm</div>
            <button onclick="this.parentElement.remove()" style="margin-top: 10px; padding: 4px 8px; background: #666; border: none; color: white; border-radius: 4px; cursor: pointer;">Close</button>
        `;
        
        // Remove existing popup
        const existing = document.getElementById('particle-info-popup');
        if (existing) existing.remove();
        
        document.body.appendChild(popup);
    }

    showContextMenu(x, y) {
        const menu = document.createElement('div');
        menu.id = 'context-menu';
        menu.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #666;
            border-radius: 6px;
            padding: 8px 0;
            z-index: 10000;
            min-width: 150px;
        `;
        
        const menuItems = [
            { label: 'Reset Camera', action: () => this.app.renderer.setCameraView('3d') },
            { label: 'Take Screenshot', action: () => this.takeScreenshot() },
            { label: 'Toggle Wireframe', action: () => this.toggleWireframe() },
            { label: 'Show Statistics', action: () => this.showStatistics() }
        ];
        
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.style.cssText = `
                padding: 8px 16px;
                cursor: pointer;
                color: white;
                font-size: 12px;
            `;
            menuItem.textContent = item.label;
            
            menuItem.addEventListener('mouseenter', () => {
                menuItem.style.backgroundColor = '#333';
            });
            
            menuItem.addEventListener('mouseleave', () => {
                menuItem.style.backgroundColor = 'transparent';
            });
            
            menuItem.addEventListener('click', () => {
                item.action();
                menu.remove();
            });
            
            menu.appendChild(menuItem);
        });
        
        // Remove existing menu
        const existing = document.getElementById('context-menu');
        if (existing) existing.remove();
        
        document.body.appendChild(menu);
        
        // Remove menu when clicking elsewhere
        setTimeout(() => {
            document.addEventListener('click', () => menu.remove(), { once: true });
        }, 100);
    }

    // Control action methods
    toggleTrails() {
        const checkbox = this.controls.showTrails;
        checkbox.checked = !checkbox.checked;
        this.app.particleSystem.setTrailsVisible(checkbox.checked);
    }

    toggleForces() {
        const checkbox = this.controls.showForces;
        checkbox.checked = !checkbox.checked;
        this.app.particleSystem.setForcesVisible(checkbox.checked);
    }

    toggleMagneticField() {
        const checkbox = this.controls.showField;
        checkbox.checked = !checkbox.checked;
        this.app.fieldVisualizer.setVisible(checkbox.checked);
    }

    toggleFocusZone() {
        const checkbox = this.controls.showFocusZone;
        checkbox.checked = !checkbox.checked;
        this.app.renderer.setFocusZoneVisible(checkbox.checked);
    }

    adjustSpeed(delta) {
        const slider = this.controls.speedSlider;
        const newValue = Math.max(0.1, Math.min(5.0, parseFloat(slider.value) + delta));
        slider.value = newValue;
        
        const valueDisplay = this.controls.speedValue;
        valueDisplay.textContent = newValue.toFixed(1) + 'x';
        
        this.app.setSimulationSpeed(newValue);
    }

    takeScreenshot() {
        const imageData = this.app.renderer.screenshot();
        
        // Create download link
        const link = document.createElement('a');
        link.download = `magnano_screenshot_${Date.now()}.png`;
        link.href = imageData;
        link.click();
        
        console.log('Screenshot taken');
    }

    toggleWireframe() {
        this.app.renderer.toggleWireframe();
        console.log('Wireframe toggled');
    }

    showStatistics() {
        const stats = {
            particle: this.app.particleSystem.getStats(),
            field: this.app.fieldVisualizer.getStats(),
            renderer: this.app.renderer.getPerformanceStats(),
            data: this.app.dataInterface.getDataStatistics()
        };
        
        this.showStatsDialog(stats);
    }

    showStatsDialog(stats) {
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.95);
            color: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #3dd8ff;
            z-index: 10000;
            max-width: 400px;
            max-height: 80vh;
            overflow-y: auto;
        `;
        
        let content = '<h3 style="margin: 0 0 15px 0; color: #3dd8ff;">System Statistics</h3>';
        
        Object.entries(stats).forEach(([category, data]) => {
            if (!data) return;
            
            content += `<h4 style="margin: 15px 0 8px 0; color: #ff6b35;">${category.toUpperCase()}</h4>`;
            
            Object.entries(data).forEach(([key, value]) => {
                const displayValue = typeof value === 'number' ? 
                    (value < 0.01 ? value.toExponential(2) : value.toFixed(2)) : 
                    value;
                content += `<div style="margin: 4px 0;"><span style="color: #ccc;">${key}:</span> ${displayValue}</div>`;
            });
        });
        
        content += `
            <button onclick="this.parentElement.remove()" 
                    style="margin-top: 15px; padding: 8px 16px; background: #666; border: none; color: white; border-radius: 4px; cursor: pointer;">
                Close
            </button>
        `;
        
        dialog.innerHTML = content;
        document.body.appendChild(dialog);
    }

    // Settings management
    saveSettings() {
        if (!this.settings.autoSaveSettings) return;
        
        const settings = {
            visualization: {
                showTrails: this.controls.showTrails?.checked || false,
                showForces: this.controls.showForces?.checked || false,
                showField: this.controls.showField?.checked || false,
                showFocusZone: this.controls.showFocusZone?.checked || false,
                autoRotate: this.controls.autoRotate?.checked || false
            },
            simulation: {
                speed: parseFloat(this.controls.speedSlider?.value || 1.0)
            },
            camera: this.app.renderer?.exportScene()?.camera || {},
            coils: this.controls.coils.map(coil => parseFloat(coil.slider?.value || 0))
        };
        
        localStorage.setItem(this.settings.settingsKey, JSON.stringify(settings));
        console.log('Settings saved');
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem(this.settings.settingsKey);
            if (!saved) return;
            
            const settings = JSON.parse(saved);
            
            // Apply visualization settings
            if (settings.visualization) {
                const viz = settings.visualization;
                
                if (this.controls.showTrails) this.controls.showTrails.checked = viz.showTrails;
                if (this.controls.showForces) this.controls.showForces.checked = viz.showForces;
                if (this.controls.showField) this.controls.showField.checked = viz.showField;
                if (this.controls.showFocusZone) this.controls.showFocusZone.checked = viz.showFocusZone;
                if (this.controls.autoRotate) this.controls.autoRotate.checked = viz.autoRotate;
            }
            
            // Apply simulation settings
            if (settings.simulation && this.controls.speedSlider) {
                this.controls.speedSlider.value = settings.simulation.speed;
                this.controls.speedValue.textContent = settings.simulation.speed.toFixed(1) + 'x';
            }
            
            // Apply coil settings
            if (settings.coils) {
                settings.coils.forEach((current, index) => {
                    if (this.controls.coils[index]) {
                        this.controls.coils[index].slider.value = current;
                        this.controls.coils[index].value.textContent = current.toFixed(1) + 'A';
                    }
                });
            }
            
            console.log('Settings loaded');
            
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    // Control state management
    lockControls() {
        this.controlsLocked = true;
        
        // Disable all interactive controls
        Object.values(this.controls).forEach(control => {
            if (control && typeof control.disabled !== 'undefined') {
                control.disabled = true;
            }
        });
        
        console.log('Controls locked');
    }

    unlockControls() {
        this.controlsLocked = false;
        
        // Re-enable all interactive controls
        Object.values(this.controls).forEach(control => {
            if (control && typeof control.disabled !== 'undefined') {
                control.disabled = false;
            }
        });
        
        console.log('Controls unlocked');
    }

    // Advanced control features
    createCustomControlPanel() {
        const panel = document.createElement('div');
        panel.id = 'custom-control-panel';
        panel.style.cssText = `
            position: fixed;
            top: 50%;
            left: 20px;
            transform: translateY(-50%);
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #3dd8ff;
            border-radius: 10px;
            padding: 20px;
            width: 250px;
            z-index: 1000;
        `;
        
        // Particle size control
        const particleSizeControl = this.createSliderControl(
            'Particle Size',
            0.5, 3.0, 1.0, 0.1,
            (value) => {
                this.app.particleSystem.setParticleSize(value);
            }
        );
        
        // Trail length control
        const trailLengthControl = this.createSliderControl(
            'Trail Length',
            10, 200, 50, 10,
            (value) => {
                this.app.particleSystem.setTrailLength(value);
            }
        );
        
        // Force scale control
        const forceScaleControl = this.createSliderControl(
            'Force Scale',
            1e10, 1e14, 1e12, 1e11,
            (value) => {
                this.app.particleSystem.setForceScale(value);
            }
        );
        
        // Field visualization controls
        const fieldControls = this.createFieldControls();
        
        panel.appendChild(particleSizeControl);
        panel.appendChild(trailLengthControl);
        panel.appendChild(forceScaleControl);
        panel.appendChild(fieldControls);
        
        // Close button
        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.style.cssText = `
            width: 100%;
            margin-top: 15px;
            padding: 8px;
            background: #666;
            border: none;
            color: white;
            border-radius: 4px;
            cursor: pointer;
        `;
        closeButton.onclick = () => panel.remove();
        panel.appendChild(closeButton);
        
        document.body.appendChild(panel);
    }

    createSliderControl(label, min, max, value, step, onChange) {
        const container = document.createElement('div');
        container.style.marginBottom = '15px';
        
        const labelEl = document.createElement('label');
        labelEl.textContent = label;
        labelEl.style.cssText = 'display: block; color: white; font-size: 12px; margin-bottom: 5px;';
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = min;
        slider.max = max;
        slider.value = value;
        slider.step = step;
        slider.style.cssText = 'width: 100%;';
        
        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = value;
        valueDisplay.style.cssText = 'color: #3dd8ff; font-size: 11px;';
        
        slider.oninput = (e) => {
            const val = parseFloat(e.target.value);
            valueDisplay.textContent = val.toExponential ? 
                (val > 1000 ? val.toExponential(1) : val.toFixed(1)) : val;
            onChange(val);
        };
        
        container.appendChild(labelEl);
        container.appendChild(slider);
        container.appendChild(valueDisplay);
        
        return container;
    }

    createFieldControls() {
        const container = document.createElement('div');
        container.style.marginBottom = '15px';
        
        const title = document.createElement('h4');
        title.textContent = 'Field Visualization';
        title.style.cssText = 'color: #3dd8ff; margin: 0 0 10px 0; font-size: 14px;';
        
        // Field lines toggle
        const fieldLinesToggle = this.createCheckboxControl(
            'Show Field Lines',
            true,
            (checked) => {
                this.app.fieldVisualizer.setFieldLinesVisible(checked);
            }
        );
        
        // Field arrows toggle
        const fieldArrowsToggle = this.createCheckboxControl(
            'Show Field Arrows',
            true,
            (checked) => {
                this.app.fieldVisualizer.setFieldArrowsVisible(checked);
            }
        );
        
        // Color scheme selector
        const colorSchemeSelect = document.createElement('select');
        colorSchemeSelect.style.cssText = 'width: 100%; background: #333; color: white; border: 1px solid #666; padding: 4px;';
        
        const schemes = [
            { value: 'magnitude', label: 'Magnitude' },
            { value: 'direction', label: 'Direction' }
        ];
        
        schemes.forEach(scheme => {
            const option = document.createElement('option');
            option.value = scheme.value;
            option.textContent = scheme.label;
            colorSchemeSelect.appendChild(option);
        });
        
        colorSchemeSelect.onchange = (e) => {
            this.app.fieldVisualizer.setFieldColorScheme(e.target.value);
        };
        
        const colorLabel = document.createElement('label');
        colorLabel.textContent = 'Color Scheme';
        colorLabel.style.cssText = 'display: block; color: white; font-size: 12px; margin: 10px 0 5px 0;';
        
        container.appendChild(title);
        container.appendChild(fieldLinesToggle);
        container.appendChild(fieldArrowsToggle);
        container.appendChild(colorLabel);
        container.appendChild(colorSchemeSelect);
        
        return container;
    }

    createCheckboxControl(label, checked, onChange) {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; align-items: center; margin-bottom: 8px;';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = checked;
        checkbox.style.marginRight = '8px';
        checkbox.onchange = (e) => onChange(e.target.checked);
        
        const labelEl = document.createElement('label');
        labelEl.textContent = label;
        labelEl.style.cssText = 'color: white; font-size: 12px; cursor: pointer;';
        labelEl.onclick = () => {
            checkbox.checked = !checkbox.checked;
            onChange(checkbox.checked);
        };
        
        container.appendChild(checkbox);
        container.appendChild(labelEl);
        
        return container;
    }

    // Preset management
    savePreset(name) {
        const preset = {
            name: name,
            created: new Date().toISOString(),
            settings: {
                visualization: this.getVisualizationSettings(),
                simulation: this.getSimulationSettings(),
                camera: this.getCameraSettings(),
                coils: this.getCoilSettings()
            }
        };
        
        const presets = JSON.parse(localStorage.getItem('magnano_presets') || '{}');
        presets[name] = preset;
        localStorage.setItem('magnano_presets', JSON.stringify(presets));
        
        console.log(`Preset '${name}' saved`);
    }

    loadPreset(name) {
        const presets = JSON.parse(localStorage.getItem('magnano_presets') || '{}');
        const preset = presets[name];
        
        if (!preset) {
            console.error(`Preset '${name}' not found`);
            return;
        }
        
        this.applySettings(preset.settings);
        console.log(`Preset '${name}' loaded`);
    }

    getAvailablePresets() {
        const presets = JSON.parse(localStorage.getItem('magnano_presets') || '{}');
        return Object.keys(presets);
    }

    getVisualizationSettings() {
        return {
            showTrails: this.controls.showTrails?.checked || false,
            showForces: this.controls.showForces?.checked || false,
            showField: this.controls.showField?.checked || false,
            showFocusZone: this.controls.showFocusZone?.checked || false,
            autoRotate: this.controls.autoRotate?.checked || false
        };
    }

    getSimulationSettings() {
        return {
            speed: parseFloat(this.controls.speedSlider?.value || 1.0)
        };
    }

    getCameraSettings() {
        return this.app.renderer?.exportScene()?.camera || {};
    }

    getCoilSettings() {
        return this.controls.coils.map(coil => ({
            current: parseFloat(coil.slider?.value || 0)
        }));
    }

    applySettings(settings) {
        // Apply visualization settings
        if (settings.visualization) {
            const viz = settings.visualization;
            if (this.controls.showTrails) {
                this.controls.showTrails.checked = viz.showTrails;
                this.app.particleSystem.setTrailsVisible(viz.showTrails);
            }
            if (this.controls.showForces) {
                this.controls.showForces.checked = viz.showForces;
                this.app.particleSystem.setForcesVisible(viz.showForces);
            }
            if (this.controls.showField) {
                this.controls.showField.checked = viz.showField;
                this.app.fieldVisualizer.setVisible(viz.showField);
            }
            if (this.controls.showFocusZone) {
                this.controls.showFocusZone.checked = viz.showFocusZone;
                this.app.renderer.setFocusZoneVisible(viz.showFocusZone);
            }
            if (this.controls.autoRotate) {
                this.controls.autoRotate.checked = viz.autoRotate;
                this.app.renderer.setAutoRotate(viz.autoRotate);
            }
        }
        
        // Apply simulation settings
        if (settings.simulation && this.controls.speedSlider) {
            this.controls.speedSlider.value = settings.simulation.speed;
            this.controls.speedValue.textContent = settings.simulation.speed.toFixed(1) + 'x';
            this.app.setSimulationSpeed(settings.simulation.speed);
        }
        
        // Apply coil settings
        if (settings.coils) {
            settings.coils.forEach((coilSetting, index) => {
                if (this.controls.coils[index] && coilSetting.current !== undefined) {
                    this.controls.coils[index].slider.value = coilSetting.current;
                    this.controls.coils[index].value.textContent = coilSetting.current.toFixed(1) + 'A';
                    this.app.updateCoilCurrent(index, coilSetting.current);
                }
            });
        }
    }

    // Help system
    showHelpDialog() {
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.95);
            color: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #3dd8ff;
            z-index: 10000;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
        `;
        
        dialog.innerHTML = `
            <h3 style="margin: 0 0 15px 0; color: #3dd8ff;">MagNanoBridge Help</h3>
            
            <h4 style="color: #ff6b35; margin: 15px 0 8px 0;">Keyboard Shortcuts</h4>
            <div style="font-size: 12px; line-height: 1.4;">
                <strong>Space:</strong> Play/Pause simulation<br>
                <strong>R:</strong> Reset simulation<br>
                <strong>S:</strong> Step simulation<br>
                <strong>1:</strong> Top camera view<br>
                <strong>2:</strong> Side camera view<br>
                <strong>3:</strong> 3D camera view<br>
                <strong>T:</strong> Toggle particle trails<br>
                <strong>F:</strong> Toggle force vectors<br>
                <strong>M:</strong> Toggle magnetic field<br>
                <strong>Z:</strong> Toggle focus zone<br>
                <strong>+/-:</strong> Adjust simulation speed<br>
                <strong>Ctrl+E:</strong> Export data<br>
                <strong>Ctrl+S:</strong> Save configuration<br>
                <strong>Ctrl+O:</strong> Load configuration<br>
            </div>
            
            <h4 style="color: #ff6b35; margin: 15px 0 8px 0;">Mouse Controls</h4>
            <div style="font-size: 12px; line-height: 1.4;">
                <strong>Left Click + Drag:</strong> Rotate camera<br>
                <strong>Mouse Wheel:</strong> Zoom in/out<br>
                <strong>Left Click on Particle:</strong> Select and highlight<br>
                <strong>Right Click:</strong> Context menu<br>
            </div>
            
            <h4 style="color: #ff6b35; margin: 15px 0 8px 0;">Visualization</h4>
            <div style="font-size: 12px; line-height: 1.4;">
                • <strong>Orange spheres:</strong> Particle cores<br>
                • <strong>Yellow transparent:</strong> Polymer shells<br>
                • <strong>Blue lines:</strong> Particle trails<br>
                • <strong>Red arrows:</strong> Force vectors<br>
                • <strong>Cyan cylinder:</strong> Focus zone<br>
                • <strong>Gray toroids:</strong> Magnetic coils<br>
            </div>
            
            <button onclick="this.parentElement.remove()" 
                    style="margin-top: 15px; padding: 8px 16px; background: #666; border: none; color: white; border-radius: 4px; cursor: pointer; width: 100%;">
                Close
            </button>
        `;
        
        document.body.appendChild(dialog);
    }

    // Cleanup
    dispose() {
        // Save settings before cleanup
        this.saveSettings();
        
        // Remove event listeners
        document.removeEventListener('keydown', null);
        
        // Clear shortcuts
        this.shortcuts.clear();
        
        // Clear control references
        this.controls = {};
        
        console.log('ControlsManager disposed');
    }

    // Debug and development helpers
    debugControls() {
        console.log('Controls Debug Info:', {
            initialized: this.isInitialized,
            locked: this.controlsLocked,
            shortcuts: Array.from(this.shortcuts.keys()),
            controlElements: Object.keys(this.controls),
            settings: this.settings
        });
    }

    simulateKeyPress(keyCode) {
        const event = { code: keyCode, preventDefault: () => {} };
        const shortcut = this.shortcuts.get(keyCode);
        if (shortcut) {
            shortcut(event);
            console.log(`Simulated key press: ${keyCode}`);
        }
    }
}