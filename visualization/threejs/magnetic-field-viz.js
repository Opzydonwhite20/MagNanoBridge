/**
 * Magnetic Field Visualizer for MagNanoBridge
 * Renders magnetic field lines, coils, and field strength indicators
 */

class MagneticFieldVisualizer {
    constructor(scene) {
        this.scene = scene;
        
        // Visual groups
        this.fieldGroup = null;
        this.coilsGroup = null;
        this.fieldLinesGroup = null;
        this.fieldArrowsGroup = null;
        
        // Field data
        this.fieldData = null;
        this.coilData = [];
        
        // Visualization settings
        this.settings = {
            showFieldLines: true,
            showFieldArrows: true,
            showCoils: true,
            fieldLineOpacity: 0.6,
            fieldArrowScale: 1000,
            coilOpacity: 0.8,
            fieldColorScheme: 'magnitude', // 'magnitude' or 'direction'
            maxFieldLines: 50,
            fieldSampleResolution: 10
        };
        
        // Materials
        this.fieldLineMaterial = null;
        this.fieldArrowMaterial = null;
        this.coilMaterial = null;
        
        // Geometries
        this.arrowGeometry = null;
        this.coilGeometry = null;
        
        // Performance
        this.lastUpdateTime = 0;
        this.updateInterval = 100; // ms between updates
    }

    async initialize() {
        console.log('Initializing MagneticFieldVisualizer...');
        
        this.createGroups();
        this.createMaterials();
        this.createGeometries();
        
        console.log('MagneticFieldVisualizer initialized');
    }

    createGroups() {
        // Main field group
        this.fieldGroup = new THREE.Group();
        this.fieldGroup.name = 'magnetic_field';
        this.scene.add(this.fieldGroup);
        
        // Coils group
        this.coilsGroup = new THREE.Group();
        this.coilsGroup.name = 'magnetic_coils';
        this.fieldGroup.add(this.coilsGroup);
        
        // Field lines group
        this.fieldLinesGroup = new THREE.Group();
        this.fieldLinesGroup.name = 'field_lines';
        this.fieldGroup.add(this.fieldLinesGroup);
        
        // Field arrows group
        this.fieldArrowsGroup = new THREE.Group();
        this.fieldArrowsGroup.name = 'field_arrows';
        this.fieldGroup.add(this.fieldArrowsGroup);
    }

    createMaterials() {
        // Field line material
        this.fieldLineMaterial = new THREE.LineBasicMaterial({
            transparent: true,
            opacity: this.settings.fieldLineOpacity,
            vertexColors: true,
            linewidth: 2
        });
        
        // Field arrow material
        this.fieldArrowMaterial = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.7,
            vertexColors: true
        });
        
        // Coil material
        this.coilMaterial = new THREE.MeshLambertMaterial({
            color: 0x666666,
            transparent: true,
            opacity: this.settings.coilOpacity,
            metalness: 0.8,
            roughness: 0.2
        });
    }

    createGeometries() {
        // Arrow geometry for field vectors
        this.createArrowGeometry();
        
        // Coil geometry (torus)
        this.coilGeometry = new THREE.TorusGeometry(1, 0.05, 8, 16);
    }

    createArrowGeometry() {
        const coneGeometry = new THREE.ConeGeometry(0.0002, 0.001, 8);
        const cylinderGeometry = new THREE.CylinderGeometry(0.0001, 0.0001, 0.003, 6);
        
        // Position cone at tip
        coneGeometry.translate(0, 0.0015, 0);
        
        // Merge geometries
        const arrowMesh = new THREE.Mesh(coneGeometry);
        const shaftMesh = new THREE.Mesh(cylinderGeometry);
        
        arrowMesh.updateMatrix();
        shaftMesh.updateMatrix();
        
        this.arrowGeometry = new THREE.BufferGeometry();
        this.arrowGeometry = THREE.BufferGeometryUtils.mergeBufferGeometries([
            coneGeometry,
            cylinderGeometry
        ]);
    }

    updateField(fieldData) {
        this.fieldData = fieldData;
        
        if (this.settings.showFieldArrows) {
            this.updateFieldArrows();
        }
        
        if (this.settings.showFieldLines) {
            this.updateFieldLines();
        }
    }

    updateFieldArrows() {
        // Clear existing arrows
        this.fieldArrowsGroup.clear();
        
        if (!this.fieldData || !this.fieldData.field_vectors) return;
        
        const positions = this.fieldData.sample_positions || [];
        const fieldVectors = this.fieldData.field_vectors;
        const fieldMagnitudes = this.fieldData.field_magnitudes || [];
        
        // Create arrow for each field sample
        for (let i = 0; i < Math.min(positions.length, fieldVectors.length); i++) {
            const position = positions[i];
            const field = fieldVectors[i];
            const magnitude = fieldMagnitudes[i] || this.calculateMagnitude(field);
            
            if (magnitude < 1e-10) continue; // Skip negligible fields
            
            // Create arrow mesh
            const arrowMesh = new THREE.Mesh(
                this.arrowGeometry.clone(),
                this.fieldArrowMaterial.clone()
            );
            
            // Position arrow
            arrowMesh.position.set(position[0], position[1], position[2]);
            
            // Scale based on field magnitude
            const scale = Math.min(magnitude * this.settings.fieldArrowScale, 0.01);
            arrowMesh.scale.setScalar(scale);
            
            // Orient arrow in field direction
            const fieldDirection = new THREE.Vector3(field[0], field[1], field[2]).normalize();
            arrowMesh.lookAt(
                arrowMesh.position.x + fieldDirection.x,
                arrowMesh.position.y + fieldDirection.y,
                arrowMesh.position.z + fieldDirection.z
            );
            
            // Color based on field strength
            const color = this.getFieldColor(magnitude, field);
            arrowMesh.material.color = color;
            
            this.fieldArrowsGroup.add(arrowMesh);
        }
    }

    updateFieldLines() {
        // Clear existing field lines
        this.fieldLinesGroup.clear();
        
        if (!this.fieldData) return;
        
        // Generate field lines using streamline integration
        this.generateFieldLines();
    }

    generateFieldLines() {
        const startingPoints = this.generateStartingPoints();
        
        startingPoints.forEach((startPoint, index) => {
            const fieldLine = this.traceFieldLine(startPoint);
            if (fieldLine.length > 2) {
                this.createFieldLineGeometry(fieldLine, index);
            }
        });
    }

    generateStartingPoints() {
        const points = [];
        const gridSize = Math.cbrt(this.settings.maxFieldLines);
        const spacing = 0.01 / gridSize; // 10mm total range
        
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                for (let z = 0; z < gridSize; z++) {
                    points.push([
                        (x - gridSize/2) * spacing,
                        (y - gridSize/2) * spacing,
                        (z - gridSize/2) * spacing
                    ]);
                }
            }
        }
        
        return points.slice(0, this.settings.maxFieldLines);
    }

    traceFieldLine(startPoint) {
        const line = [startPoint];
        let currentPoint = [...startPoint];
        const stepSize = 0.0001; // 0.1mm steps
        const maxSteps = 100;
        
        for (let step = 0; step < maxSteps; step++) {
            const field = this.interpolateField(currentPoint);
            if (!field || this.calculateMagnitude(field) < 1e-12) break;
            
            // Normalize field direction
            const magnitude = this.calculateMagnitude(field);
            const direction = [
                field[0] / magnitude,
                field[1] / magnitude,
                field[2] / magnitude
            ];
            
            // Take step in field direction
            currentPoint = [
                currentPoint[0] + direction[0] * stepSize,
                currentPoint[1] + direction[1] * stepSize,
                currentPoint[2] + direction[2] * stepSize
            ];
            
            line.push([...currentPoint]);
            
            // Stop if we've gone too far
            const distance = Math.sqrt(
                currentPoint[0]**2 + currentPoint[1]**2 + currentPoint[2]**2
            );
            if (distance > 0.05) break; // 50mm limit
        }
        
        return line;
    }

    interpolateField(position) {
        // Simple field interpolation - in practice would use more sophisticated method
        if (!this.fieldData || !this.fieldData.field_vectors) return null;
        
        // For now, return field from nearest sample point
        const samplePositions = this.fieldData.sample_positions || [];
        const fieldVectors = this.fieldData.field_vectors;
        
        let nearestIndex = 0;
        let minDistance = Infinity;
        
        for (let i = 0; i < samplePositions.length; i++) {
            const samplePos = samplePositions[i];
            const distance = Math.sqrt(
                (position[0] - samplePos[0])**2 +
                (position[1] - samplePos[1])**2 +
                (position[2] - samplePos[2])**2
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestIndex = i;
            }
        }
        
        return fieldVectors[nearestIndex] || null;
    }

    createFieldLineGeometry(linePoints, lineIndex) {
        const points = linePoints.map(point => 
            new THREE.Vector3(point[0], point[1], point[2])
        );
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        // Create colors for each point based on field strength
        const colors = [];
        linePoints.forEach(point => {
            const field = this.interpolateField(point);
            const magnitude = field ? this.calculateMagnitude(field) : 0;
            const color = this.getFieldColor(magnitude, field);
            colors.push(color.r, color.g, color.b);
        });
        
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const line = new THREE.Line(geometry, this.fieldLineMaterial.clone());
        line.name = `field_line_${lineIndex}`;
        
        this.fieldLinesGroup.add(line);
    }

    updateCoils(coilData) {
        this.coilData = coilData;
        this.renderCoils();
    }

    renderCoils() {
        // Clear existing coils
        this.coilsGroup.clear();
        
        if (!this.settings.showCoils) return;
        
        this.coilData.forEach((coil, index) => {
            this.createCoilMesh(coil, index);
        });
    }

    createCoilMesh(coilData, index) {
        const coilMesh = new THREE.Mesh(
            this.coilGeometry.clone(),
            this.coilMaterial.clone()
        );
        
        // Position coil
        if (coilData.position) {
            coilMesh.position.set(
                coilData.position[0],
                coilData.position[1],
                coilData.position[2]
            );
        }
        
        // Scale coil to correct size
        if (coilData.radius) {
            coilMesh.scale.setScalar(coilData.radius);
        }
        
        // Orient coil based on normal vector
        if (coilData.normal) {
            const normal = new THREE.Vector3(
                coilData.normal[0],
                coilData.normal[1],
                coilData.normal[2]
            );
            coilMesh.lookAt(
                coilMesh.position.x + normal.x,
                coilMesh.position.y + normal.y,
                coilMesh.position.z + normal.z
            );
        }
        
        // Color based on current
        const current = coilData.current || 0;
        const maxCurrent = coilData.max_current || 5;
        const normalizedCurrent = Math.abs(current) / maxCurrent;
        
        const color = new THREE.Color();
        if (current > 0) {
            color.setHSL(0.6, 1.0, 0.3 + normalizedCurrent * 0.4); // Blue tones
        } else {
            color.setHSL(0.0, 1.0, 0.3 + normalizedCurrent * 0.4); // Red tones
        }
        
        coilMesh.material.color = color;
        coilMesh.material.emissive = color.clone().multiplyScalar(0.2);
        
        // Add current indicator
        this.addCurrentIndicator(coilMesh, current, maxCurrent);
        
        coilMesh.name = `coil_${coilData.id || index}`;
        coilMesh.userData = { coilData, index };
        
        this.coilsGroup.add(coilMesh);
    }

    addCurrentIndicator(coilMesh, current, maxCurrent) {
        // Create text indicator for current (simplified - would use text geometry in full implementation)
        const indicatorGeometry = new THREE.SphereGeometry(0.0005, 8, 8);
        const indicatorMaterial = new THREE.MeshBasicMaterial({
            color: current > 0 ? 0x00ff00 : 0xff0000,
            transparent: true,
            opacity: Math.abs(current) / maxCurrent
        });
        
        const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
        indicator.position.copy(coilMesh.position);
        indicator.position.y += coilMesh.scale.x * 1.2; // Position above coil
        
        coilMesh.add(indicator);
    }

    getFieldColor(magnitude, fieldVector) {
        const color = new THREE.Color();
        
        if (this.settings.fieldColorScheme === 'magnitude') {
            // Color based on field magnitude
            const normalizedMag = Math.min(magnitude * 1e3, 1.0); // Scale for Tesla
            color.setHSL((1 - normalizedMag) * 0.67, 1.0, 0.5); // Blue to red
        } else if (this.settings.fieldColorScheme === 'direction') {
            // Color based on field direction
            if (fieldVector) {
                const normalized = this.normalizeVector(fieldVector);
                color.setRGB(
                    Math.abs(normalized[0]),
                    Math.abs(normalized[1]),
                    Math.abs(normalized[2])
                );
            }
        }
        
        return color;
    }

    calculateMagnitude(vector) {
        return Math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2);
    }

    normalizeVector(vector) {
        const magnitude = this.calculateMagnitude(vector);
        if (magnitude === 0) return [0, 0, 0];
        return [
            vector[0] / magnitude,
            vector[1] / magnitude,
            vector[2] / magnitude
        ];
    }

    update(currentTime) {
        // Animate field visualization
        if (currentTime - this.lastUpdateTime < this.updateInterval) return;
        
        // Animate field lines with flowing effect
        this.fieldLinesGroup.children.forEach((line, index) => {
            if (line.material) {
                const time = currentTime * 0.001;
                const phase = index * 0.2;
                const opacity = this.settings.fieldLineOpacity * (0.7 + 0.3 * Math.sin(time + phase));
                line.material.opacity = opacity;
            }
        });
        
        // Animate coil indicators
        this.coilsGroup.children.forEach((coil, index) => {
            const indicator = coil.children[0];
            if (indicator) {
                const time = currentTime * 0.001;
                const phase = index * 0.5;
                indicator.scale.setScalar(1.0 + 0.2 * Math.sin(time * 3 + phase));
            }
        });
        
        this.lastUpdateTime = currentTime;
    }

    setVisible(visible) {
        this.fieldGroup.visible = visible;
    }

    setFieldLinesVisible(visible) {
        this.settings.showFieldLines = visible;
        this.fieldLinesGroup.visible = visible;
        
        if (visible && this.fieldData) {
            this.updateFieldLines();
        }
    }

    setFieldArrowsVisible(visible) {
        this.settings.showFieldArrows = visible;
        this.fieldArrowsGroup.visible = visible;
        
        if (visible && this.fieldData) {
            this.updateFieldArrows();
        }
    }

    setCoilsVisible(visible) {
        this.settings.showCoils = visible;
        this.coilsGroup.visible = visible;
        
        if (visible && this.coilData.length > 0) {
            this.renderCoils();
        }
    }

    setFieldColorScheme(scheme) {
        this.settings.fieldColorScheme = scheme;
        
        // Update existing visualization
        if (this.fieldData) {
            this.updateField(this.fieldData);
        }
    }

    setFieldLineOpacity(opacity) {
        this.settings.fieldLineOpacity = opacity;
        this.fieldLineMaterial.opacity = opacity;
        
        // Update existing field lines
        this.fieldLinesGroup.children.forEach(line => {
            if (line.material) {
                line.material.opacity = opacity;
            }
        });
    }

    setArrowScale(scale) {
        this.settings.fieldArrowScale = scale;
        
        if (this.fieldData) {
            this.updateFieldArrows();
        }
    }

    exportConfiguration() {
        return {
            settings: { ...this.settings }
        };
    }

    loadConfiguration(config) {
        if (config.settings) {
            Object.assign(this.settings, config.settings);
            
            // Update materials
            this.createMaterials();
            
            // Refresh visualization
            if (this.fieldData) {
                this.updateField(this.fieldData);
            }
            if (this.coilData.length > 0) {
                this.renderCoils();
            }
        }
    }

    getStats() {
        return {
            fieldLines: this.fieldLinesGroup.children.length,
            fieldArrows: this.fieldArrowsGroup.children.length,
            coils: this.coilsGroup.children.length,
            totalObjects: this.fieldGroup.children.length
        };
    }

    dispose() {
        // Clean up geometries
        if (this.arrowGeometry) this.arrowGeometry.dispose();
        if (this.coilGeometry) this.coilGeometry.dispose();
        
        // Clean up materials
        if (this.fieldLineMaterial) this.fieldLineMaterial.dispose();
        if (this.fieldArrowMaterial) this.fieldArrowMaterial.dispose();
        if (this.coilMaterial) this.coilMaterial.dispose();
        
        // Clear groups
        this.fieldLinesGroup.clear();
        this.fieldArrowsGroup.clear();
        this.coilsGroup.clear();
        this.fieldGroup.clear();
        
        // Remove from scene
        this.scene.remove(this.fieldGroup);
        
        console.log('MagneticFieldVisualizer disposed');
    }
}