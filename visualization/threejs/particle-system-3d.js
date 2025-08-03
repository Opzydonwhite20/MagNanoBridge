/**
 * 3D Particle System for MagNanoBridge visualization
 * Handles particle rendering, trails, and force visualization
 */

class ParticleSystem3D {
    constructor(scene) {
        this.scene = scene;
        
        // Particle data
        this.particles = [];
        this.particleCount = 0;
        this.maxParticles = 2000;
        
        // Visual components
        this.particleGroup = null;
        this.trailsGroup = null;
        this.forcesGroup = null;
        this.shellsGroup = null;
        
        // Geometries and materials
        this.particleGeometry = null;
        this.particleMaterial = null;
        this.shellMaterial = null;
        this.trailMaterial = null;
        this.forceMaterial = null;
        
        // Particle trails
        this.trailHistory = new Map(); // particle_id -> position history
        this.maxTrailLength = 50;
        this.trailsEnabled = true;
        
        // Force visualization
        this.forceVectors = new Map(); // particle_id -> force vector
        this.forcesEnabled = true;
        this.forceScale = 1e12; // Scale factor for force visualization
        
        // Performance optimization
        this.instancedMesh = null;
        this.useInstancing = true;
        this.lastUpdateTime = 0;
        
        // Visual settings
        this.settings = {
            particleColor: 0xff6b35,
            shellColor: 0xffd23f,
            trailColor: 0x3dd8ff,
            forceColor: 0xff4444,
            particleOpacity: 0.9,
            shellOpacity: 0.3,
            trailOpacity: 0.6,
            particleSize: 1.0, // Multiplier for actual particle size
            showShells: true,
            animateParticles: true
        };
    }

    async initialize() {
        console.log('Initializing ParticleSystem3D...');
        
        this.createGroups();
        this.createGeometries();
        this.createMaterials();
        
        console.log('ParticleSystem3D initialized');
    }

    createGroups() {
        // Main particle group
        this.particleGroup = new THREE.Group();
        this.particleGroup.name = 'particles';
        this.scene.add(this.particleGroup);
        
        // Particle shells group
        this.shellsGroup = new THREE.Group();
        this.shellsGroup.name = 'particle_shells';
        this.scene.add(this.shellsGroup);
        
        // Trails group
        this.trailsGroup = new THREE.Group();
        this.trailsGroup.name = 'particle_trails';
        this.scene.add(this.trailsGroup);
        
        // Force vectors group
        this.forcesGroup = new THREE.Group();
        this.forcesGroup.name = 'force_vectors';
        this.scene.add(this.forcesGroup);
    }

    createGeometries() {
        // Base particle geometry (sphere)
        this.particleGeometry = new THREE.SphereGeometry(1, 16, 12); // Will be scaled per particle
        
        // Shell geometry (larger sphere)
        this.shellGeometry = new THREE.SphereGeometry(1, 12, 8);
        
        // Force vector geometry (arrow)
        this.createArrowGeometry();
    }

    createArrowGeometry() {
        // Create arrow geometry for force vectors
        const arrowGeometry = new THREE.ConeGeometry(0.0001, 0.0005, 8);
        const shaftGeometry = new THREE.CylinderGeometry(0.00005, 0.00005, 0.002, 6);
        
        // Combine into single geometry
        const arrowMesh = new THREE.Mesh(arrowGeometry);
        const shaftMesh = new THREE.Mesh(shaftGeometry);
        
        arrowMesh.position.y = 0.001;
        shaftMesh.position.y = 0;
        
        arrowMesh.updateMatrix();
        shaftMesh.updateMatrix();
        
        this.arrowGeometry = new THREE.BufferGeometry();
        this.arrowGeometry = THREE.BufferGeometryUtils.mergeBufferGeometries([
            arrowGeometry,
            shaftGeometry
        ]);
    }

    createMaterials() {
        // Particle core material
        this.particleMaterial = new THREE.MeshLambertMaterial({
            color: this.settings.particleColor,
            transparent: true,
            opacity: this.settings.particleOpacity,
            emissive: new THREE.Color(this.settings.particleColor).multiplyScalar(0.1)
        });
        
        // Particle shell material
        this.shellMaterial = new THREE.MeshBasicMaterial({
            color: this.settings.shellColor,
            transparent: true,
            opacity: this.settings.shellOpacity,
            side: THREE.DoubleSide,
            wireframe: false
        });
        
        // Trail material
        this.trailMaterial = new THREE.LineBasicMaterial({
            color: this.settings.trailColor,
            transparent: true,
            opacity: this.settings.trailOpacity,
            linewidth: 2
        });
        
        // Force vector material
        this.forceMaterial = new THREE.MeshBasicMaterial({
            color: this.settings.forceColor,
            transparent: true,
            opacity: 0.8
        });
    }

    updateParticles(particleData) {
        if (!particleData || !particleData.positions) return;
        
        const positions = particleData.positions;
        const velocities = particleData.velocities || [];
        const radii = particleData.radii || [];
        const shellThickness = particleData.shell_thickness || [];
        const ids = particleData.ids || [];
        
        this.particleCount = positions.length;
        
        // Clear existing particles
        this.clearParticles();
        
        // Create or update particles
        for (let i = 0; i < this.particleCount; i++) {
            const position = positions[i];
            const velocity = velocities[i] || [0, 0, 0];
            const radius = radii[i] || 50e-9;
            const shell = shellThickness[i] || 10e-9;
            const id = ids[i] || `particle_${i}`;
            
            this.createParticle(i, position, radius, shell, id);
            this.updateTrail(id, position);
            
            // Update force vector if available
            if (particleData.forces && particleData.forces[i]) {
                this.updateForceVector(id, position, particleData.forces[i]);
            }
        }
        
        // Update trails
        if (this.trailsEnabled) {
            this.updateTrails();
        }
        
        // Update force vectors
        if (this.forcesEnabled) {
            this.updateForceVectors();
        }
    }

    createParticle(index, position, radius, shellThickness, id) {
        // Create particle core
        const particleMesh = new THREE.Mesh(
            this.particleGeometry.clone(),
            this.particleMaterial.clone()
        );
        
        particleMesh.position.set(position[0], position[1], position[2]);
        particleMesh.scale.setScalar(radius * this.settings.particleSize);
        particleMesh.name = `particle_core_${id}`;
        particleMesh.userData = { id, index, radius, shellThickness };
        
        this.particleGroup.add(particleMesh);
        
        // Create particle shell if enabled
        if (this.settings.showShells && shellThickness > 0) {
            const totalRadius = radius + shellThickness;
            const shellMesh = new THREE.Mesh(
                this.shellGeometry.clone(),
                this.shellMaterial.clone()
            );
            
            shellMesh.position.set(position[0], position[1], position[2]);
            shellMesh.scale.setScalar(totalRadius * this.settings.particleSize);
            shellMesh.name = `particle_shell_${id}`;
            shellMesh.userData = { id, index, totalRadius };
            
            this.shellsGroup.add(shellMesh);
        }
        
        // Add to particles array
        this.particles[index] = {
            id,
            coreMesh: particleMesh,
            shellMesh: this.shellsGroup.getObjectByName(`particle_shell_${id}`),
            position: [...position],
            radius,
            shellThickness
        };
    }

    updateTrail(particleId, position) {
        if (!this.trailHistory.has(particleId)) {
            this.trailHistory.set(particleId, []);
        }
        
        const trail = this.trailHistory.get(particleId);
        trail.push([...position]);
        
        // Limit trail length
        if (trail.length > this.maxTrailLength) {
            trail.shift();
        }
    }

    updateTrails() {
        // Clear existing trails
        this.trailsGroup.clear();
        
        if (!this.trailsEnabled) return;
        
        this.trailHistory.forEach((trail, particleId) => {
            if (trail.length < 2) return;
            
            // Create trail geometry
            const points = trail.map(pos => new THREE.Vector3(pos[0], pos[1], pos[2]));
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            
            // Create trail line
            const line = new THREE.Line(geometry, this.trailMaterial.clone());
            line.name = `trail_${particleId}`;
            
            // Fade trail opacity based on age
            const material = line.material;
            material.opacity = this.settings.trailOpacity * 0.8;
            
            this.trailsGroup.add(line);
        });
    }

    updateForceVector(particleId, position, force) {
        this.forceVectors.set(particleId, {
            position: [...position],
            force: [...force]
        });
    }

    updateForceVectors() {
        // Clear existing force vectors
        this.forcesGroup.clear();
        
        if (!this.forcesEnabled) return;
        
        this.forceVectors.forEach((data, particleId) => {
            const { position, force } = data;
            const forceMagnitude = Math.sqrt(force[0]**2 + force[1]**2 + force[2]**2);
            
            if (forceMagnitude < 1e-20) return; // Skip negligible forces
            
            // Create force vector arrow
            const arrowMesh = new THREE.Mesh(
                this.arrowGeometry.clone(),
                this.forceMaterial.clone()
            );
            
            // Position at particle
            arrowMesh.position.set(position[0], position[1], position[2]);
            
            // Scale based on force magnitude
            const scale = Math.min(forceMagnitude * this.forceScale, 0.005); // Max 5mm length
            arrowMesh.scale.setScalar(scale);
            
            // Orient toward force direction
            const forceDirection = new THREE.Vector3(force[0], force[1], force[2]).normalize();
            arrowMesh.lookAt(
                arrowMesh.position.x + forceDirection.x,
                arrowMesh.position.y + forceDirection.y,
                arrowMesh.position.z + forceDirection.z
            );
            
            // Color based on force magnitude (blue -> red)
            const normalizedMag = Math.min(forceMagnitude * 1e15, 1.0);
            const color = new THREE.Color();
            color.setHSL((1 - normalizedMag) * 0.67, 1.0, 0.5); // Blue to red
            arrowMesh.material.color = color;
            
            arrowMesh.name = `force_${particleId}`;
            this.forcesGroup.add(arrowMesh);
        });
    }

    clearParticles() {
        this.particleGroup.clear();
        this.shellsGroup.clear();
        this.particles = [];
    }

    update(currentTime) {
        if (!this.settings.animateParticles) return;
        
        const deltaTime = currentTime - this.lastUpdateTime;
        this.lastUpdateTime = currentTime;
        
        // Animate particles (subtle breathing/pulsing effect)
        this.particles.forEach((particle, index) => {
            if (!particle || !particle.coreMesh) return;
            
            const time = currentTime * 0.001; // Convert to seconds
            const offset = index * 0.1; // Phase offset per particle
            
            // Subtle pulsing animation
            const pulseScale = 1.0 + 0.05 * Math.sin(time * 2 + offset);
            const baseScale = particle.radius * this.settings.particleSize;
            particle.coreMesh.scale.setScalar(baseScale * pulseScale);
            
            // Shell animation (if exists)
            if (particle.shellMesh) {
                const shellScale = (particle.radius + particle.shellThickness) * this.settings.particleSize;
                particle.shellMesh.scale.setScalar(shellScale * pulseScale * 0.98);
            }
        });
        
        // Update trail opacity animation
        this.trailsGroup.children.forEach((trail, index) => {
            if (trail.material) {
                const time = currentTime * 0.001;
                const opacity = this.settings.trailOpacity * (0.8 + 0.2 * Math.sin(time + index * 0.3));
                trail.material.opacity = opacity;
            }
        });
    }

    setTrailsVisible(visible) {
        this.trailsEnabled = visible;
        this.trailsGroup.visible = visible;
        
        if (!visible) {
            this.trailsGroup.clear();
        }
    }

    setForcesVisible(visible) {
        this.forcesEnabled = visible;
        this.forcesGroup.visible = visible;
        
        if (!visible) {
            this.forcesGroup.clear();
        }
    }

    setShellsVisible(visible) {
        this.settings.showShells = visible;
        this.shellsGroup.visible = visible;
    }

    setParticleSize(multiplier) {
        this.settings.particleSize = multiplier;
        
        // Update existing particles
        this.particles.forEach(particle => {
            if (particle && particle.coreMesh) {
                const baseScale = particle.radius * multiplier;
                particle.coreMesh.scale.setScalar(baseScale);
                
                if (particle.shellMesh) {
                    const shellScale = (particle.radius + particle.shellThickness) * multiplier;
                    particle.shellMesh.scale.setScalar(shellScale);
                }
            }
        });
    }

    setTrailLength(length) {
        this.maxTrailLength = Math.max(1, Math.min(length, 200));
        
        // Trim existing trails
        this.trailHistory.forEach(trail => {
            while (trail.length > this.maxTrailLength) {
                trail.shift();
            }
        });
    }

    setForceScale(scale) {
        this.forceScale = scale;
    }

    highlightParticle(particleId, highlight = true) {
        const particle = this.particles.find(p => p && p.id === particleId);
        if (!particle) return;
        
        if (highlight) {
            // Change color to highlight
            particle.coreMesh.material.color.setHex(0x00ff00);
            particle.coreMesh.material.emissive.setHex(0x004400);
            
            if (particle.shellMesh) {
                particle.shellMesh.material.color.setHex(0x88ff88);
            }
        } else {
            // Restore original colors
            particle.coreMesh.material.color.setHex(this.settings.particleColor);
            particle.coreMesh.material.emissive.setHex(
                new THREE.Color(this.settings.particleColor).multiplyScalar(0.1).getHex()
            );
            
            if (particle.shellMesh) {
                particle.shellMesh.material.color.setHex(this.settings.shellColor);
            }
        }
    }

    getParticleById(particleId) {
        return this.particles.find(p => p && p.id === particleId);
    }

    getParticleAt(position, tolerance = 1e-4) {
        const searchPos = new THREE.Vector3(position[0], position[1], position[2]);
        
        for (const particle of this.particles) {
            if (!particle || !particle.coreMesh) continue;
            
            const distance = particle.coreMesh.position.distanceTo(searchPos);
            if (distance <= tolerance) {
                return particle;
            }
        }
        
        return null;
    }

    reset() {
        this.clearParticles();
        this.trailsGroup.clear();
        this.forcesGroup.clear();
        this.trailHistory.clear();
        this.forceVectors.clear();
        
        console.log('ParticleSystem3D reset');
    }

    getStats() {
        return {
            particleCount: this.particleCount,
            trailCount: this.trailHistory.size,
            forceVectorCount: this.forceVectors.size,
            totalObjects: this.particleGroup.children.length + 
                         this.shellsGroup.children.length + 
                         this.trailsGroup.children.length + 
                         this.forcesGroup.children.length
        };
    }

    exportConfiguration() {
        return {
            settings: { ...this.settings },
            maxTrailLength: this.maxTrailLength,
            forceScale: this.forceScale,
            trailsEnabled: this.trailsEnabled,
            forcesEnabled: this.forcesEnabled,
            particleCount: this.particleCount
        };
    }

    loadConfiguration(config) {
        if (config.settings) {
            Object.assign(this.settings, config.settings);
        }
        
        if (config.maxTrailLength !== undefined) {
            this.setTrailLength(config.maxTrailLength);
        }
        
        if (config.forceScale !== undefined) {
            this.setForceScale(config.forceScale);
        }
        
        if (config.trailsEnabled !== undefined) {
            this.setTrailsVisible(config.trailsEnabled);
        }
        
        if (config.forcesEnabled !== undefined) {
            this.setForcesVisible(config.forcesEnabled);
        }
        
        // Recreate materials with new settings
        this.createMaterials();
    }

    dispose() {
        // Clean up geometries
        if (this.particleGeometry) this.particleGeometry.dispose();
        if (this.shellGeometry) this.shellGeometry.dispose();
        if (this.arrowGeometry) this.arrowGeometry.dispose();
        
        // Clean up materials
        if (this.particleMaterial) this.particleMaterial.dispose();
        if (this.shellMaterial) this.shellMaterial.dispose();
        if (this.trailMaterial) this.trailMaterial.dispose();
        if (this.forceMaterial) this.forceMaterial.dispose();
        
        // Clear groups
        this.particleGroup.clear();
        this.shellsGroup.clear();
        this.trailsGroup.clear();
        this.forcesGroup.clear();
        
        // Remove from scene
        this.scene.remove(this.particleGroup);
        this.scene.remove(this.shellsGroup);
        this.scene.remove(this.trailsGroup);
        this.scene.remove(this.forcesGroup);
        
        // Clear data structures
        this.particles = [];
        this.trailHistory.clear();
        this.forceVectors.clear();
        
        console.log('ParticleSystem3D disposed');
    }
}