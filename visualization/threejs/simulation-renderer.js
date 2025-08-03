/**
 * Main Three.js renderer for MagNanoBridge simulation
 * Handles scene setup, camera controls, and rendering pipeline
 */

class SimulationRenderer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Three.js core objects
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Lighting
        this.ambientLight = null;
        this.directionalLights = [];
        
        // Focus zone visualization
        this.focusZoneGroup = null;
        this.focusZoneMesh = null;
        
        // Performance monitoring
        this.renderStats = {
            lastFrameTime: 0,
            frameCount: 0,
            avgFrameTime: 0
        };
        
        // Settings
        this.settings = {
            antialias: true,
            shadows: true,
            autoRotate: false,
            focusZoneVisible: true,
            backgroundColor: 0x1a1a1a
        };
    }

    async initialize() {
        console.log('Initializing SimulationRenderer...');
        
        // Check WebGL support
        if (!this.checkWebGLSupport()) {
            throw new Error('WebGL not supported');
        }
        
        // Initialize Three.js components
        this.initializeScene();
        this.initializeCamera();
        this.initializeRenderer();
        this.initializeLighting();
        this.initializeControls();
        this.initializeFocusZone();
        
        // Add scene helpers
        this.addSceneHelpers();
        
        console.log('SimulationRenderer initialized successfully');
    }

    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && 
                     (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }

    initializeScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.settings.backgroundColor);
        this.scene.fog = new THREE.Fog(this.settings.backgroundColor, 0.01, 0.1);
    }

    initializeCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 1e-6, 1);
        
        // Position camera for good initial view
        this.camera.position.set(0.02, 0.02, 0.02); // 20mm from origin
        this.camera.lookAt(0, 0, 0);
    }

    initializeRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: this.settings.antialias,
            alpha: true,
            powerPreference: 'high-performance'
        });
        
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        // Enable shadows
        if (this.settings.shadows) {
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        }
        
        // Set color space
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        this.container.appendChild(this.renderer.domElement);
    }

    initializeLighting() {
        // Ambient light for overall illumination
        this.ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(this.ambientLight);
        
        // Key light
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
        keyLight.position.set(0.05, 0.05, 0.05);
        keyLight.castShadow = this.settings.shadows;
        
        if (this.settings.shadows) {
            keyLight.shadow.mapSize.width = 2048;
            keyLight.shadow.mapSize.height = 2048;
            keyLight.shadow.camera.near = 0.001;
            keyLight.shadow.camera.far = 0.1;
            keyLight.shadow.camera.left = -0.02;
            keyLight.shadow.camera.right = 0.02;
            keyLight.shadow.camera.top = 0.02;
            keyLight.shadow.camera.bottom = -0.02;
        }
        
        this.scene.add(keyLight);
        this.directionalLights.push(keyLight);
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
        fillLight.position.set(-0.03, 0.02, -0.03);
        this.scene.add(fillLight);
        this.directionalLights.push(fillLight);
        
        // Rim light
        const rimLight = new THREE.DirectionalLight(0xff6b35, 0.2);
        rimLight.position.set(0, -0.03, 0.03);
        this.scene.add(rimLight);
        this.directionalLights.push(rimLight);
    }

    initializeControls() {
        // Note: Using basic mouse controls since OrbitControls might not be available
        // In a full implementation, you'd import OrbitControls from Three.js examples
        
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let targetX = 0;
        let targetY = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            isMouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!isMouseDown) return;
            
            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;
            
            targetX += deltaX * 0.01;
            targetY += deltaY * 0.01;
            
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });
        
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const scale = event.deltaY > 0 ? 1.1 : 0.9;
            this.camera.position.multiplyScalar(scale);
            
            // Limit zoom
            const distance = this.camera.position.length();
            if (distance < 0.005) this.camera.position.normalize().multiplyScalar(0.005);
            if (distance > 0.2) this.camera.position.normalize().multiplyScalar(0.2);
        });
        
        // Auto-rotation update
        this.updateCameraRotation = () => {
            if (this.settings.autoRotate) {
                const time = Date.now() * 0.0005;
                const radius = this.camera.position.length();
                this.camera.position.x = Math.cos(time) * radius;
                this.camera.position.z = Math.sin(time) * radius;
                this.camera.lookAt(0, 0, 0);
            } else {
                // Manual rotation
                const radius = this.camera.position.length();
                this.camera.position.x = Math.cos(targetY) * Math.cos(targetX) * radius;
                this.camera.position.y = Math.sin(targetY) * radius;
                this.camera.position.z = Math.cos(targetY) * Math.sin(targetX) * radius;
                this.camera.lookAt(0, 0, 0);
            }
        };
    }

    initializeFocusZone() {
        this.focusZoneGroup = new THREE.Group();
        this.scene.add(this.focusZoneGroup);
        
        // Create default cylindrical focus zone
        this.createCylindricalFocusZone(0.001, 0.002); // 1mm radius, 2mm height
    }

    createCylindricalFocusZone(radius, height) {
        // Clear existing focus zone
        this.focusZoneGroup.clear();
        
        // Create cylinder geometry
        const geometry = new THREE.CylinderGeometry(radius, radius, height, 32, 1, true);
        
        // Create material with transparency
        const material = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide,
            wireframe: false
        });
        
        this.focusZoneMesh = new THREE.Mesh(geometry, material);
        this.focusZoneGroup.add(this.focusZoneMesh);
        
        // Add wireframe overlay
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            wireframe: true,
            transparent: true,
            opacity: 0.8
        });
        
        const wireframeMesh = new THREE.Mesh(geometry.clone(), wireframeMaterial);
        this.focusZoneGroup.add(wireframeMesh);
        
        // Add center indicator
        const centerGeometry = new THREE.SphereGeometry(radius * 0.05, 8, 8);
        const centerMaterial = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            transparent: true,
            opacity: 0.6
        });
        
        const centerMesh = new THREE.Mesh(centerGeometry, centerMaterial);
        this.focusZoneGroup.add(centerMesh);
    }

    addSceneHelpers() {
        // Coordinate system axes
        const axesHelper = new THREE.AxesHelper(0.005); // 5mm axes
        axesHelper.material.transparent = true;
        axesHelper.material.opacity = 0.6;
        this.scene.add(axesHelper);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(0.02, 20, 0x444444, 0x222222); // 20mm grid
        gridHelper.material.transparent = true;
        gridHelper.material.opacity = 0.3;
        this.scene.add(gridHelper);
        
        // Scale reference (1mm cube)
        const scaleGeometry = new THREE.BoxGeometry(0.001, 0.001, 0.001);
        const scaleMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.5,
            wireframe: true
        });
        
        const scaleCube = new THREE.Mesh(scaleGeometry, scaleMaterial);
        scaleCube.position.set(0.008, -0.008, 0.008); // Corner position
        this.scene.add(scaleCube);
        
        // Add scale label (would need text rendering in full implementation)
        console.log('Scene helpers added (axes, grid, scale reference)');
    }

    update(currentTime) {
        // Update camera rotation
        this.updateCameraRotation();
        
        // Update performance stats
        const frameTime = currentTime - this.renderStats.lastFrameTime;
        this.renderStats.frameCount++;
        
        if (this.renderStats.frameCount > 60) {
            this.renderStats.avgFrameTime = frameTime;
            this.renderStats.frameCount = 0;
        }
        
        this.renderStats.lastFrameTime = currentTime;
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }

    setCameraView(viewType) {
        const distance = 0.025; // 25mm from origin
        
        switch (viewType) {
            case 'top':
                this.camera.position.set(0, distance, 0);
                this.camera.lookAt(0, 0, 0);
                break;
                
            case 'side':
                this.camera.position.set(distance, 0, 0);
                this.camera.lookAt(0, 0, 0);
                break;
                
            case 'front':
                this.camera.position.set(0, 0, distance);
                this.camera.lookAt(0, 0, 0);
                break;
                
            case '3d':
            default:
                this.camera.position.set(distance * 0.7, distance * 0.7, distance * 0.7);
                this.camera.lookAt(0, 0, 0);
                break;
        }
    }

    setAutoRotate(enabled) {
        this.settings.autoRotate = enabled;
    }

    setFocusZoneVisible(visible) {
        this.settings.focusZoneVisible = visible;
        if (this.focusZoneGroup) {
            this.focusZoneGroup.visible = visible;
        }
    }

    updateFocusZone(focusZoneData) {
        if (!focusZoneData || !this.focusZoneGroup) return;
        
        // Update position
        if (focusZoneData.center) {
            this.focusZoneGroup.position.set(
                focusZoneData.center[0],
                focusZoneData.center[1],
                focusZoneData.center[2]
            );
        }
        
        // Update geometry based on shape
        if (focusZoneData.shape === 'cylinder' && focusZoneData.parameters) {
            const radius = focusZoneData.parameters.radius;
            const height = focusZoneData.parameters.height;
            
            if (radius !== this.currentFocusRadius || height !== this.currentFocusHeight) {
                this.createCylindricalFocusZone(radius, height);
                this.currentFocusRadius = radius;
                this.currentFocusHeight = height;
            }
        }
        
        // Update color based on efficiency
        if (focusZoneData.efficiency_history && this.focusZoneMesh) {
            const latestEfficiency = focusZoneData.efficiency_history[focusZoneData.efficiency_history.length - 1] || 0;
            const targetEfficiency = focusZoneData.target_efficiency || 0.8;
            
            // Color transitions from red (low efficiency) to green (high efficiency)
            const normalizedEfficiency = Math.min(latestEfficiency / targetEfficiency, 1.0);
            const color = new THREE.Color();
            
            if (normalizedEfficiency < 0.5) {
                // Red to yellow
                color.setHSL(normalizedEfficiency * 0.16, 1.0, 0.5); // 0 to 60 degrees (red to yellow)
            } else {
                // Yellow to green
                color.setHSL(0.16 + (normalizedEfficiency - 0.5) * 0.17, 1.0, 0.5); // 60 to 120 degrees (yellow to green)
            }
            
            this.focusZoneMesh.material.color = color;
        }
    }

    createSphericalFocusZone(radius) {
        this.focusZoneGroup.clear();
        
        const geometry = new THREE.SphereGeometry(radius, 32, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });
        
        this.focusZoneMesh = new THREE.Mesh(geometry, material);
        this.focusZoneGroup.add(this.focusZoneMesh);
        
        // Wireframe overlay
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            wireframe: true,
            transparent: true,
            opacity: 0.8
        });
        
        const wireframeMesh = new THREE.Mesh(geometry.clone(), wireframeMaterial);
        this.focusZoneGroup.add(wireframeMesh);
    }

    createBoxFocusZone(dimensions) {
        this.focusZoneGroup.clear();
        
        const geometry = new THREE.BoxGeometry(dimensions[0], dimensions[1], dimensions[2]);
        const material = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });
        
        this.focusZoneMesh = new THREE.Mesh(geometry, material);
        this.focusZoneGroup.add(this.focusZoneMesh);
        
        // Wireframe overlay
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x3dd8ff,
            wireframe: true,
            transparent: true,
            opacity: 0.8
        });
        
        const wireframeMesh = new THREE.Mesh(geometry.clone(), wireframeMaterial);
        this.focusZoneGroup.add(wireframeMesh);
    }

    addCustomObject(object) {
        this.scene.add(object);
    }

    removeCustomObject(object) {
        this.scene.remove(object);
    }

    setBackgroundColor(color) {
        this.settings.backgroundColor = color;
        this.scene.background.setHex(color);
        if (this.scene.fog) {
            this.scene.fog.color.setHex(color);
        }
    }

    enableShadows(enabled) {
        this.settings.shadows = enabled;
        this.renderer.shadowMap.enabled = enabled;
        
        this.directionalLights.forEach(light => {
            light.castShadow = enabled;
        });
    }

    setLightingIntensity(intensity) {
        this.ambientLight.intensity = intensity * 0.4;
        this.directionalLights.forEach((light, index) => {
            switch (index) {
                case 0: // Key light
                    light.intensity = intensity * 0.8;
                    break;
                case 1: // Fill light
                    light.intensity = intensity * 0.3;
                    break;
                case 2: // Rim light
                    light.intensity = intensity * 0.2;
                    break;
            }
        });
    }

    screenshot(width = 1920, height = 1080) {
        // Store current size
        const currentWidth = this.renderer.domElement.width;
        const currentHeight = this.renderer.domElement.height;
        
        // Resize for screenshot
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        // Render
        this.renderer.render(this.scene, this.camera);
        
        // Get image data
        const imageData = this.renderer.domElement.toDataURL('image/png');
        
        // Restore original size
        this.renderer.setSize(currentWidth, currentHeight, false);
        this.camera.aspect = currentWidth / currentHeight;
        this.camera.updateProjectionMatrix();
        
        return imageData;
    }

    getPerformanceStats() {
        return {
            avgFrameTime: this.renderStats.avgFrameTime,
            memoryUsage: this.renderer.info.memory,
            renderInfo: this.renderer.info.render
        };
    }

    dispose() {
        // Clean up Three.js resources
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        // Clear scene
        if (this.scene) {
            this.scene.clear();
        }
        
        // Remove event listeners
        if (this.renderer && this.renderer.domElement) {
            this.renderer.domElement.removeEventListener('mousedown', null);
            this.renderer.domElement.removeEventListener('mousemove', null);
            this.renderer.domElement.removeEventListener('mouseup', null);
            this.renderer.domElement.removeEventListener('wheel', null);
        }
        
        console.log('SimulationRenderer disposed');
    }

    // Debug methods
    logSceneInfo() {
        console.log('Scene Info:', {
            children: this.scene.children.length,
            triangles: this.renderer.info.render.triangles,
            geometries: this.renderer.info.memory.geometries,
            textures: this.renderer.info.memory.textures
        });
    }

    toggleWireframe() {
        this.scene.traverse((child) => {
            if (child.isMesh && child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => {
                        material.wireframe = !material.wireframe;
                    });
                } else {
                    child.material.wireframe = !child.material.wireframe;
                }
            }
        });
    }

    exportScene() {
        // Export scene configuration for debugging
        const sceneData = {
            camera: {
                position: this.camera.position.toArray(),
                rotation: this.camera.rotation.toArray(),
                fov: this.camera.fov,
                aspect: this.camera.aspect
            },
            settings: this.settings,
            objects: this.scene.children.length,
            performance: this.getPerformanceStats()
        };
        
        console.log('Scene Export:', sceneData);
        return sceneData;
    }
}