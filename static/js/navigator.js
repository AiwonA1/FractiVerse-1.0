// 3D Unipixel Navigator
class UnipixelNavigator {
    constructor() {
        console.log("Initializing UnipixelNavigator");
        
        // Initialize Three.js components
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = null;
        this.controls = null;
        this.patterns = new Map();
        this.gridHelper = null;
        
        // Pattern visualization settings
        this.pointSize = 2;
        this.maxPoints = 65536; // 256 x 256
        this.colorScale = new THREE.Color();
        
        // Add pattern material
        this.patternMaterial = new THREE.PointsMaterial({
            size: this.pointSize,
            vertexColors: true,
            sizeAttenuation: false
        });
        
        // WebSocket configuration
        this.wsConfig = {
            reconnectAttempts: 0,
            maxAttempts: 5,
            baseDelay: 1000,
            maxDelay: 30000,
            url: `ws://${window.location.host}/ws`
        };
        
        // Initialize command interface first
        if (!window.commandInterface) {
            console.log("Creating new CommandInterface");
            window.commandInterface = new CommandInterface();
        }
        
        // Initialize viewer and WebSocket
        this.initViewer();
        this.setupWebSocket();
        
        console.log("Navigator initialization complete");
    }

    setupWebSocket() {
        if (this.wsConfig.reconnectAttempts >= this.wsConfig.maxAttempts) {
            console.error("Max WebSocket reconnection attempts reached");
            document.getElementById('connection-status').innerHTML = '❌ Connection failed';
            return;
        }

        try {
            console.log("Initializing WebSocket connection...");
            this.ws = new WebSocket(this.wsConfig.url);

            this.ws.onopen = () => {
                console.log("WebSocket Connected");
                document.getElementById('connection-status').innerHTML = '✅ Connected';
                this.wsConfig.reconnectAttempts = 0;
                // Request initial pattern data
                this.ws.send(JSON.stringify({type: 'request_patterns'}));
            };

            this.ws.onclose = () => {
                console.log("WebSocket disconnected");
                this.handleDisconnect();
            };

            this.ws.onerror = (error) => {
                console.error("WebSocket error:", error);
                // Let onclose handle reconnection
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handlePatternUpdate(data);
                } catch (e) {
                    console.error("Error handling message:", e);
                }
            };

        } catch (error) {
            console.error("WebSocket setup error:", error);
            this.handleDisconnect();
        }
    }

    handleDisconnect() {
        const delay = Math.min(
            this.wsConfig.baseDelay * Math.pow(2, this.wsConfig.reconnectAttempts),
            this.wsConfig.maxDelay
        );

        this.wsConfig.reconnectAttempts++;
        
        if (this.wsConfig.reconnectAttempts < this.wsConfig.maxAttempts) {
            console.log(`Scheduling reconnection attempt ${this.wsConfig.reconnectAttempts} in ${delay}ms`);
            setTimeout(() => this.setupWebSocket(), delay);
        } else {
            console.error("Max reconnection attempts reached");
            document.getElementById('connection-status').innerHTML = 
                '❌ Connection failed - Please refresh page';
        }
    }

    initViewer() {
        console.log("Initializing viewer");
        // Get container
        const container = document.getElementById('hologram-viewer');
        if (!container) {
            console.error("Could not find hologram-viewer container");
            return;
        }

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            canvas: container.querySelector('canvas') || undefined
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setClearColor(0x000000, 0);
        container.appendChild(this.renderer.domElement);

        // Setup camera
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Add grid
        this.gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
        this.scene.add(this.gridHelper);

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);

        // Remove loading indicator
        const loading = container.querySelector('.loading');
        if (loading) loading.remove();

        // Handle resize
        window.addEventListener('resize', () => {
            const width = container.clientWidth;
            const height = container.clientHeight;
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        });

        console.log("Starting animation loop");
        this.animate();
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    updatePattern(id, data) {
        try {
            console.log("Updating pattern:", id);
            const {magnitude, phase, holographic_coords} = data;
            
            // Debug data
            console.log("Pattern data:", {
                magnitude: magnitude.length,
                phase: phase.length,
                coords: holographic_coords
            });

            // Flatten arrays if they're 2D
            const flatMagnitude = magnitude.flat();
            const flatPhase = phase.flat();
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(flatMagnitude.length * 3);
            const colors = new Float32Array(flatMagnitude.length * 3);
            
            // Fill buffers
            let pointCount = 0;
            for (let i = 0; i < flatMagnitude.length; i++) {
                // Lower threshold to see more points
                if (flatMagnitude[i] > 0.001) {
                    // Get 2D coordinates
                    const x = i % 256;
                    const y = Math.floor(i / 256);
                    
                    // Scale and position in 3D space
                    const px = (x / 256 - 0.5) * 10 + holographic_coords.x;
                    const py = (y / 256 - 0.5) * 10 + holographic_coords.y;
                    const pz = (flatMagnitude[i] - 0.5) * 10 + holographic_coords.z;
                    
                    positions[pointCount * 3] = px;
                    positions[pointCount * 3 + 1] = py;
                    positions[pointCount * 3 + 2] = pz;
                    
                    // Color based on phase
                    const hue = (flatPhase[i] + Math.PI) / (2 * Math.PI);
                    this.colorScale.setHSL(hue, 1, 0.5);
                    colors[pointCount * 3] = this.colorScale.r;
                    colors[pointCount * 3 + 1] = this.colorScale.g;
                    colors[pointCount * 3 + 2] = this.colorScale.b;
                    
                    pointCount++;
                }
            }
            
            console.log(`Processing ${pointCount} points out of ${flatMagnitude.length}`);
            
            // Only create points if we have any
            if (pointCount > 0) {
                // Update geometry
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions.slice(0, pointCount * 3), 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors.slice(0, pointCount * 3), 3));
                
                // Create or update points
                if (this.patterns.has(id)) {
                    const points = this.patterns.get(id);
                    points.geometry.dispose();
                    points.geometry = geometry;
                } else {
                    const points = new THREE.Points(geometry, this.patternMaterial);
                    points.name = `pattern-${id}`;
                    this.scene.add(points);
                    this.patterns.set(id, points);
                }
                
                console.log(`Created ${pointCount} points for pattern visualization`);
                
                // Update camera target
                this.controls.target.set(
                    holographic_coords.x,
                    holographic_coords.y,
                    holographic_coords.z
                );
                this.controls.update();
            }
            
        } catch (error) {
            console.error("Error updating pattern:", error, error.stack);
        }
    }

    resetView() {
        console.log("Resetting view");
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }

    toggleGrid() {
        console.log("Toggling grid");
        if (this.gridHelper) {
            this.gridHelper.visible = !this.gridHelper.visible;
        }
    }

    handlePatternUpdate(data) {
        try {
            console.log("Received data:", data);
            
            if (data.type === 'update') {
                // Handle pattern update
                if (data.pattern) {
                    console.log("Processing pattern:", data.pattern);
                    this.updatePattern(data.pattern.id, data.pattern.data);
                }
                
                // Handle metrics update
                if (data.metrics) {
                    console.log("Processing metrics:", data.metrics);
                    if (window.commandInterface) {
                        window.commandInterface.updateMetrics(data.metrics);
                    } else {
                        console.warn("CommandInterface not initialized");
                    }
                }
            } else if (data.type === 'pattern_update') {
                // Handle legacy pattern updates
                this.updatePattern(data.id, data.data);
            }
        } catch (error) {
            console.error("Error handling pattern update:", error, error.stack);
        }
    }
}

// Initialize navigator when document loads
document.addEventListener('DOMContentLoaded', () => {
    console.log("Document loaded, creating navigator");
    window.navigator = new UnipixelNavigator();
});
