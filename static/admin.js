// FractiVerse Admin UI

let ws;
let charts = {};

// Initialize WebSocket connection
function initWebSocket() {
    console.log("Initializing WebSocket connection...");
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log("WebSocket connected!");
        document.getElementById('connection-status').innerHTML = '✅ Connected';
        requestUpdate();
    };
    
    ws.onclose = () => {
        console.log("WebSocket disconnected!");
        document.getElementById('connection-status').innerHTML = '❌ Disconnected';
        setTimeout(initWebSocket, 1000);
    };
    
    ws.onmessage = (event) => {
        console.log("Received message:", event.data);
        const message = JSON.parse(event.data);
        handleMessage(message);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };
}

// Handle incoming messages
function handleMessage(message) {
    console.log("Processing message:", message.type);
    switch(message.type) {
        case 'initial_state':
            updateComponentStatus(message.data.component_status);
            updateMetrics(message.data);
            updatePatternLog(message.data.patterns);
            break;
            
        case 'state_update':
            updateMetrics(message.data);
            break;
            
        case 'pattern_update':
            updatePatternLog(message.data.patterns);
            updateCharts(message.data);
            break;
            
        default:
            console.log("Unknown message type:", message.type);
    }
}

// Update component status display
function updateComponentStatus(status) {
    const container = document.getElementById('component-status');
    container.innerHTML = '';
    
    for (const [component, active] of Object.entries(status)) {
        container.innerHTML += `
            <div class="status-item ${active ? 'active' : 'inactive'}">
                ${component}: ${active ? '✓' : '✗'}
            </div>
        `;
    }
}

// Update metric displays
function updateMetrics(data) {
    // Update cognitive metrics
    if (data.cognitive) {
        document.getElementById('cognitive-metrics').innerHTML = `
            <div>PEFF Coherence: ${data.cognitive.peff_coherence.toFixed(3)}</div>
            <div>Memory Usage: ${(data.cognitive.memory_usage * 100).toFixed(1)}%</div>
            <div>Learning Rate: ${data.cognitive.learning_rate.toFixed(3)}</div>
            <div>Pattern Count: ${data.cognitive.pattern_count}</div>
        `;
    }
    
    // Update network metrics
    if (data.network) {
        document.getElementById('network-metrics').innerHTML = `
            <div>Peers: ${data.network.peer_count}</div>
            <div>Sync: ${data.network.sync_status}</div>
            <div>Bandwidth: ${formatBytes(data.network.bandwidth_usage)}/s</div>
        `;
    }
    
    // Update blockchain metrics
    if (data.blockchain) {
        document.getElementById('blockchain-metrics').innerHTML = `
            <div>Height: ${data.blockchain.block_height}</div>
            <div>Patterns: ${data.blockchain.pattern_count}</div>
            <div>Consensus: ${data.blockchain.consensus_status}</div>
        `;
    }
    
    // Update treasury metrics
    if (data.treasury) {
        document.getElementById('treasury-metrics').innerHTML = `
            <div>Supply: ${formatNumber(data.treasury.total_supply)} FRACT</div>
            <div>Reward Pool: ${formatNumber(data.treasury.reward_pool)} FRACT</div>
            <div>Distribution: ${data.treasury.distribution_rate.toFixed(2)}/block</div>
        `;
    }
}

// Handle cognitive events
function handleCognitiveEvent(event) {
    const log = document.getElementById('event-log');
    log.innerHTML = `
        <div class="event">
            <div class="event-time">${new Date(event.timestamp * 1000).toLocaleTimeString()}</div>
            <div class="event-type">${event.event_type}</div>
            <div class="event-pattern">Pattern: ${event.pattern_id}</div>
            <div class="event-coherence">Coherence: ${event.metrics.peff_coherence.toFixed(3)}</div>
        </div>
    ` + log.innerHTML;
    
    // Trim log
    if (log.children.length > 100) {
        log.removeChild(log.lastChild);
    }
}

// Initialize charts
function initCharts() {
    charts.coherence = new Chart('coherence-chart', {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { responsive: true }
    });
    
    // Initialize other charts...
}

// Update charts with new data
function updateCharts(data) {
    // Update coherence chart
    if (data.cognitive) {
        charts.coherence.data.labels.push(new Date().toLocaleTimeString());
        charts.coherence.data.datasets[0].data.push(data.cognitive.peff_coherence);
        
        if (charts.coherence.data.labels.length > 50) {
            charts.coherence.data.labels.shift();
            charts.coherence.data.datasets[0].data.shift();
        }
        
        charts.coherence.update();
    }
    
    // Update other charts...
}

// Utility functions
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
}

function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Initialize UI
window.onload = () => {
    console.log("Page loaded, initializing...");
    initWebSocket();
    initCharts();
    
    // Request updates periodically
    setInterval(requestUpdate, 5000);
};

// Request state update
function requestUpdate() {
    if (ws.readyState === WebSocket.OPEN) {
        console.log("Requesting state update...");
        ws.send(JSON.stringify({ type: 'request_update' }));
    }
}

// Add pattern handling functions
function updatePatternLog(patterns) {
    const log = document.getElementById('pattern-log');
    patterns.forEach(pattern => {
        const entry = document.createElement('div');
        entry.className = 'pattern-entry';
        entry.innerHTML = `
            <div class="pattern-time">${new Date().toLocaleTimeString()}</div>
            <div class="pattern-type">${pattern.type}</div>
            <div class="pattern-content">${pattern.content}</div>
            <div class="pattern-metrics">
                <span>Coherence: ${pattern.coherence.toFixed(3)}</span>
                <span>Connections: ${pattern.connections.length}</span>
            </div>
        `;
        log.insertBefore(entry, log.firstChild);
    });
    
    // Trim log
    while (log.children.length > 100) {
        log.removeChild(log.lastChild);
    }
} 