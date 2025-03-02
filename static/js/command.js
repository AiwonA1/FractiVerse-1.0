// Command interface handler
class CommandInterface {
    constructor() {
        console.log("Initializing CommandInterface");
        this.outputElement = document.getElementById('command-output');
        this.inputElement = document.getElementById('command-input');
        this.metricsElement = document.getElementById('core-metrics');
        
        if (!this.metricsElement) {
            console.error("Could not find core-metrics element");
        }
        
        this.setupEventListeners();
        console.log("CommandInterface initialization complete");
    }

    setupEventListeners() {
        this.inputElement.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendCommand();
            }
        });
    }

    sendCommand() {
        const command = this.inputElement.value;
        if (!command) return;

        // Add command to output
        this.addToOutput(`> ${command}`, 'command');
        
        // Send to server
        if (window.navigator && window.navigator.ws) {
            window.navigator.ws.send(JSON.stringify({
                type: 'command',
                command: command
            }));
        }

        this.inputElement.value = '';
    }

    addToOutput(text, type = 'response') {
        const entry = document.createElement('div');
        entry.className = `output-entry ${type}`;
        entry.textContent = text;
        this.outputElement.appendChild(entry);
        this.outputElement.scrollTop = this.outputElement.scrollHeight;
    }

    updateMetrics(metrics) {
        try {
            console.log("Updating metrics:", metrics);
            if (!this.metricsElement) {
                this.metricsElement = document.getElementById('core-metrics');
            }
            if (!this.metricsElement) {
                console.error("Could not find core-metrics element");
                return;
            }
            
            this.metricsElement.innerHTML = `
                <div class="metric">
                    <span class="label">FPU Level:</span>
                    <span class="value">${(metrics.fpu_level * 100).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="label">Coherence:</span>
                    <span class="value">${(metrics.coherence * 100).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="label">Patterns:</span>
                    <span class="value">${metrics.pattern_count}</span>
                </div>
                <div class="metric">
                    <span class="label">Learning Rate:</span>
                    <span class="value">${metrics.learning_rate.toFixed(4)}</span>
                </div>
            `;
        } catch (error) {
            console.error("Error updating metrics:", error, error.stack);
        }
    }
}

// Initialize command interface
document.addEventListener('DOMContentLoaded', () => {
    window.commandInterface = new CommandInterface();
}); 