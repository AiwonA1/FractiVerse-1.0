import logging
from typing import Dict

class CognitiveDisplay:
    """Display component for cognitive metrics"""
    def __init__(self):
        self.metrics = {
            'fpu_level': 0.0001,
            'pattern_recognition': 0.0,
            'learning_efficiency': 0.0,
            'reasoning_depth': 0.0
        }
        
    def update(self, metrics: Dict):
        """Update display with validated metrics"""
        try:
            # Validate FPU level
            fpu = float(metrics.get('fpu_level', 0.0001))
            if not 0.0001 <= fpu <= 1.0:
                fpu = 0.0001
                
            # Validate percentage metrics
            pattern_recog = min(100, max(0, float(metrics.get('pattern_recognition', 0))))
            learning_eff = min(100, max(0, float(metrics.get('learning_efficiency', 0))))
            reasoning = min(100, max(0, float(metrics.get('reasoning_depth', 0))))
            
            # Store validated metrics
            self.metrics = {
                'fpu_level': fpu,
                'pattern_recognition': pattern_recog,
                'learning_efficiency': learning_eff,
                'reasoning_depth': reasoning
            }
            
        except Exception as e:
            print(f"Cognitive display update error: {e}")
            # Keep existing metrics on error
            
    def render(self):
        """Render metrics display"""
        return f"""
🧠 Cognitive Metrics:
├── FPU Level: {self.metrics['fpu_level']*100:.2f}%
├── Pattern Recognition: {self.metrics['pattern_recognition']:.1f}%
├── Learning Efficiency: {self.metrics['learning_efficiency']:.1f}%
└── Reasoning Depth: {self.metrics['reasoning_depth']:.1f}%
"""

class FractiUI:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.engine = None  # Will be set during initialization
        self.cognitive_display = None
        self.memory_display = None
        self.learning_progress = None

    async def initialize(self, engine):
        """Initialize UI with engine reference"""
        self.engine = engine
        await self.setup_displays()
        
    async def setup_displays(self):
        """Setup UI display components"""
        # Initialize display components
        self.cognitive_display = CognitiveDisplay()
        self.memory_display = MemoryDisplay()
        self.learning_progress = LearningProgress()
        
    async def update_metrics(self):
        """Update UI with current metrics"""
        try:
            if not self.engine:
                self.logger.error("UI not initialized with engine")
                return
                
            # Get raw metrics from engine
            raw_metrics = await self.engine.get_ui_metrics()
            
            # Debug log raw metrics
            print("\n🔍 UI Metrics Debug:")
            print(f"├── Raw FPU: {raw_metrics['cognitive_metrics']['fpu_level']:.6f}")
            
            # Update cognitive metrics with strict validation
            cognitive = raw_metrics.get('cognitive_metrics', {})
            self.cognitive_display.update({
                'fpu_level': max(0.0001, min(1.0, cognitive.get('fpu_level', 0.0001))),
                'pattern_recognition': max(0, min(100, cognitive.get('pattern_recognition', 0))),
                'learning_efficiency': max(0, min(100, cognitive.get('learning_efficiency', 0))),
                'reasoning_depth': max(0, min(100, cognitive.get('reasoning_depth', 0)))
            })
            
            # Debug log processed metrics
            print(f"└── Processed FPU: {self.cognitive_display.metrics['fpu_level']:.6f}")
            
            # Update memory status with validation
            memory = raw_metrics.get('memory_status', {})
            self.memory_display.update({
                'coherence': max(0, min(100, memory.get('coherence', 0))),
                'integration': max(0, min(100, memory.get('integration', 0))),
                'pattern_count': max(0, memory.get('pattern_count', 0)),
                'stage': memory.get('learning_stage', 'NEWBORN').upper()
            })
            
            # Update learning progress
            progress = raw_metrics.get('learning_progress', {})
            self.learning_progress.update(progress)
            
        except Exception as e:
            self.logger.error(f"UI update error: {e}")
            self.show_error_state()
            
    def show_error_state(self):
        """Show error state in UI"""
        safe_defaults = self._get_safe_default_metrics()
        self.cognitive_display.update(safe_defaults['cognitive'])
        self.memory_display.update(safe_defaults['memory'])
        
    def format_metrics_for_display(self, metrics: Dict) -> Dict:
        """Format and validate metrics before display"""
        try:
            # Get raw metrics
            fpu = metrics.get('fpu_level', 0.0001)
            
            # Validate FPU is within bounds
            if not 0.0001 <= fpu <= 1.0:
                self.logger.error(f"Invalid FPU value: {fpu}, resetting to minimum")
                fpu = 0.0001
                
            # Format cognitive metrics with validation
            cognitive_metrics = {
                'fpu_level': f"{fpu * 100:.2f}%",
                'pattern_recognition': f"{min(100, max(0, metrics.get('pattern_recognition', 0))):.1f}%",
                'learning_efficiency': f"{min(100, max(0, metrics.get('learning_efficiency', 0))):.1f}%",
                'reasoning_depth': f"{min(100, max(0, metrics.get('reasoning_depth', 0))):.1f}%"
            }
            
            # Format memory metrics with validation
            memory_metrics = {
                'coherence': f"{min(100, max(0, metrics.get('memory_coherence', 0))):.1f}%",
                'integration': f"{min(100, max(0, metrics.get('integration_level', 0))):.1f}%",
                'pattern_count': max(0, metrics.get('total_patterns', 0)),
                'stage': metrics.get('stage', 'newborn').upper()
            }
            
            return {
                'cognitive': cognitive_metrics,
                'memory': memory_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Metrics formatting error: {e}")
            return self._get_safe_default_metrics()
            
    def _get_safe_default_metrics(self) -> Dict:
        """Get safe default metrics for display"""
        return {
            'cognitive': {
                'fpu_level': "0.01%",
                'pattern_recognition': "0.0%",
                'learning_efficiency': "0.0%", 
                'reasoning_depth': "0.0%"
            },
            'memory': {
                'coherence': "0.0%",
                'integration': "0.0%",
                'pattern_count': 0,
                'stage': "NEWBORN"
            }
        }
        
    def update_cognitive_display(self, metrics: Dict):
        """Update cognitive metrics display"""
        self.cognitive_display.clear()
        self.cognitive_display.write(f"""
🧠 Cognitive Metrics:
├── FPU Level: {metrics['fpu_level']*100:.2f}%
├── Pattern Recognition: {metrics['pattern_recognition']:.1f}%
├── Learning Efficiency: {metrics['learning_efficiency']:.1f}%
└── Reasoning Depth: {metrics['reasoning_depth']:.1f}%
""")
        
    def update_memory_status(self, status: Dict):
        """Update memory status display"""
        self.memory_display.clear()
        self.memory_display.write(f"""
💾 Memory Status:
├── Coherence: {status['coherence']:.1f}%
├── Integration: {status['integration']:.1f}%
├── Patterns: {status['pattern_count']}
└── Stage: {status['learning_stage'].upper()}
""") 