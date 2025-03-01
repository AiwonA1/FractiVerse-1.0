from fastapi import FastAPI
import time
from .fractal_cognition import FractalCognition
from .memory_manager import MemoryManager
from .fracti_decision_engine import FractiDecisionEngine
from .fracti_fpu import FractiProcessingUnit
from .metrics_manager import MetricsManager
from typing import Dict
import asyncio

class FractiCodyEngine:
    """Core engine managing component lifecycle"""
    
    def __init__(self):
        self._initialized = False
        self.metrics_manager = MetricsManager()
        self.components = {}
        
    async def initialize(self):
        """Initialize all components in dependency order"""
        try:
            print("\n🚀 Initializing FractiCody system...")
            
            # Initialize metrics manager first
            await self.metrics_manager.initialize()
            
            # Create and initialize memory manager
            self.components['memory'] = MemoryManager()
            if not await self.components['memory'].initialize(
                metrics_manager=self.metrics_manager
            ):
                raise Exception("Failed to initialize memory")
                
            # Start continuous metrics updates
            asyncio.create_task(self._update_metrics_loop())
            
            # Create and initialize cognition with memory
            self.components['cognition'] = FractalCognition()
            if not await self.components['cognition'].initialize(
                memory_manager=self.components['memory'],
                metrics_manager=self.metrics_manager
            ):
                raise Exception("Failed to initialize cognition")
                
            # Create and initialize decision engine
            self.components['decision'] = FractiDecisionEngine()
            if not await self.components['decision'].initialize(
                memory_manager=self.components['memory'],
                metrics_manager=self.metrics_manager,
                cognition=self.components['cognition']
            ):
                raise Exception("Failed to initialize decision")
                
            # Create and initialize processing unit
            self.components['processor'] = FractiProcessingUnit()
            if not await self.components['processor'].initialize(
                memory_manager=self.components['memory'],
                metrics_manager=self.metrics_manager
            ):
                raise Exception("Failed to initialize processor")
            
            self._initialized = True
            print("\n✅ System initialization complete")
            
            # Launch browser interface in separate thread
            import threading
            browser_thread = threading.Thread(target=self._launch_browser)
            browser_thread.daemon = True  # Make thread exit when main program exits
            browser_thread.start()
            
        except Exception as e:
            print(f"❌ System initialization failed: {str(e)}")
            raise

    async def _update_metrics_loop(self):
        """Continuously update metrics from real data"""
        while True:
            try:
                # Get real memory data
                memory_data = self.components['memory'].long_term_memory
                
                # Debug logging
                print("\n🔍 Debug Metrics Update:")
                print(f"├── Memory Data: {len(memory_data['patterns'])} patterns")
                print(f"├── Before FPU: {self.metrics_manager.metrics['fpu_level']:.6f}")
                
                # Update metrics with real data
                self.metrics_manager.update_metrics(memory_data)
                
                # Debug logging
                print(f"└── After FPU: {self.metrics_manager.metrics['fpu_level']:.6f}")
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"❌ Metrics update error: {e}")
                await asyncio.sleep(1)

    def _launch_browser(self):
        """Launch browser interface"""
        try:
            import webbrowser
            import time
            import socket
            
            # Wait for server to be ready
            time.sleep(3)  # Increased wait time
            
            # Try to find the active server port
            ports = [8000, 8001, 8002, 8003]
            active_port = None
            
            for port in ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    # Try to connect to the port
                    result = sock.connect_ex(('127.0.0.1', port))
                    if result == 0:  # Port is open and in use
                        active_port = port
                        break
                finally:
                    sock.close()
            
            if active_port:
                url = f"http://127.0.0.1:{active_port}"
                webbrowser.open(url)
                print(f"\n🌐 Launched interface at {url}")
            else:
                print("\n❌ Could not find active server port")
                print("💡 Available ports checked:", ports)
                
        except Exception as e:
            print(f"\n❌ Browser launch error: {str(e)}")
            print("💡 Try opening http://127.0.0.1:8000 manually")

    # Property getters to ensure components are accessible
    @property
    def fractal_cognition(self):
        return self.components['cognition']

    @property
    def memory_manager(self):
        return self.components['memory']

    @property
    def decision_engine(self):
        return self.components['decision']

    @property
    def processing_unit(self):
        return self.components['processor']

    def start(self):
        """Starts the AI engine"""
        try:
            print("🔹 Activating Fractal Cognition...")
            
            # Verify all components
            if not all([
                self.fractal_cognition,
                self.memory_manager,
                self.decision_engine,
                self.processing_unit
            ]):
                raise Exception("Component verification failed")
            
            # Activate core component
            self.fractal_cognition.activate()
            
            print(f"✅ FractiCody Running at Cognition Level: {self.cognition_level}")
            return True
            
        except Exception as e:
            print(f"❌ Start-up failed: {str(e)}")
            raise

    async def process_input(self, user_input):
        """Process user input"""
        try:
            # Process through memory first
            memory_pattern = await self.memory_manager.process_input(user_input)
            
            if memory_pattern:
                # Process through fractal cognition
                result = await self.fractal_cognition.process_input(memory_pattern)
                
                # Show learning progress
                print(self.memory_manager.get_learning_progress())
                
                return result
            return "Error processing input"
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            return "Error processing input"

    def make_decision(self, context, options):
        """Make a decision"""
        try:
            return self.decision_engine.process_decision(context, options)
        except Exception as e:
            print(f"❌ Decision error: {str(e)}")
            return None

    async def reset_system(self):
        """Reset system to initial state"""
        try:
            print("\n🔄 Resetting system...")
            
            # Reset metrics
            self.metrics_manager.reset_metrics()
            
            # Reset memory
            await self.memory_manager.reset_memory()
            
            # Reset other components
            await self.fractal_cognition.reset()
            await self.decision_engine.reset()
            await self.processing_unit.reset()
            
            print("\n📊 System Reset Complete:")
            print(self.metrics_manager.get_metrics_report())
            
        except Exception as e:
            print(f"❌ Reset error: {str(e)}")

    def get_ui_metrics(self) -> Dict:
        """Get formatted metrics for UI display"""
        try:
            if not self._initialized:
                return self._get_default_metrics()
            
            # Get raw metrics with validation
            metrics = self.metrics_manager.metrics
            
            # Validate FPU level
            fpu = max(0.0001, min(1.0, metrics.get('fpu_level', 0.0001)))
            
            # Debug log
            print(f"\n🔍 Engine Metrics Debug:")
            print(f"├── Raw FPU: {metrics['fpu_level']:.6f}")
            print(f"└── Validated FPU: {fpu:.6f}")
            
            return {
                'system_status': {
                    'status': 'Connected ✅',
                    'stage': self.metrics_manager.get_current_stage().upper(),
                    'last_update': time.strftime('%H:%M:%S')
                },
                'cognitive_metrics': {
                    'fpu_level': fpu,  # Validated FPU level
                    'pattern_recognition': max(0, min(100, metrics.get('pattern_recognition', 0))),
                    'learning_efficiency': max(0, min(100, metrics.get('learning_efficiency', 0))),
                    'reasoning_depth': max(0, min(100, metrics.get('reasoning_depth', 0)))
                },
                'memory_status': {
                    'coherence': max(0, min(100, metrics.get('memory_coherence', 0))),
                    'integration': max(0, min(100, metrics.get('integration_level', 0))),
                    'pattern_count': max(0, metrics.get('total_patterns', 0)),
                    'learning_stage': self.metrics_manager.get_current_stage()
                },
                'learning_progress': {
                    'recent_patterns': self.components['memory'].get_recent_patterns(),
                    'activity_log': self.components['memory'].get_activity_log()
                }
            }
            
        except Exception as e:
            self.logger.error(f"UI metrics error: {e}")
            return self._get_default_metrics()
            
    def _get_default_metrics(self) -> Dict:
        """Get safe default metrics if error occurs"""
        return {
            'system_status': {
                'status': 'Initializing...',
                'stage': 'NEWBORN',
                'last_update': time.strftime('%H:%M:%S')
            },
            'cognitive_metrics': {
                'fpu_level': 0.0001,  # Raw minimum value
                'pattern_recognition': 0.0,
                'learning_efficiency': 0.0,
                'reasoning_depth': 0.0
            },
            'memory_status': {
                'coherence': 0.0,
                'integration': 0.0,
                'pattern_count': 0,
                'learning_stage': 'newborn'
            },
            'learning_progress': {
                'recent_patterns': [],
                'activity_log': []
            }
        }
