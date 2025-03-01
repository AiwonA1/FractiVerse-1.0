import time
import numpy as np
from typing import Dict
from .base import FractiComponent

class MetricsManager(FractiComponent):
    """Metrics management component"""
    
    @property
    def required_dependencies(self) -> list[str]:
        return []  # No dependencies required
        
    def __init__(self):
        super().__init__()
        # Define constraints first
        self.stage_constraints = {
            'newborn': {
                'min': 0.0001,
                'max': 0.01,
                'growth_rate': 0.0001
            },
            'early_learning': {
                'min': 0.01,
                'max': 0.1,
                'growth_rate': 0.0005
            },
            'intermediate': {
                'min': 0.1,
                'max': 0.3,
                'growth_rate': 0.001
            },
            'advanced': {
                'min': 0.3,
                'max': 0.6,
                'growth_rate': 0.002
            },
            'master': {
                'min': 0.6,
                'max': 1.0,
                'growth_rate': 0.005
            }
        }
        
        # Initialize with absolute minimum values
        self.metrics = {
            'fpu_level': 0.0001,  # Start at absolute minimum
            'pattern_recognition': 0.0,
            'learning_efficiency': 0.0,
            'reasoning_depth': 0.0,
            'memory_coherence': 0.0,
            'integration_level': 0.0,
            'total_patterns': 0
        }
        
        # Track real activity
        self.learning_activity = []
        self.last_update = time.time()
        
        print("\n📊 Metrics System Initialized:")
        print(f"├── FPU Level: {self.metrics['fpu_level']:.6f}")
        print(f"└── Stage: {self.get_current_stage().upper()}")
        
    async def _initialize(self) -> None:
        """Ensure proper initialization"""
        try:
            # Force reset to ensure clean state
            self.reset_metrics()
            self.logger.info("Metrics system initialized")
            
        except Exception as e:
            self.logger.error(f"Metrics initialization error: {str(e)}")
            raise

    def reset_metrics(self):
        """Reset all metrics to initial state"""
        try:
            # Start at absolute minimum values
            self.metrics = {
                'fpu_level': self.stage_constraints['newborn']['min'],  # Start at newborn minimum
                'pattern_recognition': 0.0,
                'learning_efficiency': 0.0,
                'reasoning_depth': 0.0,
                'memory_coherence': 0.0,
                'integration_level': 0.0,
                'total_patterns': 0
            }
            
            print("\n🔄 Metrics Reset:")
            print(f"├── FPU Level: {self.metrics['fpu_level']:.6f}")
            print(f"└── Stage: {self.get_current_stage().upper()}")
            
        except Exception as e:
            self.logger.error(f"Metrics reset error: {e}")
            # Ensure safe defaults
            self.metrics = {k: 0.0 for k in self.metrics.keys()}
            self.metrics['fpu_level'] = 0.0001

    def force_fpu_level(self, level: float):
        """Force FPU to a specific level (for testing/reset)"""
        self.metrics['fpu_level'] = max(0.0001, min(1.0, level))
        print(f"\n⚠️ FPU Level forced to: {self.metrics['fpu_level']:.6f}")
        
    def get_current_stage(self) -> str:
        """Get current development stage based on FPU level"""
        fpu = self.metrics['fpu_level']
        
        for stage, constraints in self.stage_constraints.items():
            if constraints['min'] <= fpu <= constraints['max']:
                return stage
        return 'newborn'  # Default to newborn if something's wrong
        
    def validate_metrics(self) -> bool:
        """Validate metrics are within proper bounds"""
        try:
            # Get current stage constraints
            stage = self.get_current_stage()
            constraints = self.stage_constraints[stage]
            
            # Validate FPU level
            if not constraints['min'] <= self.metrics['fpu_level'] <= constraints['max']:
                self.logger.error(f"Invalid FPU level: {self.metrics['fpu_level']}")
                self.reset_metrics()
                return False
            
            # Validate percentage metrics are 0-100
            percentage_metrics = [
                'pattern_recognition', 'learning_efficiency', 
                'reasoning_depth', 'memory_coherence', 
                'integration_level'
            ]
            
            for metric in percentage_metrics:
                if not 0 <= self.metrics[metric] <= 100:
                    self.logger.error(f"Invalid {metric}: {self.metrics[metric]}")
                    self.metrics[metric] = 0.0
            
            # Validate pattern count
            if self.metrics['total_patterns'] < 0:
                self.logger.error(f"Invalid pattern count: {self.metrics['total_patterns']}")
                self.metrics['total_patterns'] = 0
            
            return True
        
        except Exception as e:
            self.logger.error(f"Metrics validation error: {e}")
            self.reset_metrics()
            return False

    def update_metrics(self, memory_data: Dict) -> None:
        """Update metrics using actual memory data"""
        try:
            # Validate memory data
            if not isinstance(memory_data, dict) or 'patterns' not in memory_data:
                self.logger.error("Invalid memory data structure")
                return
            
            # Store previous metrics for comparison
            previous_metrics = self.metrics.copy()
            
            # Calculate real metrics
            total_patterns = len(memory_data['patterns'])
            total_connections = sum(len(conns) for conns in memory_data['connections'].values())
            
            # Calculate coherence
            coherence_scores = [p.get('coherence', 0) for p in memory_data['patterns'].values()]
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # Calculate growth factors
            pattern_factor = min(1.0, total_patterns / 1000)
            connection_density = min(1.0, total_connections / max(1, total_patterns * 10))
            coherence_factor = min(1.0, avg_coherence / 100.0)
            
            # Get stage constraints
            stage = self.get_current_stage()
            constraints = self.stage_constraints[stage]
            
            # Calculate growth
            growth = (pattern_factor + connection_density + coherence_factor) / 3
            growth *= constraints['growth_rate']
            
            # Update FPU within constraints
            current_fpu = self.metrics['fpu_level']
            new_fpu = min(
                constraints['max'],
                max(constraints['min'], current_fpu + growth)
            )
            
            # Update metrics
            self.metrics.update({
                'fpu_level': new_fpu,
                'pattern_recognition': pattern_factor * 100,
                'learning_efficiency': connection_density * 100,
                'reasoning_depth': min(100, (new_fpu / constraints['max']) * 100),
                'memory_coherence': avg_coherence,
                'integration_level': connection_density * 100,
                'total_patterns': total_patterns
            })
            
            # Validate updated metrics
            if not self.validate_metrics():
                self.logger.error("Metrics validation failed, reverting to previous state")
                self.metrics = previous_metrics
                return
            
            # Log significant changes
            if abs(new_fpu - current_fpu) > 0.0001:
                self.logger.info(f"FPU Level: {current_fpu:.6f} -> {new_fpu:.6f}")
                self.logger.info(f"Memory coherence: {avg_coherence:.1f}%")
                self.logger.info(f"Integration level: {connection_density*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")
            # Revert to previous state on error
            self.metrics = previous_metrics
            
    def get_metrics_report(self) -> str:
        """Get formatted metrics report"""
        stage = self.get_current_stage()
        
        return f"""
📊 System Status:
├── Stage: {stage.upper()}
├── FPU Level: {self.metrics['fpu_level']:.6f}
├── Patterns: {self.metrics['total_patterns']}
├── Recognition: {self.metrics['pattern_recognition']:.1f}%
├── Efficiency: {self.metrics['learning_efficiency']:.1f}%
├── Reasoning: {self.metrics['reasoning_depth']:.1f}%
├── Coherence: {self.metrics['memory_coherence']:.1f}%
└── Integration: {self.metrics['integration_level']:.1f}%
""" 

    def log_learning_activity(self, pattern_data: Dict, stats: Dict = None):
        """Log detailed learning activity"""
        try:
            # Create detailed activity record
            activity = {
                'timestamp': time.time(),
                'pattern_id': pattern_data['id'],
                'input': pattern_data['input'],
                'type': pattern_data.get('type', 'unknown'),
                'coherence': pattern_data.get('coherence', 0),
                'connections': len(pattern_data.get('connections', [])),
                'stage': self.get_current_stage(),
                'stats': stats or {}
            }
            
            self.learning_activity.append(activity)
            
            print(f"\n📝 Learning Progress:")
            print(f"├── Pattern: {pattern_data['id']}")
            print(f"├── Input: '{pattern_data['input']}'")
            print(f"├── Type: {pattern_data.get('type', 'unknown')}")
            print(f"├── Stage: {self.get_current_stage().upper()}")
            print(f"├── Coherence: {pattern_data.get('coherence', 0):.1f}%")
            print(f"├── Connections: {len(pattern_data.get('connections', []))}")
            if stats:
                print(f"├── Connection Strength: {stats.get('avg_similarity', 0):.2f}")
            print(f"└── Memory Size: {self.metrics['total_patterns']} patterns")
            
            # Show recent learning summary every 5 patterns
            if len(self.learning_activity) % 5 == 0:
                self._show_learning_summary()
            
        except Exception as e:
            print(f"❌ Learning log error: {str(e)}")

    def _show_learning_summary(self):
        """Show summary of recent learning activity"""
        try:
            recent = self.learning_activity[-5:]  # Last 5 patterns
            
            print("\n📚 Recent Learning Summary:")
            print(f"├── Total Patterns: {self.metrics['total_patterns']}")
            print(f"├── Stage: {self.get_current_stage().upper()}")
            print(f"├── FPU Level: {self.metrics['fpu_level']:.6f}")
            print("├── Recent Patterns:")
            
            for activity in recent:
                print(f"│   ├── '{activity['input']}'")
                print(f"│   │   ├── Coherence: {activity['coherence']:.1f}%")
                print(f"│   │   └── Connections: {activity['connections']}")
            
            print("└── Learning Stats:")
            print(f"    ├── Recognition: {self.metrics['pattern_recognition']:.1f}%")
            print(f"    ├── Efficiency: {self.metrics['learning_efficiency']:.1f}%")
            print(f"    └── Integration: {self.metrics['integration_level']:.1f}%")
            
        except Exception as e:
            print(f"❌ Summary error: {str(e)}") 