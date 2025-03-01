"""Memory Manager Component"""

import torch
import numpy as np
from datetime import datetime
import json
import os
import asyncio
import random
from typing import Dict, List, Optional
from ..quantum.vector import FractalVector3D
from ..quantum.hologram import QuantumHologram
from ..base import FractiComponent

class MemoryManager(FractiComponent):
    def __init__(self):
        super().__init__()
        self.patterns = {}
        self.learning_rate = 0.001
        self.memory_file = "memory/patterns.json"
        
        # Initialize metrics
        self.memory_metrics = {
            'pattern_count': 0,
            'integration_level': 0.0,
            'pattern_complexity': 0.0,
            'memory_coherence': 0.0,
            'learning_stage': 'newborn',
            'active_patterns': set()
        }
        
        # Initialize cognitive metrics (will be linked by system)
        self.cognitive_metrics = {
            'fpu_level': 0.0001,
            'pattern_recognition': 0.0,
            'learning_efficiency': 0.0,
            'reasoning_depth': 0.0
        }
        
        # Initialize memory structures
        self.active_memory = torch.zeros((256, 256))
        self.pattern_relationships = {}
        self.activity_log = []
        
        # Load existing memory
        os.makedirs('memory', exist_ok=True)
        self.load_memory()
        
    async def initialize(self):
        """Initialize memory systems"""
        await super().initialize()
        print("📚 Memory Manager initializing...")
        # Start memory monitoring
        asyncio.create_task(self._monitor_memory())
        return True
        
    async def _monitor_memory(self):
        """Monitor memory growth and integration"""
        print("🔄 Starting memory monitoring...")
        while True:
            try:
                await self._memory_growth()
                await asyncio.sleep(15)
            except Exception as e:
                print(f"Memory monitoring error: {str(e)}")
                await asyncio.sleep(15)
                
    async def _memory_growth(self):
        """Handle memory growth and pattern strengthening"""
        try:
            # Process active patterns
            for pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                
                # Strengthen pattern
                new_strength = min(1.0, pattern['strength'] * 1.01)
                pattern['strength'] = new_strength
                
                # Log significant strengthening
                if new_strength > 0.5 and new_strength - pattern['strength'] > 0.1:
                    self.activity_log.append({
                        'type': 'consolidation',
                        'message': f"🧠 Pattern {pattern_id[:8]} consolidated with {new_strength*100:.1f}% strength"
                    })
            
            # Form new relationships
            self._form_relationships()
            
            # Update integration level
            self._update_integration_level()
            
            # Generate learning insights
            if random.random() < 0.3:  # 30% chance
                insight = self._generate_learning_insight()
                self.activity_log.append({
                    'type': 'insight',
                    'message': f"💡 {insight}"
                })
            
            # Save memory periodically
            self.save_memory()
            
        except Exception as e:
            print(f"Memory growth error: {e}")
            
    def _form_relationships(self):
        """Form relationships between patterns"""
        for pattern_id, pattern in self.patterns.items():
            for other_id, other in self.patterns.items():
                if pattern_id != other_id:
                    similarity = self._calculate_similarity(pattern, other)
                    if similarity > 0.7:  # High similarity threshold
                        if other_id not in pattern['relationships']:
                            pattern['relationships'].append(other_id)
                            self.activity_log.append({
                                'type': 'connection',
                                'message': f"🔗 Connected patterns {pattern_id[:8]} and {other_id[:8]}"
                            })
                            
    def _calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate pattern similarity"""
        # Simple similarity based on cognitive level and strength
        level_diff = abs(pattern1['cognitive_level'] - pattern2['cognitive_level'])
        strength_similarity = 1 - abs(pattern1['strength'] - pattern2['strength'])
        return (1 - level_diff + strength_similarity) / 2
        
    def _generate_learning_insight(self) -> str:
        """Generate insight about learning progress"""
        total_patterns = len(self.patterns)
        strong_patterns = len([p for p in self.patterns.values() if p['strength'] > 0.7])
        total_connections = sum(len(p['relationships']) for p in self.patterns.values())
        
        insights = [
            f"Developed {strong_patterns} strong memory patterns",
            f"Formed {total_connections} cognitive connections",
            f"Memory coherence at {self.memory_metrics['memory_coherence']*100:.1f}%",
            f"Integration level: {self.memory_metrics['integration_level']*100:.1f}%"
        ]
        
        return random.choice(insights)
        
    def load_memory(self):
        """Load stored memory patterns"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.patterns = json.load(f)
                print(f"📚 Loaded {len(self.patterns)} memory patterns")
            else:
                print("📝 Creating new memory storage")
                self.save_memory()
        except Exception as e:
            print(f"Memory load error: {str(e)}")
            self.patterns = {}
            
    def save_memory(self):
        """Save memory patterns"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.patterns, f)
        except Exception as e:
            print(f"Memory save error: {str(e)}")
            
    async def learn_pattern(self, pattern: Dict, context: Optional[Dict] = None) -> Dict:
        """Learn new pattern with detailed cognitive processing"""
        try:
            pattern_id = self._generate_pattern_id(pattern)
            
            # Analyze input pattern
            pattern_analysis = self._analyze_pattern(pattern)
            
            # Log cognitive processing start
            self.activity_log.append({
                'type': 'learning_details',
                'message': f"\n🧠 Cognitive Processing:\n"
                          f"├── Input: {pattern['content'][:50]}...\n"
                          f"├── Type: {pattern_analysis['type']}\n"
                          f"├── Complexity: {pattern_analysis['complexity']*100:.1f}%\n"
                          f"└── Processing at FPU Level: {self.cognitive_metrics['fpu_level']*100:.2f}%"
            })
            
            # Store pattern with enhanced metadata
            self.patterns[pattern_id] = {
                'content': pattern['content'],
                'timestamp': datetime.now().isoformat(),
                'strength': 0.1,
                'relationships': [],
                'cognitive_level': pattern.get('cognitive_level', 0.0),
                'analysis': pattern_analysis,
                'context': context or {},
                'learning_history': [],
                'activation_count': 0
            }
            
            # Process through cognitive layers
            await self._process_cognitive_layers(pattern_id)
            
            # Form initial relationships
            formed_relationships = self._form_initial_relationships(pattern_id)
            
            # Log relationship formation
            if formed_relationships:
                self.activity_log.append({
                    'type': 'integration_details',
                    'message': f"\n🔄 Neural Integration:\n"
                              f"├── Pattern: {pattern_id[:8]}\n"
                              f"├── Formed Relationships: {len(formed_relationships)}\n"
                              f"└── Integration Level: {self.memory_metrics['integration_level']*100:.1f}%"
                })
            
            # Update metrics
            self._update_metrics()
            
            # Log pattern consolidation
            self.activity_log.append({
                'type': 'pattern_details',
                'message': f"\n💡 Pattern Consolidated:\n"
                          f"├── ID: {pattern_id[:8]}\n"
                          f"├── Strength: {self.patterns[pattern_id]['strength']*100:.1f}%\n"
                          f"├── Relationships: {len(self.patterns[pattern_id]['relationships'])}\n"
                          f"└── Cognitive Level: {self.patterns[pattern_id]['cognitive_level']*100:.1f}%"
            })
            
            return {
                'status': 'success',
                'pattern_id': pattern_id,
                'analysis': pattern_analysis,
                'metrics': self.memory_metrics
            }
            
        except Exception as e:
            print(f"Learning error: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _process_cognitive_layers(self, pattern_id: str):
        """Process pattern through cognitive layers"""
        pattern = self.patterns[pattern_id]
        
        # Layer 1: Pattern Recognition
        recognition_score = self._calculate_recognition_score(pattern)
        
        # Layer 2: Semantic Analysis
        semantic_features = self._extract_semantic_features(pattern)
        
        # Layer 3: Cognitive Integration
        integration_level = self._calculate_integration_level(pattern)
        
        # Update pattern with processing results
        pattern['learning_history'].append({
            'timestamp': datetime.now().isoformat(),
            'recognition_score': recognition_score,
            'semantic_features': semantic_features,
            'integration_level': integration_level,
            'fpu_level': self.cognitive_metrics['fpu_level']
        })
        
        # Log cognitive processing
        self.activity_log.append({
            'type': 'learning_details',
            'message': f"\n🔄 Cognitive Processing Layers:\n"
                      f"├── Pattern Recognition: {recognition_score*100:.1f}%\n"
                      f"├── Semantic Features: {len(semantic_features)}\n"
                      f"└── Integration Level: {integration_level*100:.1f}%"
        })

    def _calculate_recognition_score(self, pattern: Dict) -> float:
        """Calculate pattern recognition score"""
        # Basic implementation - can be enhanced
        base_score = 0.3
        if len(pattern['content']) > 50:
            base_score += 0.2
        if pattern['analysis']['type'] == 'logical':
            base_score += 0.3
        return min(1.0, base_score)

    def _extract_semantic_features(self, pattern: Dict) -> List[str]:
        """Extract semantic features from pattern"""
        content = pattern['content'].lower()
        features = []
        
        # Extract key cognitive indicators
        if 'why' in content or 'how' in content:
            features.append('analytical')
        if 'is' in content or 'are' in content:
            features.append('declarative')
        if 'if' in content or 'then' in content:
            features.append('logical')
        if '?' in content:
            features.append('interrogative')
            
        return features

    def _form_initial_relationships(self, pattern_id: str) -> List[str]:
        """Form initial pattern relationships"""
        pattern = self.patterns[pattern_id]
        formed_relationships = []
        
        for other_id, other in self.patterns.items():
            if other_id != pattern_id:
                similarity = self._calculate_similarity(pattern, other)
                if similarity > 0.7:  # High similarity threshold
                    pattern['relationships'].append(other_id)
                    formed_relationships.append(other_id)
                    
                    # Log relationship formation
                    self.activity_log.append({
                        'type': 'connection',
                        'message': f"🔗 Connected patterns {pattern_id[:8]} and {other_id[:8]} (similarity: {similarity:.2f})"
                    })
                    
        return formed_relationships

    def _generate_pattern_id(self, pattern: Dict) -> str:
        """Generate unique pattern ID"""
        content = str(pattern['content'])
        timestamp = datetime.now().isoformat()
        return f"pattern_{hash(content + timestamp) % 10000:04d}"
        
    def _update_metrics(self):
        """Update memory metrics"""
        total_patterns = len(self.patterns)
        self.memory_metrics.update({
            'pattern_count': total_patterns,
            'pattern_complexity': min(1.0, total_patterns * 0.025),
            'memory_coherence': min(1.0, self.cognitive_metrics['fpu_level'] * 2)
        })

    def get_metrics(self) -> Dict:
        """Get current memory metrics"""
        return {
            'pattern_count': len(self.patterns),
            'integration_level': self.memory_metrics['integration_level'],
            'pattern_complexity': self.memory_metrics['pattern_complexity'],
            'memory_coherence': self.memory_metrics['memory_coherence']
        }

    def _update_integration_level(self):
        """Update and log integration level details"""
        try:
            total_patterns = len(self.patterns)
            if total_patterns > 0:
                # Calculate relationships and density
                total_relationships = sum(len(p.get('relationships', [])) 
                                       for p in self.patterns.values())
                density = total_relationships / (total_patterns * (total_patterns - 1))
                
                # Update integration metrics
                self.memory_metrics['integration_level'] = min(1.0, density * 2)
                
                # Log detailed integration status
                self.activity_log.append({
                    'type': 'integration_details',
                    'message': (
                        f"\n📊 Integration Analysis:\n"
                        f"├── Patterns: {total_patterns}\n"
                        f"├── Relationships: {total_relationships}\n"
                        f"├── Network Density: {density*100:.1f}%\n"
                        f"└── Integration Level: {self.memory_metrics['integration_level']*100:.1f}%"
                    )
                })

                # Log pattern details if significant changes
                strong_patterns = [p for p in self.patterns.values() if p.get('strength', 0) > 0.7]
                if strong_patterns:
                    pattern_details = "\n🧠 Strong Pattern Analysis:"
                    for i, pattern in enumerate(strong_patterns[:3], 1):  # Show top 3
                        pattern_details += (
                            f"\n├── Pattern {i}:\n"
                            f"│   ├── Content: {pattern['content'][:50]}...\n"
                            f"│   ├── Strength: {pattern['strength']*100:.1f}%\n"
                            f"│   ├── Cognitive Level: {pattern['cognitive_level']*100:.1f}%\n"
                            f"│   └── Relationships: {len(pattern['relationships'])}"
                        )
                    self.activity_log.append({
                        'type': 'pattern_details',
                        'message': pattern_details
                    })

        except Exception as e:
            print(f"Integration update error: {e}")

    def _analyze_pattern(self, pattern: Dict) -> Dict:
        """Analyze pattern characteristics"""
        content = pattern['content']
        
        # Analyze pattern complexity
        complexity = min(1.0, len(content) / 100)  # Simple length-based complexity
        
        # Determine pattern type
        if any(marker in content.lower() for marker in ['?', 'how', 'why', 'what']):
            pattern_type = 'query'
        elif any(marker in content.lower() for marker in ['if', 'then', 'because', 'therefore']):
            pattern_type = 'logical'
        else:
            pattern_type = 'statement'
            
        return {
            'type': pattern_type,
            'complexity': complexity,
            'length': len(content),
            'timestamp': datetime.now().isoformat()
        }

    def _log_metric_changes(self, old_metrics: Dict):
        """Log significant metric changes"""
        changes = []
        for key in old_metrics:
            if key in self.memory_metrics:
                old_val = old_metrics[key]
                new_val = self.memory_metrics[key]
                if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
                    if abs(new_val - old_val) > 0.05:  # 5% change threshold
                        changes.append(f"├── {key}: {old_val*100:.1f}% → {new_val*100:.1f}%")
        
        if changes:
            self.activity_log.append({
                'type': 'metric_changes',
                'message': "📈 Significant Metric Changes:\n" + "\n".join(changes)
            }) 