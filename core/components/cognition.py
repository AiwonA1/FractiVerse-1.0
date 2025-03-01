"""Cognition Component"""

import torch
import numpy as np
from typing import Dict, List
from datetime import datetime
import asyncio
import json
import os
from ..base import FractiComponent

class FractalCognition(FractiComponent):
    def __init__(self):
        super().__init__()
        self.memory = None  # Will be linked by system
        
        # Initialize cognitive metrics with defaults
        self.cognitive_metrics = {
            'fpu_level': 0.0001,  # Starting FPU level
            'pattern_recognition': 0.0,
            'learning_efficiency': 0.0,
            'reasoning_depth': 0.0,
            'cognitive_coherence': 0.0
        }
        
        print("✅ Cognitive System initialized")
        
        # Load cognitive templates and datasets
        self.templates = self._load_cognitive_templates()
        self.knowledge_base = self._load_knowledge_base()
        self.response_patterns = self._load_response_patterns()
        
        print("\n📚 Cognitive Resources Loaded:")
        print(f"├── Templates: {len(self.templates)} cognitive templates")
        print(f"├── Knowledge Base: {len(self.knowledge_base)} concepts")
        print(f"└── Response Patterns: {len(self.response_patterns)} patterns")

    def _load_cognitive_templates(self) -> Dict:
        """Load cognitive processing templates"""
        try:
            with open('data/cognitive_templates.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Template loading error: {str(e)}")
            # Provide basic fallback templates
            return {
                'basic_pattern': {
                    'recognition': 'Identifying basic pattern: {}',
                    'learning': 'Learning new concept: {}',
                    'synthesis': 'Synthesizing information about: {}'
                },
                'neural_pathways': {
                    'formation': 'Forming neural connections for: {}',
                    'strengthening': 'Strengthening understanding of: {}',
                    'integration': 'Integrating knowledge about: {}'
                }
            }

    def _load_knowledge_base(self) -> Dict:
        """Load base knowledge datasets"""
        try:
            with open('data/knowledge_base.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Knowledge base loading error: {str(e)}")
            return {
                'core_concepts': {},
                'relationships': {},
                'patterns': {}
            }

    def _load_response_patterns(self) -> List:
        """Load response generation patterns"""
        try:
            with open('data/response_patterns.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Response patterns loading error: {str(e)}")
            return [
                "Understanding concept: {}",
                "Analyzing pattern: {}",
                "Processing information about: {}"
            ]

    async def initialize(self):
        """Initialize cognitive systems"""
        await super().initialize()
        print("🧠 Cognitive Systems initializing...")
        
        # Start cognitive monitoring
        asyncio.create_task(self._monitor_cognition())
        return True

    async def _monitor_cognition(self):
        """Monitor and update cognitive metrics"""
        while True:
            try:
                # Update cognitive metrics
                self.cognitive_metrics['fpu_level'] = min(1.0, self.cognitive_metrics['fpu_level'] * 1.01)
                self.cognitive_metrics['pattern_recognition'] = min(1.0, self.cognitive_metrics['pattern_recognition'] + 0.001)
                self.cognitive_metrics['learning_efficiency'] = min(1.0, self.cognitive_metrics['learning_efficiency'] + 0.002)
                self.cognitive_metrics['reasoning_depth'] = min(1.0, self.cognitive_metrics['reasoning_depth'] + 0.001)
                self.cognitive_metrics['cognitive_coherence'] = min(1.0, self.cognitive_metrics['cognitive_coherence'] + 0.003)
                
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Cognitive monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def process(self, input_data: Dict) -> Dict:
        """Process input through cognitive system"""
        try:
            content = input_data['content']
            print(f"\n🧠 Processing Input at FPU Level: {self.cognitive_metrics['fpu_level']*100:.2f}%")
            print(f"├── Pattern Recognition: {self.cognitive_metrics['pattern_recognition']*100:.1f}%")
            print(f"└── Learning Efficiency: {self.cognitive_metrics['learning_efficiency']*100:.1f}%")
            
            # Match input against knowledge base
            matched_concepts = self._match_knowledge_base(content)
            print(f"\n📚 Knowledge Matching:")
            print(f"├── Matched Concepts: {len(matched_concepts)}")
            for concept in matched_concepts[:3]:  # Show top 3 matches
                print(f"├── {concept['type']}: {concept['name']}")
                print(f"└── Confidence: {concept['confidence']*100:.1f}%")
            
            # Select cognitive template based on FPU level
            template = self._select_cognitive_template(self.cognitive_metrics['fpu_level'])
            print(f"\n🔄 Using Cognitive Template: {template['name']}")
            
            # Generate response using template and knowledge
            response = self._generate_response(content, template, matched_concepts)
            
            # Create learning context
            context = {
                'timestamp': datetime.now().isoformat(),
                'fpu_level': self.cognitive_metrics['fpu_level'],
                'cognitive_state': self.cognitive_metrics.copy(),
                'matched_concepts': matched_concepts,
                'template_used': template['name']
            }
            
            # Learn pattern with enhanced context
            learning_results = await self.memory.learn_pattern({
                'content': content,
                'type': 'user_input',
                'cognitive_level': self.cognitive_metrics['fpu_level'],
                'concepts': [c['name'] for c in matched_concepts],
                'template': template['name']
            }, context)
            
            # Log cognitive growth
            print(f"\n📈 Cognitive Growth Status:")
            print(f"├── FPU Level: {self.cognitive_metrics['fpu_level']*100:.2f}%")
            print(f"├── Pattern Recognition: {self.cognitive_metrics['pattern_recognition']*100:.1f}%")
            print(f"├── Learning Efficiency: {self.cognitive_metrics['learning_efficiency']*100:.1f}%")
            print(f"├── Reasoning Depth: {self.cognitive_metrics['reasoning_depth']*100:.1f}%")
            print(f"└── Cognitive Coherence: {self.cognitive_metrics['cognitive_coherence']*100:.1f}%")
            
            return {
                'content': response,
                'metrics': {
                    'fpu_level': self.cognitive_metrics['fpu_level'],
                    'cognitive_metrics': self.cognitive_metrics,
                    'memory_metrics': self.memory.get_metrics(),
                    'concepts_used': [c['name'] for c in matched_concepts],
                    'template_used': template['name']
                }
            }
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            return {'content': f"Processing error: {str(e)}"}

    def _match_knowledge_base(self, content: str) -> List[Dict]:
        """Match input against knowledge base concepts"""
        matches = []
        words = set(content.lower().split())
        
        for concept_type, concepts in self.knowledge_base.items():
            for name, data in concepts.items():
                keywords = set(data.get('keywords', []))
                if keywords:
                    overlap = len(words.intersection(keywords))
                    if overlap > 0:
                        confidence = overlap / len(keywords)
                        matches.append({
                            'type': concept_type,
                            'name': name,
                            'confidence': confidence,
                            'data': data
                        })
        
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)

    def _select_cognitive_template(self, fpu_level: float) -> Dict:
        """Select appropriate cognitive template based on FPU level"""
        if fpu_level < 0.1:
            return {
                'name': 'basic_recognition',
                'pattern': self.templates['basic_pattern']['recognition']
            }
        elif fpu_level < 0.3:
            return {
                'name': 'neural_formation',
                'pattern': self.templates['neural_pathways']['formation']
            }
        elif fpu_level < 0.6:
            return {
                'name': 'knowledge_integration',
                'pattern': self.templates['neural_pathways']['integration']
            }
        else:
            return {
                'name': 'advanced_synthesis',
                'pattern': self.templates['basic_pattern']['synthesis']
            }

    def _generate_response(self, content: str, template: Dict, concepts: List[Dict]) -> str:
        """Generate response using template and matched concepts"""
        if not concepts:
            return template['pattern'].format(content)
            
        concept_names = [c['name'] for c in concepts[:2]]
        concept_str = ' and '.join(concept_names)
        return template['pattern'].format(concept_str)

    async def _learn_pattern(self, content: str):
        """Learn from interaction"""
        try:
            pattern_id = f"pattern_{len(self.pattern_memory)}"
            self.pattern_memory[pattern_id] = {
                'content': content,
                'strength': self.learning_rate * self.cognitive_metrics['fpu_level']
            }
        except Exception as e:
            print(f"Learning error: {str(e)}") 