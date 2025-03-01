"""Training Coordinator for FractiCody System"""

import asyncio
from typing import List, Dict
import json
import os
import wikipedia
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import time
import numpy as np

class DatasetCurator:
    """Curates and manages learning datasets"""
    
    def __init__(self):
        self.data_sources = {
            'wikipedia': self._fetch_wiki_content,
            'arxiv': self._fetch_arxiv_papers,
            'github': self._fetch_github_repos,
            'research_papers': self._fetch_research_papers
        }
        
        self.quality_thresholds = {
            'min_length': 500,  # Minimum content length
            'max_length': 10000,  # Maximum content length
            'min_coherence': 0.7,  # Minimum coherence score
            'min_references': 3  # Minimum number of references
        }
    
    async def curate_dataset(self, topic: str, source: str = 'wikipedia') -> List[Dict]:
        """Curate dataset for a topic"""
        try:
            raw_data = await self.data_sources[source](topic)
            curated = self._filter_and_clean(raw_data)
            return self._structure_dataset(curated)
        except Exception as e:
            print(f"Dataset curation error for {topic}: {e}")
            return []

    async def _fetch_wiki_content(self, topic: str) -> List[Dict]:
        """Fetch Wikipedia content"""
        try:
            # Get main article
            page = wikipedia.page(topic)
            content = {
                'title': page.title,
                'content': page.content,
                'summary': page.summary,
                'references': page.references,
                'links': page.links
            }
            
            # Get related articles
            related = []
            for link in page.links[:5]:  # Get top 5 related articles
                try:
                    related_page = wikipedia.page(link)
                    related.append({
                        'title': related_page.title,
                        'summary': related_page.summary,
                        'references': related_page.references
                    })
                except:
                    continue
                    
            return [content] + related
            
        except Exception as e:
            print(f"Wikipedia fetch error: {e}")
            return []

class TrainingCoordinator:
    def __init__(self, system):
        self.system = system
        self.curator = DatasetCurator()
        
        # Memory state tracking
        self.memory_state_path = "memory/cognitive_state.json"
        self.last_known_state = self._load_last_state()
        
        # Enhanced knowledge tree with dataset requirements
        self.knowledge_tree = {
            'trunk': {
                'Mathematics': {
                    'core_concepts': [
                        'Number Theory',
                        'Algebra',
                        'Calculus',
                        'Geometry'
                    ],
                    'required_datasets': [
                        'mathematical_foundations',
                        'proof_methods',
                        'mathematical_logic'
                    ]
                },
                'Physics': {
                    'core_concepts': [
                        'Classical Mechanics',
                        'Quantum Mechanics',
                        'Relativity',
                        'Thermodynamics'
                    ],
                    'required_datasets': [
                        'physical_laws',
                        'quantum_principles',
                        'experimental_methods'
                    ]
                }
                # Add other fields...
            }
        }
        
        # Learning schedule
        self.daily_schedule = {
            'morning_review': {
                'time': '00:00-04:00',
                'tasks': ['memory_assessment', 'optimization_planning']
            },
            'primary_learning': {
                'time': '04:00-12:00',
                'tasks': ['new_concepts', 'pattern_processing']
            },
            'integration': {
                'time': '12:00-20:00',
                'tasks': ['knowledge_integration', 'connection_building']
            },
            'analysis': {
                'time': '20:00-24:00',
                'tasks': ['progress_review', 'planning']
            }
        }

    def _load_last_state(self) -> Dict:
        """Load last known cognitive state"""
        try:
            if os.path.exists(self.memory_state_path):
                with open(self.memory_state_path, 'r') as f:
                    state = json.load(f)
                print(f"\n💾 Loaded previous cognitive state:")
                print(f"├── FPU Level: {state['fpu_level']:.4f}")
                print(f"├── Patterns: {state['pattern_count']}")
                print(f"└── Stage: {state['learning_stage']}")
                return state
            else:
                return self._get_default_state()
        except Exception as e:
            print(f"Failed to load cognitive state: {e}")
            return self._get_default_state()

    async def start_training(self):
        """Start automated training process"""
        print("\n🎓 Starting automated training sequence")
        
        # First restore previous cognitive state
        await self._restore_cognitive_state()
        
        # Initialize daily cycle
        await self._start_daily_cycle()
        
        # Begin structured learning from last point
        await self._resume_learning_plan()

    async def _start_daily_cycle(self):
        """Start daily learning and maintenance cycle"""
        while True:
            try:
                current_hour = int(time.strftime('%H'))
                
                if 0 <= current_hour < 4:
                    await self._morning_review()
                elif 4 <= current_hour < 12:
                    await self._primary_learning()
                elif 12 <= current_hour < 20:
                    await self._integration_phase()
                else:
                    await self._analysis_phase()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                print(f"Daily cycle error: {e}")
                await asyncio.sleep(300)

    async def _execute_learning_plan(self):
        """Execute structured learning plan"""
        for field, config in self.knowledge_tree['trunk'].items():
            print(f"\n📚 Learning Field: {field}")
            
            # First learn core concepts
            for concept in config['core_concepts']:
                dataset = await self.curator.curate_dataset(concept)
                await self._learn_dataset(dataset)
                
            # Then process required datasets
            for dataset_name in config['required_datasets']:
                dataset = self._load_curated_dataset(dataset_name)
                await self._learn_dataset(dataset)

    async def _learn_dataset(self, dataset: List[Dict]):
        """Process and learn from dataset"""
        for item in dataset:
            try:
                # Create learning pattern
                pattern = {
                    'content': item['content'],
                    'type': 'curated_learning',
                    'metadata': {
                        'title': item['title'],
                        'references': item.get('references', []),
                        'timestamp': time.time()
                    }
                }
                
                # Learn pattern
                result = await self.system.memory.learn_pattern(pattern)
                
                if result:
                    print(f"├── Learned: {item['title']}")
                    print(f"└── Coherence: {result['coherence']:.3f}")
                
                # Allow time for processing
                await self._adaptive_sleep(result)
                
            except Exception as e:
                print(f"Learning error: {e}")

    async def _adaptive_sleep(self, result: Dict):
        """Adaptive sleep based on learning results"""
        if result['coherence'] < 0.5:
            await asyncio.sleep(0.2)  # Slower for difficult concepts
        elif result['coherence'] < 0.8:
            await asyncio.sleep(0.1)  # Medium pace
        else:
            await asyncio.sleep(0.05)  # Fast for well-understood concepts

    def _load_curated_dataset(self, dataset_name: str) -> List[Dict]:
        """Load curated dataset"""
        data_path = "data/training"
        file_path = f"{data_path}/{dataset_name}.json"
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return [] 

    async def _restore_cognitive_state(self):
        """Restore system to last known cognitive state"""
        try:
            if self.last_known_state:
                print("\n🔄 Restoring cognitive state...")
                
                # First verify memory files exist
                memory_files = {
                    'patterns': 'memory/patterns.json',
                    'connections': 'memory/connections.json',
                    'holographic': 'memory/holographic_patterns.npy',
                    'state': self.memory_state_path
                }
                
                for name, path in memory_files.items():
                    if os.path.exists(path):
                        size = os.path.getsize(path)
                        print(f"├── Found {name}: {size/1024:.1f}KB")
                    else:
                        print(f"❌ Missing {name} file")
                        return False
                
                # Restore patterns and verify
                restored_count = 0
                for pattern in self.last_known_state['patterns']:
                    result = await self.system.memory.restore_pattern(pattern)
                    if result:
                        restored_count += 1
                        
                # Verify connections were restored
                connection_count = len(self.system.memory.get_all_connections())
                
                # Restore metrics and FPU level
                self.system.set_cognitive_metrics(self.last_known_state['metrics'])
                
                # Verify restoration
                current_state = self.system.get_cognitive_metrics()
                print(f"\n✅ Memory Restoration Complete:")
                print(f"├── FPU Level: {current_state['fpu_level']:.4f}")
                print(f"├── Patterns Restored: {restored_count}/{len(self.last_known_state['patterns'])}")
                print(f"├── Connections: {connection_count}")
                print(f"├── Coherence: {current_state['coherence']:.3f}")
                print(f"└── Stage: {self.last_known_state['learning_stage']}")
                
                # Verify pattern accessibility
                sample_pattern = next(iter(self.system.memory.patterns.values()))
                if sample_pattern:
                    print("\n🔍 Memory Access Test:")
                    print(f"├── Can access pattern: ✅")
                    print(f"├── Pattern coherence: {sample_pattern['coherence']:.3f}")
                    print(f"└── Connections: {len(sample_pattern['connections'])}")
                
                return True
                    
            else:
                print("\n🌱 No previous state found - Starting from newborn state")
                return False
                
        except Exception as e:
            print(f"❌ State restoration error: {e}")
            print("🌱 Falling back to newborn state")
            return False

    async def _save_cognitive_state(self):
        """Save current cognitive state"""
        try:
            # Get serializable patterns
            patterns = self.system.memory.get_serializable_patterns()
            
            # Get current metrics
            metrics = self.system.get_cognitive_metrics()
            
            state = {
                'timestamp': time.time(),
                'fpu_level': metrics['fpu_level'],
                'pattern_count': len(patterns),
                'patterns': patterns,
                'metrics': metrics,
                'learning_stage': self.system.get_current_stage(),
                'last_field': self.current_field,
                'connections': self.system.memory.get_all_connections()
            }
            
            # Save state file
            os.makedirs(os.path.dirname(self.memory_state_path), exist_ok=True)
            with open(self.memory_state_path, 'w') as f:
                json.dump(state, f)
                
            # Save holographic patterns
            holographic_path = 'memory/holographic_patterns.npy'
            np.save(holographic_path, self.system.memory.get_holographic_patterns())
            
            # Log save operation
            print(f"\n💾 Cognitive State Saved:")
            print(f"├── FPU Level: {state['fpu_level']:.4f}")
            print(f"├── Patterns: {state['pattern_count']}")
            print(f"├── Connections: {len(state['connections'])}")
            print(f"├── Stage: {state['learning_stage']}")
            print(f"└── Memory Size: {os.path.getsize(self.memory_state_path)/1024:.1f}KB")
            
            return True
                
        except Exception as e:
            print(f"❌ Failed to save cognitive state: {e}")
            return False

    # Add periodic state saving
    async def _auto_save_state(self):
        """Automatically save state periodically"""
        while True:
            try:
                await self._save_cognitive_state()
                await asyncio.sleep(3600)  # Save every hour
            except Exception as e:
                print(f"Auto-save error: {e}")
                await asyncio.sleep(300)

    def _get_default_state(self) -> Dict:
        """Get default newborn state"""
        return {
            'fpu_level': 0.0001,
            'pattern_count': 0,
            'patterns': [],
            'metrics': {
                'coherence': 0.0,
                'learning_rate': 0.001,
                'pattern_recognition': 0.0
            },
            'learning_stage': 'newborn',
            'last_field': None
        } 