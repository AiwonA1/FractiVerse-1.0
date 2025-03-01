"""Knowledge Harvesting System for FractiCody"""

import wikipedia  # This is the correct package
import asyncio
from typing import List, Dict
import networkx as nx
from datetime import datetime
import re
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class KnowledgeHarvester:
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.learning_paths = {
            'quantum_computing': [
                'Quantum computing',
                'Quantum entanglement',
                'Quantum superposition',
                'Quantum neural network',
                'Quantum machine learning'
            ],
            'neuroscience': [
                'Neural coding',
                'Neuroplasticity',
                'Cognitive neuroscience',
                'Neural oscillation',
                'Memory consolidation'
            ],
            'consciousness': [
                'Consciousness',
                'Integrated information theory',
                'Neural correlates of consciousness',
                'Quantum mind',
                'Cognitive architecture'
            ],
            'advanced_computation': [
                'Artificial neural network',
                'Deep learning',
                'Quantum algorithm',
                'Information theory',
                'Complex systems'
            ],
            'emergent_systems': [
                'Emergence',
                'Self-organization',
                'Complex adaptive system',
                'Dynamical system',
                'Fractal'
            ]
        }
        print("✅ Knowledge Harvester initialized")
        
    async def harvest_knowledge(self, system) -> Dict:
        """Harvest knowledge in expanding cycles"""
        try:
            print("\n🌟 Starting Advanced Knowledge Harvesting\n")
            
            for cycle, (domain, topics) in enumerate(self.learning_paths.items(), 1):
                print(f"\n📚 Learning Cycle {cycle}: {domain.replace('_', ' ').title()}")
                await self._process_domain(system, domain, topics)
                
                # Allow for integration
                await asyncio.sleep(2)
                
                # Check cognitive growth
                metrics = system.memory.get_metrics()
                print(f"\n📊 Cognitive State After {domain}:")
                print(f"├── Integration Level: {metrics['integration_level']*100:.1f}%")
                print(f"├── Pattern Complexity: {metrics['pattern_complexity']*100:.1f}%")
                print(f"└── Memory Coherence: {metrics['memory_coherence']*100:.1f}%")
                
            return {"status": "success", "cycles_completed": len(self.learning_paths)}
            
        except Exception as e:
            print(f"Knowledge harvesting error: {e}")
            return {"status": "error", "message": str(e)}

    async def _process_domain(self, system, domain: str, topics: List[str]):
        """Process a knowledge domain"""
        domain_knowledge = []
        
        for topic in topics:
            try:
                # Get Wikipedia content with error handling
                print(f"\n📖 Fetching: {topic}")
                try:
                    # Search for the page first
                    search_results = wikipedia.search(topic)
                    if not search_results:
                        print(f"⚠️ No results found for {topic}")
                        continue
                        
                    # Get the most relevant page
                    page = wikipedia.page(search_results[0], auto_suggest=False)
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    print(f"⚠️ Disambiguation for {topic}, using first option")
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                except wikipedia.exceptions.PageError:
                    print(f"❌ Page not found for {topic}")
                    continue
                
                # Get content
                summary = page.summary
                content = page.content[:2000]  # First 2000 chars
                
                # Clean text
                content = re.sub(r'\[\d+\]', '', content)  # Remove reference numbers
                
                # Extract key concepts using NLTK for better sentence splitting
                sentences = nltk.sent_tokenize(content)
                concepts = self._extract_key_concepts(content)
                
                # Create knowledge packet
                knowledge = {
                    'topic': topic,
                    'summary': summary,
                    'concepts': concepts,
                    'url': page.url,
                    'timestamp': datetime.now().isoformat()
                }
                domain_knowledge.append(knowledge)
                
                # Process through cognitive system
                await self._cognitive_process(system, knowledge)
                
                print(f"✓ Processed: {topic}")
                print(f"├── Extracted Concepts: {len(concepts)}")
                print(f"├── Content Length: {len(content)} chars")
                print(f"└── URL: {page.url}")
                
            except Exception as e:
                print(f"Error processing {topic}: {e}")
                continue
            
            # Allow for neural integration
            await asyncio.sleep(1)
            
        # Update knowledge graph
        self._update_knowledge_graph(domain, domain_knowledge)

    def _extract_key_concepts(self, content: str) -> List[Dict]:
        """Extract key concepts from content"""
        concepts = []
        sentences = content.split('.')
        
        for sentence in sentences:
            if len(sentence.strip()) > 50:  # Meaningful sentences only
                concepts.append({
                    'content': sentence.strip(),
                    'type': self._determine_concept_type(sentence),
                    'timestamp': datetime.now().isoformat()
                })
                
        return concepts

    def _determine_concept_type(self, sentence: str) -> str:
        """Determine the type of concept"""
        sentence = sentence.lower()
        
        if any(word in sentence for word in ['is', 'are', 'was', 'were']):
            return 'definition'
        elif any(word in sentence for word in ['because', 'therefore', 'thus']):
            return 'relationship'
        elif any(word in sentence for word in ['can', 'could', 'may', 'might']):
            return 'possibility'
        else:
            return 'information'

    async def _cognitive_process(self, system, knowledge: Dict):
        """Process knowledge through cognitive system"""
        try:
            # Process summary for high-level understanding
            await system.process({
                'content': knowledge['summary'],
                'type': 'summary',
                'cognitive_level': 0.7
            })
            
            # Process each concept for detailed learning
            for concept in knowledge['concepts']:
                await system.process({
                    'content': concept['content'],
                    'type': concept['type'],
                    'cognitive_level': 0.8
                })
                
        except Exception as e:
            print(f"Cognitive processing error: {e}")

    def _update_knowledge_graph(self, domain: str, knowledge: List[Dict]):
        """Update knowledge graph with new information"""
        # Add domain node if not exists
        if domain not in self.knowledge_graph:
            self.knowledge_graph.add_node(domain, type='domain')
            
        # Add topics and concepts
        for item in knowledge:
            topic = item['topic']
            self.knowledge_graph.add_node(topic, type='topic')
            self.knowledge_graph.add_edge(domain, topic)
            
            for concept in item['concepts']:
                concept_id = hash(concept['content'])
                self.knowledge_graph.add_node(concept_id, 
                                           type='concept',
                                           content=concept['content'])
                self.knowledge_graph.add_edge(topic, concept_id) 