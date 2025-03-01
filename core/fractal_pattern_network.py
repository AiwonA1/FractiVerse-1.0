import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class FractalPatternNetwork:
    """Dynamic pattern network using fractal resonance"""
    def __init__(self):
        self.pattern_nodes = {}
        self.pattern_connections = csr_matrix((0, 0))
        self.node_index = {}
        self.resonance_threshold = 0.1
        
    def learn(self, pattern_data):
        """Learn new patterns through natural resonance"""
        # Convert pattern to node representation
        node = self._pattern_to_node(pattern_data)
        
        # Find resonating patterns
        resonances = self._find_resonating_patterns(node)
        
        # Form new connections
        self._form_connections(node, resonances)
        
        # Update network structure
        self._update_network()
        
        return self._get_pattern_info(node)
        
    def _pattern_to_node(self, pattern):
        """Convert pattern to network node"""
        # Generate unique node ID
        node_id = len(self.pattern_nodes)
        
        # Create node with pattern features
        node = {
            'id': node_id,
            'pattern': pattern,
            'features': self._extract_features(pattern),
            'connections': set(),
            'strength': 0.0
        }
        
        self.pattern_nodes[node_id] = node
        self.node_index[node_id] = len(self.node_index)
        return node
        
    def _extract_features(self, pattern):
        """Extract key features from pattern"""
        if isinstance(pattern, np.ndarray):
            # For unipixel patterns
            features = {
                'mean': pattern.mean(),
                'std': pattern.std(),
                'max': pattern.max(),
                'min': pattern.min(),
                'gradient': np.gradient(pattern).mean()
            }
        else:
            # For other pattern types
            features = {
                'length': len(pattern),
                'complexity': self._calculate_complexity(pattern)
            }
        return features 

    def _find_resonating_patterns(self, node):
        """Find patterns that resonate with the new node"""
        resonances = {}
        for node_id, existing in self.pattern_nodes.items():
            if node_id != node['id']:
                resonance = self._calculate_resonance(node, existing)
                if resonance > self.resonance_threshold:
                    resonances[node_id] = resonance
        return resonances

    def _calculate_resonance(self, node1, node2):
        """Calculate resonance between two patterns"""
        # Feature similarity
        feature_sim = self._feature_similarity(node1['features'], node2['features'])
        
        # Pattern interaction
        interaction = self._pattern_interaction(node1['pattern'], node2['pattern'])
        
        # Structural resonance
        structure = self._structural_resonance(node1, node2)
        
        return (feature_sim + interaction + structure) / 3

    def _form_connections(self, node, resonances):
        """Form connections between resonating patterns"""
        n = len(self.pattern_nodes)
        new_connections = np.zeros((n, n))
        
        # Copy existing connections
        if self.pattern_connections.shape[0] > 0:
            new_connections[:self.pattern_connections.shape[0], 
                           :self.pattern_connections.shape[1]] = self.pattern_connections.toarray()
        
        # Add new connections
        for node_id, strength in resonances.items():
            idx1 = self.node_index[node['id']]
            idx2 = self.node_index[node_id]
            new_connections[idx1, idx2] = strength
            new_connections[idx2, idx1] = strength
        
        self.pattern_connections = csr_matrix(new_connections)

    def _update_network(self):
        """Update network structure and strengths"""
        # Find connected components
        n_components, labels = connected_components(self.pattern_connections)
        
        # Update node strengths based on connectivity
        for node_id in self.pattern_nodes:
            idx = self.node_index[node_id]
            connections = self.pattern_connections[idx].toarray().flatten()
            self.pattern_nodes[node_id]['strength'] = connections.sum() 