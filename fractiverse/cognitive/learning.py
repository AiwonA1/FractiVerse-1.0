"""Cognitive learning system module."""
import numpy as np
import json
from datetime import datetime
from pathlib import Path

class LearningSystem:
    """Main class for cognitive learning system."""
    
    def __init__(self, test_mode=False):
        """Initialize learning system.
        
        Args:
            test_mode (bool): Whether to run in test mode
        """
        self.test_mode = test_mode
        self.patterns = []
        self.learning_rate = 0.1
        self.memory = AdaptiveMemory()
        self.pattern_recognizer = PatternRecognition()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the learning system."""
        if not self._initialized:
            self.memory.initialize()
            await self.pattern_recognizer.initialize()
            self._initialized = True
        
    async def shutdown(self):
        """Shutdown the learning system."""
        if self._initialized:
            await self.pattern_recognizer.shutdown()
            self._initialized = False
        
    async def learn_pattern(self, pattern):
        """Learn a new pattern.
        
        Args:
            pattern (np.ndarray): Pattern to learn
            
        Returns:
            dict: Learning result with success and confidence
        """
        if not self._initialized:
            await self.initialize()
            
        self.patterns.append(pattern)
        confidence = await self.pattern_recognizer.compute_confidence(pattern)
        self.memory.store_pattern(pattern)
        
        # Adjust learning rate based on confidence
        if confidence > 0.8:
            self.learning_rate *= 0.9
        else:
            self.learning_rate *= 1.1
        self.learning_rate = np.clip(self.learning_rate, 0.01, 1.0)
        
        return {"success": True, "confidence": confidence}
        
    async def get_learned_patterns(self):
        """Get all learned patterns.
        
        Returns:
            list: List of learned patterns
        """
        if not self._initialized:
            await self.initialize()
        return self.patterns
        
    async def get_memory_capacity(self):
        """Get current memory capacity.
        
        Returns:
            int: Memory capacity
        """
        if not self._initialized:
            await self.initialize()
        return self.memory.capacity
        
    async def get_memory_usage(self):
        """Get current memory usage.
        
        Returns:
            float: Memory usage ratio (0-1)
        """
        if not self._initialized:
            await self.initialize()
        return len(self.patterns) / self.memory.capacity
        
    async def save_state(self):
        """Save current learning state.
        
        Returns:
            str: State ID
        """
        if not self._initialized:
            await self.initialize()
            
        state_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        state = {
            "patterns": [p.tolist() for p in self.patterns],
            "learning_rate": self.learning_rate
        }
        
        if not self.test_mode:
            with open(f"learning_state_{state_id}.json", "w") as f:
                json.dump(state, f)
        
        return state_id
        
    async def load_state(self, state_id):
        """Load learning state.
        
        Args:
            state_id (str): State ID to load
            
        Returns:
            bool: Success status
        """
        if not self._initialized:
            await self.initialize()
            
        if self.test_mode:
            return True
            
        try:
            with open(f"learning_state_{state_id}.json", "r") as f:
                state = json.load(f)
            
            self.patterns = [np.array(p) for p in state["patterns"]]
            self.learning_rate = state["learning_rate"]
            return True
        except:
            return False
            
    async def clear(self):
        """Clear all learned patterns."""
        if not self._initialized:
            await self.initialize()
            
        self.patterns = []
        self.learning_rate = 0.1
        self.memory.clear()
        
    async def recognize_pattern(self, pattern):
        """Recognize a pattern.
        
        Args:
            pattern (np.ndarray): Pattern to recognize
            
        Returns:
            dict: Recognition result
        """
        if not self._initialized:
            await self.initialize()
            
        confidence = await self.pattern_recognizer.compute_confidence(pattern)
        recognized = confidence > 0.6
        
        return {
            "recognized": recognized,
            "confidence": confidence
        }

class PatternRecognition:
    """Pattern recognition system."""
    
    def __init__(self):
        """Initialize pattern recognition system."""
        self.threshold = 0.6
        self._initialized = False
        
    async def initialize(self):
        """Initialize the recognition system."""
        self._initialized = True
        
    async def shutdown(self):
        """Shutdown the recognition system."""
        self._initialized = False
        
    async def compute_confidence(self, pattern):
        """Compute confidence score for pattern recognition.
        
        Args:
            pattern (np.ndarray): Pattern to evaluate
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simple implementation for testing
        if np.all(pattern >= 0) and np.all(pattern <= 1):
            return 0.9
        return 0.3

class AdaptiveMemory:
    """Adaptive memory system."""
    
    def __init__(self):
        """Initialize adaptive memory system."""
        self.capacity = 1000
        self.patterns = []
        self._initialized = False
        
    def initialize(self):
        """Initialize the memory system."""
        self._initialized = True
        
    def store_pattern(self, pattern):
        """Store a pattern in memory.
        
        Args:
            pattern (np.ndarray): Pattern to store
        """
        self.patterns.append(pattern)
        if len(self.patterns) > self.capacity * 0.9:
            self.capacity *= 1.5
            
    def clear(self):
        """Clear memory contents."""
        self.patterns = []
        self.capacity = 1000 