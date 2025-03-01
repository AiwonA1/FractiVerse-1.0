"""Base Component System for FractiCody"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class FractiComponent(ABC):
    """Abstract base component with lifecycle management"""
    
    def __init__(self):
        self.initialized = False
        self.active = False
        self.metrics = {}
        self.dependencies = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def required_dependencies(self) -> list[str]:
        """List of required dependency names"""
        return []

    async def initialize(self, **dependencies) -> bool:
        """Initialize component with dependencies"""
        try:
            # Store dependencies
            self.dependencies = dependencies
            
            # Verify required dependencies
            missing = [dep for dep in self.required_dependencies 
                      if dep not in dependencies]
            if missing:
                self.logger.error(f"Missing required dependencies: {missing}")
                return False
            
            # Store dependencies as attributes
            for name, component in dependencies.items():
                setattr(self, name, component)
            
            # Component-specific initialization
            await self._initialize()
            
            self.initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False

    @abstractmethod
    async def _initialize(self) -> None:
        """Component-specific initialization logic"""
        pass

    async def reset(self) -> bool:
        """Reset component to initial state"""
        try:
            self.initialized = False
            self.active = False
            self.metrics = {}
            return await self.initialize(**self.dependencies)
        except Exception as e:
            self.logger.error(f"Reset error: {str(e)}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return self.metrics

    def activate(self):
        """Activate the component"""
        if self.initialized:
            self.active = True
            return True
        return False
        
    def deactivate(self):
        """Deactivate the component"""
        self.active = False

    def get_metrics(self) -> Dict:
        """Get component metrics"""
        return self.metrics 