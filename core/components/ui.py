"""UI Interface Component"""

from typing import Dict
from ..base import FractiComponent

class FractiUI(FractiComponent):
    """UI Interface handler"""
    
    def __init__(self):
        super().__init__()
        self.cognition = None
        
    def format_input(self, user_input: str) -> Dict:
        """Format user input"""
        return {
            'type': 'user_input',
            'content': user_input,
            'format': 'text'
        }
        
    def format_response(self, response: Dict) -> str:
        """Format response"""
        if isinstance(response, dict):
            return response.get('text', str(response))
        return str(response) 