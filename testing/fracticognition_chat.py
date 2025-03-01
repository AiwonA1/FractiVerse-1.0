import asyncio
import os
import sys
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.fractal_cognition import FractalCognition

class FractiChat:
    """Natural language interface to FractiCognition"""
    
    def __init__(self):
        print("🧠 Initializing FractiCognition neural network...")
        try:
            self.cognitive_system = FractalCognition()
            # Initialize the cognitive system
            self.cognitive_system.bootstrap_cognition()
            self.cognitive_system.initialize_core_knowledge()
            self.prompt = "You: "
            print("✨ FractiCognition 1.0 is ready!")
        except Exception as e:
            print(f"❌ Initialization error: {str(e)}")
            raise
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_welcome(self):
        self.clear_screen()
        print("\n🌟 FractiCognition Neural Interface 1.0 🌟")
        print("=========================================")
        print("Neural Capacity:", f"{self.cognitive_system.fpu_level*100:.2f}%")
        print("Active Neural Patterns:", len(self.cognitive_system.pattern_network.patterns))
        print("\nInteract naturally or type 'help' for commands")
        print("=========================================\n")
        
    async def print_thinking(self):
        """Show thinking animation"""
        print("\nThinking", end="", flush=True)
        for _ in range(3):
            print(".", end="", flush=True)
            await asyncio.sleep(0.3)
        print("\n")
        
    async def process_input(self, user_input):
        """Process natural language input"""
        try:
            if user_input.lower() == 'help':
                print("\n📚 Commands:")
                print("- expand mind: Grow neural capacity")
                print("- neural status: Show network state")
                print("- clear mind: Reset neural state")
                print("- exit: Close interface")
                print("Otherwise, just chat naturally!\n")
                return
                
            if "expand mind" in user_input.lower():
                print("\n🧠 Growing neural capacity...")
                async for status in self.cognitive_system.expand_cognitive_capacity(100):
                    print(f"Neural growth: {status['fpu_level']*100:.1f}%")
                print("✨ Neural expansion complete\n")
                return
                
            if "neural status" in user_input.lower():
                status = self.cognitive_system.get_status()
                print("\n🧠 Neural Network Status:")
                print(f"Capacity: {status['fpu_level']*100:.1f}%")
                print(f"Active Patterns: {status['active_patterns']}")
                print(f"Pattern Complexity: {status.get('complexity', 0):.3f}")
                print(f"Learning Rate: {status.get('learning', 0):.3f}")
                print(f"Memory Integration: {status.get('memory', 0):.3f}\n")
                return
                
            if user_input.lower() == 'clear mind':
                self.cognitive_system = FractalCognition()
                print("\n🧠 Neural state reset\n")
                return
                
            # Process natural language through neural network
            await self.print_thinking()
            response = await self.cognitive_system.process_command(user_input)
            print(f"FractiCognition: {response}\n")
            
        except Exception as e:
            print(f"\n❌ Neural processing error: {str(e)}\n")
            
    async def run(self):
        """Run interactive neural interface"""
        self.print_welcome()
        
        while True:
            try:
                user_input = input(self.prompt).strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("\n👋 Closing neural interface...")
                    break
                    
                await self.process_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Neural interface closed")
                break
                
            except Exception as e:
                print(f"\n❌ Interface error: {str(e)}\n")

if __name__ == "__main__":
    print("🧠 Starting FractiCognition neural interface...")
    chat = FractiChat()
    asyncio.run(chat.run()) 