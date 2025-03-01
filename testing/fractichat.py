import asyncio
import os
import sys
from datetime import datetime
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.fractal_cognition import FractalCognition

class FractiChat:
    """ChatGPT-like interface for FractiCognition"""
    
    def __init__(self):
        print("\n🧠 Initializing FractiCognition neural network...")
        try:
            self.cognitive_system = FractalCognition()
            self.cognitive_system.bootstrap_cognition()
            self.cognitive_system.initialize_core_knowledge()
            self.history = []
            self.prompt = "\n\033[94mYou:\033[0m "  # Blue color for user prompt
            print("\n✨ FractiCognition 1.0 is ready for conversation!")
        except Exception as e:
            print(f"\n❌ Initialization error: {str(e)}")
            raise
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_welcome(self):
        self.clear_screen()
        print("\n" + "="*50)
        print("🌟 FractiCognition Chat Interface 1.0")
        print("="*50)
        print("\nCurrent Neural Capacity:", f"{self.cognitive_system.fpu_level*100:.2f}%")
        print("Active Neural Patterns:", len(self.cognitive_system.pattern_network.patterns))
        print("\nSpecial Commands:")
        print("- /expand    : Grow neural capacity")
        print("- /status    : Show neural network status")
        print("- /clear     : Clear conversation")
        print("- /help      : Show this help")
        print("- /exit      : End conversation")
        print("\nOtherwise, just chat naturally!")
        print("="*50 + "\n")
        
    async def print_thinking(self):
        """Show thinking animation"""
        print("\n\033[93mFractiCognition:\033[0m Thinking", end="", flush=True)
        for _ in range(3):
            print(".", end="", flush=True)
            await asyncio.sleep(0.3)
        print()
        
    async def process_input(self, user_input):
        """Process user input"""
        try:
            # Handle special commands
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                
                if command == 'help':
                    self.print_welcome()
                    return
                    
                elif command == 'expand':
                    print("\n\033[93mFractiCognition:\033[0m Starting neural expansion...")
                    async for status in self.cognitive_system.expand_cognitive_capacity(100):
                        print(f"📈 Neural growth: {status['fpu_level']*100:.1f}%")
                        print(f"🔄 Active Patterns: {status.get('patterns', 0)}")
                        print(f"✨ Resonance Events: {status.get('resonance', 0)}")
                        await asyncio.sleep(0.1)
                    print("\n✅ Neural expansion complete!")
                    return
                    
                elif command == 'status':
                    status = self.cognitive_system.get_status()
                    print("\n\033[93mFractiCognition Neural Status:\033[0m")
                    print(f"📊 Neural Capacity: {status['fpu_level']*100:.1f}%")
                    print(f"🧠 Active Patterns: {status['active_patterns']}")
                    print(f"🔄 Pattern Complexity: {status.get('complexity', 0):.3f}")
                    print(f"📈 Learning Rate: {status.get('learning', 0):.3f}")
                    print(f"💫 Memory Integration: {status.get('memory', 0):.3f}")
                    return
                    
                elif command == 'clear':
                    self.clear_screen()
                    self.print_welcome()
                    return
                    
                elif command == 'exit':
                    print("\n👋 Goodbye! Thanks for chatting!")
                    sys.exit(0)
            
            # Process natural language input
            await self.print_thinking()
            response = await self.cognitive_system.process_command(user_input)
            
            # Print response in yellow
            print(f"\033[93mFractiCognition:\033[0m {response}")
            
            # Save to history
            self.history.append({
                'user': user_input,
                'assistant': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"\n❌ Processing error: {str(e)}")
            
    async def run(self):
        """Run interactive chat interface"""
        self.print_welcome()
        
        while True:
            try:
                user_input = input(self.prompt).strip()
                
                if not user_input:
                    continue
                    
                await self.process_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                break
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🌟 Starting FractiCognition Chat Interface...")
    chat = FractiChat()
    asyncio.run(chat.run()) 