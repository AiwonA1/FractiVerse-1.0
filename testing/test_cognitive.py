import asyncio
from interface.cognitive_chat import start_interface

async def run_test():
    """Run cognitive interface test"""
    try:
        print("\n🚀 Launching FractiCognition 1.0 Test")
        await start_interface()
        
    except KeyboardInterrupt:
        print("\n👋 Test completed")
    except Exception as e:
        print(f"\n❌ Test error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_test()) 