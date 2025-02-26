"""Server management functions for FractiVerse"""
import asyncio
import signal
import uvicorn
from fastapi import FastAPI
import structlog
from typing import Optional

logger = structlog.get_logger("fractiverse_server")

def shutdown_server(server: Optional[uvicorn.Server] = None, shutdown_event: Optional[asyncio.Event] = None):
    """Handle graceful shutdown on signals"""
    print("\nℹ️ Received shutdown signal, stopping server...")
    if server:
        server.should_exit = True
    if shutdown_event:
        shutdown_event.set()

async def shutdown_after_timeout(timeout: int = 3, force_exit_func=None):
    """Force shutdown after timeout"""
    try:
        await asyncio.sleep(timeout)
        print(f"\n⚠️ Server timeout reached ({timeout}s), forcing shutdown...")
        
        # Try graceful shutdown first
        shutdown_server()
        
        # Give it a second to shutdown gracefully
        await asyncio.sleep(1)
        
        # Force exit if still running
        if force_exit_func:
            force_exit_func()
    except Exception as e:
        print(f"❌ Error during shutdown: {str(e)}")
        if force_exit_func:
            force_exit_func()

def start_server(app: FastAPI, port: int, debug: bool = False, is_testing: bool = False) -> uvicorn.Server:
    """Initialize and start the FractiVerse server"""
    try:
        logger.info("Initializing FractiVerse server")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, lambda s, f: shutdown_server())
        signal.signal(signal.SIGTERM, lambda s, f: shutdown_server())
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info" if debug else "warning",
            reload=debug,
            timeout_keep_alive=30 if not is_testing else 2,
            timeout_graceful_shutdown=5 if not is_testing else 1,
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        return server
    
    except Exception as e:
        logger.error("Server initialization failed", error=str(e))
        raise

async def run_server(server: uvicorn.Server, is_testing: bool = False, force_exit_func=None):
    """Run the server with shutdown handling"""
    try:
        await server.serve()
    except Exception as e:
        logger.error("Server error", error=str(e))
    finally:
        if is_testing and force_exit_func:
            force_exit_func() 