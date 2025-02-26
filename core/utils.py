"""Utility functions for FractiVerse"""
import os
import socket
from typing import Optional

def find_free_port(start_port: int, max_attempts: int = 10) -> Optional[int]:
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def force_exit():
    """Force exit the application"""
    print("\n⚠️ Forcing application shutdown...")
    os._exit(0) 