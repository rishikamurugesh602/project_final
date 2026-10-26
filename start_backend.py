#!/usr/bin/env python
"""
Smart startup script for the Deepfake Voice Interceptor backend.
Handles port conflicts and firewall issues gracefully.
"""

import subprocess
import sys
import socket
import time

def is_port_available(host, port):
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result != 0

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available('127.0.0.1', port):
            return port
    return None

def start_backend(port):
    """Start the FastAPI backend on specified port."""
    print(f"\n{'='*60}")
    print(f"Starting Backend on http://127.0.0.1:{port}")
    print(f"{'='*60}\n")
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--reload"
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=False)
        
    except KeyboardInterrupt:
        print("\n\nBackend stopped (Ctrl+C pressed)")
    except Exception as e:
        print(f"\nError starting backend: {e}")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("DEEPFAKE VOICE INTERCEPTOR - BACKEND STARTUP")
    print("="*60)
    print(f"Python: {sys.executable}\n")
    
    # Try to find available port
    preferred_port = 8000
    available_port = find_available_port(preferred_port)
    
    if available_port is None:
        print("ERROR: Could not find any available port between 8000-8009")
        print("\nTroubleshooting:")
        print("1. Check if another FastAPI app is running")
        print("2. Disable Windows Firewall temporarily")
        print("3. Restart your computer")
        sys.exit(1)
    
    if available_port != preferred_port:
        print(f"⚠ Port {preferred_port} is in use, using {available_port} instead\n")
    else:
        print(f"✓ Port {preferred_port} is available\n")
    
    print("FIREWALL NOTE:")
    print("-" * 60)
    print("If Windows Firewall blocks this, you'll see a prompt.")
    print("Click 'Allow access' to enable the backend API.")
    print("-" * 60)
    
    # Start backend
    start_backend(available_port)

if __name__ == "__main__":
    main()
