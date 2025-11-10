#!/usr/bin/env python3
"""
Simple startup script for the AI Lead Management System
"""

import sys
import os
import asyncio
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    """Main entry point"""
    
    print("🚀 Starting AI Lead Management & Voice Communication System...")
    print("=" * 60)
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", str(backend_dir))
    
    # Configuration
    config = {
        "app": "main_simple:app",  # Use simplified main
        "host": "0.0.0.0", 
        "port": 8001,  # Use different port to avoid conflicts
        "reload": True,
        "reload_dirs": [str(backend_dir)],
        "log_level": "info"
    }
    
    print(f"🌐 Starting FastAPI server...")
    print(f"📍 URL: http://localhost:{config['port']}")
    print(f"📖 API Docs: http://localhost:{config['port']}/docs")
    print(f"🧠 NLP Service: http://localhost:{config['port']}/api/v1/nlp/info")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Change to backend directory
        os.chdir(backend_dir)
        
        # Start uvicorn server
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())