#!/usr/bin/env python3
"""
Script to run the TOOL-BOX API server
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'python-multipart'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

    return True

def main():
    """Main function to run the API server."""
    print("TOOL-BOX API Server")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists('api_server.py'):
        print("Error: api_server.py not found in current directory")
        print("Please run this script from the TOOL-BOX root directory")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Import and run the server
    try:
        import uvicorn

        print("Starting TOOL-BOX API server...")
        print("API will be available at: http://localhost:8000")
        print("API documentation at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)

        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure all TOOL-BOX modules are available")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()