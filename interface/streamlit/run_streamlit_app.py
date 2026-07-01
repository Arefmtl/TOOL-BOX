#!/usr/bin/env python3
"""
Script to run the TOOL-BOX Streamlit ML Platform
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'loguru'
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
    """Main function to run the Streamlit app."""
    print("TOOL-BOX v2.0 - Professional ML Platform")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        # Try to find app.py relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, 'app.py')
        if not os.path.exists(app_path):
            print("Error: app.py not found in current directory")
            print("Please run this script from the interface/streamlit directory")
            sys.exit(1)
        # Change to the script directory
        os.chdir(script_dir)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Run Streamlit app
    try:
        print("Starting TOOL-BOX Streamlit application...")
        print("The app will open in your default web browser")
        print("Press Ctrl+C to stop the application")
        print("=" * 60)

        # Run streamlit
        os.system("streamlit run app.py --server.headless true --server.port 8501")

    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()