#!/usr/bin/env python3
"""
Launcher script for Shor's Algorithm Streamlit Application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'qiskit',
        'qiskit_aer',
        'numpy',
        'matplotlib',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🔬 Shor's Algorithm - Streamlit Application")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('shor_streamlit.py'):
        print("❌ Error: shor_streamlit.py not found in current directory")
        print("   Please run this script from the Quantum directory")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    print("✅ All dependencies found!")
    print("🚀 Starting Streamlit application...")
    print("=" * 50)
    print("📱 The application will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'shor_streamlit.py',
            '--server.headless', 'false',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
