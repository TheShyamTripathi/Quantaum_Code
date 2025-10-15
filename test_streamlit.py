#!/usr/bin/env python3
"""
Test script to verify Streamlit application components
"""

def test_imports():
    """Test if all required imports work"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import qiskit
        print("✅ Qiskit imported successfully")
    except ImportError as e:
        print(f"❌ Qiskit import failed: {e}")
        return False
    
    try:
        from qiskit_aer import AerSimulator
        print("✅ Qiskit-Aer imported successfully")
    except ImportError as e:
        print(f"❌ Qiskit-Aer import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_shor_algorithm():
    """Test the ShorAlgorithm class"""
    try:
        from shor_streamlit import ShorAlgorithm
        shor = ShorAlgorithm()
        print("✅ ShorAlgorithm class created successfully")
        return True
    except Exception as e:
        print(f"❌ ShorAlgorithm class failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Streamlit Application Components")
    print("=" * 50)
    
    # Test imports
    print("🔍 Testing imports...")
    if not test_imports():
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return
    
    # Test ShorAlgorithm class
    print("\n🔍 Testing ShorAlgorithm class...")
    if not test_shor_algorithm():
        print("❌ ShorAlgorithm class test failed")
        return
    
    print("\n✅ All tests passed!")
    print("🚀 Ready to run: streamlit run shor_streamlit.py")

if __name__ == "__main__":
    main()
