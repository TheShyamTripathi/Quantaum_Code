#!/usr/bin/env python3
"""
Demo script to showcase the enhanced circuit visualization features
"""

import streamlit as st
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shor_streamlit import ShorAlgorithm, create_educational_plots

def demo_circuit_visualization():
    """Demonstrate the circuit visualization features"""
    
    st.title("🔬 Shor's Algorithm - Circuit Visualization Demo")
    st.markdown("This demo showcases the enhanced circuit visualization and educational features.")
    
    # Create ShorAlgorithm instance
    shor = ShorAlgorithm()
    
    # Demo parameters
    N = 15
    n_count = 6
    shots = 1024
    
    st.header("🚀 Running Shor's Algorithm Demo")
    
    # Run the algorithm
    with st.spinner("Executing quantum algorithm..."):
        factors = shor.shors_algorithm(
            N=N,
            n_count=n_count,
            shots=shots,
            tries=3,
            show_circuit=True
        )
    
    if factors:
        st.success(f"✅ Found factors: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}")
    else:
        st.warning("❌ No factors found")
    
    # Show all the new features
    st.header("🔧 Circuit Visualization Features")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 Detailed Circuit", "🔧 Components", "📊 Objects & Functions", 
        "📐 Mathematical Foundation", "⚙️ Implementation"
    ])
    
    with tab1:
        st.subheader("📐 Detailed Quantum Circuit")
        shor.display_detailed_circuit()
    
    with tab2:
        st.subheader("🔧 Circuit Components Explanation")
        shor.display_circuit_components()
    
    with tab3:
        st.subheader("🔧 Circuit Objects and Functions")
        shor.display_circuit_objects()
    
    with tab4:
        st.subheader("📐 Mathematical Foundation")
        shor.display_mathematical_foundation()
    
    with tab5:
        st.subheader("⚙️ Implementation Details")
        shor.display_implementation_details()
    
    # Show educational content
    st.header("📚 Educational Content")
    
    # Show probability analysis
    if hasattr(shor, 'measurement_results') and shor.measurement_results:
        shor.display_probabilities_streamlit()
    
    # Show step-by-step explanation
    if hasattr(shor, 'step_explanations') and shor.step_explanations:
        shor.display_step_by_step_streamlit()
    
    # Show quantum concepts
    st.header("🔬 Quantum Concepts")
    fig = create_educational_plots()
    st.pyplot(fig)

def main():
    """Main demo function"""
    demo_circuit_visualization()

if __name__ == "__main__":
    main()
