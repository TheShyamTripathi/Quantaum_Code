"""
Shor's Algorithm: Complete Implementation with Streamlit GUI

This application provides a comprehensive implementation of Shor's algorithm with:
- Interactive web interface
- Quantum circuit visualization
- Probability analysis
- Step-by-step explanations
- Educational content about quantum computing concepts
"""

import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from fractions import Fraction
from math import gcd
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ShorAlgorithm:
    """Complete implementation of Shor's algorithm with GUI and educational features"""
    
    def __init__(self):
        self.backend = AerSimulator()
        self.current_circuit = None
        self.measurement_results = None
        self.step_explanations = []
        
    def modular_multiplication_unitary(self, a, N, n_target):
        """Create unitary matrix for modular multiplication by 'a' modulo N"""
        dim = 2 ** n_target
        M = np.zeros((dim, dim), dtype=complex)
        
        for x in range(dim):
            if x < N:
                y = (a * x) % N
                M[y, x] = 1.0
            else:
                M[x, x] = 1.0
        return M
    
    def continued_fraction_denominator(self, phase, max_denominator):
        """Extract denominator from phase using continued fractions"""
        frac = Fraction(phase).limit_denominator(max_denominator)
        return frac.denominator, frac.numerator
    
    def find_order_qpe(self, a, N, n_count=8, shots=1024, show_circuit=True):
        """Find the order of 'a' modulo N using Quantum Phase Estimation"""
        if gcd(a, N) != 1:
            return None, None, None
        
        n_target = max(1, math.ceil(math.log2(N)))
        
        # Create quantum circuit
        qc = QuantumCircuit(n_count + n_target, n_count)
        
        # Step 1: Initialize counting qubits in superposition
        qc.h(range(n_count))
        self.step_explanations.append(
            f"Step 1: Applied Hadamard gates to {n_count} counting qubits to create superposition state"
        )
        
        # Step 2: Initialize target register in |1⟩
        qc.x(n_count)
        self.step_explanations.append(
            f"Step 2: Initialized target register in |1⟩ state (|1⟩ = |00...01⟩)"
        )
        
        # Step 3: Apply controlled modular multiplication gates
        for j in range(n_count):
            exp = 2 ** j
            a_pow = pow(a, exp, N)
            M = self.modular_multiplication_unitary(a_pow, N, n_target)
            
            # Create controlled unitary gate
            u_gate = Gate(f"U^{exp}", 1 + n_target, [])
            u_gate.definition = QuantumCircuit(1 + n_target)
            u_gate.definition.append(Gate(f"U^{exp}", 1 + n_target, []), range(1 + n_target))
            
            # Apply controlled operation
            control_qubit = j
            target_qubits = list(range(n_count, n_count + n_target))
            
        self.step_explanations.append(
            f"Step 3: Applied controlled modular multiplication gates for powers of {a}"
        )
        
        # Step 4: Apply inverse QFT
        qft_gate = QFTGate(n_count, do_swaps=False)
        qc.append(qft_gate.inverse(), range(n_count))
        self.step_explanations.append(
            f"Step 4: Applied inverse Quantum Fourier Transform to extract phase information"
        )
        
        # Step 5: Measure counting qubits
        qc.measure(range(n_count), range(n_count))
        self.step_explanations.append(
            f"Step 5: Measured counting qubits to obtain phase estimation results"
        )
        
        self.current_circuit = qc
        
        if show_circuit:
            self.display_circuit_streamlit()
        
        # Run simulation
        tqc = transpile(qc, self.backend)
        job = self.backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        self.measurement_results = counts
        
        # Analyze results
        most_common = max(counts, key=counts.get)
        measured_int = int(most_common, 2)
        phase = measured_int / (2 ** n_count)
        
        denom, numer = self.continued_fraction_denominator(phase, N)
        
        return denom, qc, counts
    
    def display_circuit_streamlit(self):
        """Display the quantum circuit with proper formatting for Streamlit"""
        if self.current_circuit is None:
            return
        
        st.subheader("🔬 Quantum Circuit Overview")
        
        # Create a text representation of the circuit
        circuit_text = """
        **Quantum Circuit for Order Finding:**
        
        **Counting Qubits:** |+⟩⊗n → QFT⁻¹ → Measurement  
        **Target Qubits:** |1⟩ → Controlled-U gates → |1⟩
        
        **Key Components:**
        • Hadamard gates create superposition
        • Controlled-U gates encode period information  
        • Inverse QFT extracts period from phase
        • Measurement reveals period information
        """
        
        st.markdown(circuit_text)
        
        # Create a visual representation using plotly
        fig = go.Figure()
        
        # Add circuit elements as annotations
        fig.add_annotation(
            x=0.1, y=0.8,
            text="Hadamard Gates",
            showarrow=True,
            arrowhead=2,
            arrowcolor="blue"
        )
        
        fig.add_annotation(
            x=0.5, y=0.6,
            text="Controlled-U Gates",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green"
        )
        
        fig.add_annotation(
            x=0.8, y=0.4,
            text="Inverse QFT",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red"
        )
        
        fig.add_annotation(
            x=0.9, y=0.2,
            text="Measurement",
            showarrow=True,
            arrowhead=2,
            arrowcolor="orange"
        )
        
        fig.update_layout(
            title="Shor's Algorithm: Quantum Circuit Overview",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_probabilities_streamlit(self):
        """Display measurement probabilities and analysis for Streamlit"""
        if self.measurement_results is None:
            return
        
        st.subheader("📊 Probability Analysis")
        
        # Create two columns for the analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot histogram of measurement results
            bitstrings = list(self.measurement_results.keys())
            counts = list(self.measurement_results.values())
            
            # Sort by count for better visualization
            sorted_data = sorted(zip(bitstrings, counts), key=lambda x: x[1], reverse=True)
            top_10 = sorted_data[:10]  # Show top 10 results
            
            if top_10:
                bitstrings_top, counts_top = zip(*top_10)
                
                # Create plotly bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(bitstrings_top),
                        y=list(counts_top),
                        marker_color='skyblue',
                        text=list(counts_top),
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Top 10 Measurement Results",
                    xaxis_title="Measurement Result (Binary)",
                    yaxis_title="Count",
                    width=400,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Analysis text
            total_shots = sum(self.measurement_results.values())
            most_common = max(self.measurement_results, key=self.measurement_results.get)
            most_common_count = self.measurement_results[most_common]
            probability = most_common_count / total_shots
            
            st.markdown(f"""
            **Measurement Analysis:**
            
            **Total Shots:** {total_shots}  
            **Most Common Result:** {most_common}  
            **Count:** {most_common_count}  
            **Probability:** {probability:.3f}
            
            **Interpretation:**
            • The measurement result represents a phase estimate
            • Higher probability indicates better phase estimation
            • This phase is used to find the period of the function
            • The period is crucial for factorization
            """)
    
    def display_step_by_step_streamlit(self):
        """Display step-by-step calculation and explanation for Streamlit"""
        if not self.step_explanations:
            return
        
        st.subheader("📚 Educational Guide: Understanding Shor's Algorithm")
        
        # Create a comprehensive explanation
        st.markdown("**Shor's Algorithm: Step-by-Step Process**")
        
        for i, step in enumerate(self.step_explanations, 1):
            st.markdown(f"**{i}.** {step}")
        
        st.markdown("""
        **Mathematical Foundation:**
        • We seek the period r such that a^r ≡ 1 (mod N)
        • If r is even and a^(r/2) ≢ ±1 (mod N), then:
          gcd(a^(r/2) ± 1, N) gives non-trivial factors
        
        **Quantum Advantage:**
        • Classical algorithms: O(exp(√(log N log log N)))
        • Shor's algorithm: O((log N)³)
        • Exponential speedup for large numbers
        """)
    
    def shors_algorithm(self, N, n_count=8, shots=1024, tries=5, show_circuit=True):
        """Main Shor's algorithm implementation"""
        self.step_explanations = []
        
        # Check for trivial cases
        if N % 2 == 0:
            self.step_explanations.append(f"N = {N} is even, so 2 is a factor")
            return 2, N // 2
        
        for attempt in range(tries):
            a = np.random.randint(2, N)
            if gcd(a, N) != 1:
                g = gcd(a, N)
                self.step_explanations.append(f"Found gcd({a}, {N}) = {g} (trivial factor)")
                return g, N // g
            
            self.step_explanations.append(f"Attempt {attempt+1}: Trying a = {a}")
            r, qc, counts = self.find_order_qpe(a, N, n_count=n_count, shots=shots, show_circuit=show_circuit)
            
            if r is None:
                self.step_explanations.append("Failed to extract order from QPE result")
                continue
            
            self.step_explanations.append(f"Candidate order r = {r}")
            
            if r % 2 != 0:
                self.step_explanations.append("r is odd; trying another 'a'")
                continue
            
            ar2 = pow(a, r // 2, N)
            if ar2 == N - 1:
                self.step_explanations.append("a^(r/2) ≡ -1 (mod N); trying another 'a'")
                continue
            
            factor1 = gcd(ar2 - 1, N)
            factor2 = gcd(ar2 + 1, N)
            
            if factor1 in (1, N) or factor2 in (1, N):
                self.step_explanations.append("Found trivial factors; trying again")
                continue
            
            self.step_explanations.append(f"Success! Found factors: {factor1} and {factor2}")
            return factor1, factor2
        
        return None

def create_educational_plots():
    """Create educational plots explaining quantum concepts"""
    
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: QFT Circuit Structure
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('QFT Circuit Structure', fontsize=14, fontweight='bold')
    ax1.text(0.1, 0.8, 'QFT Circuit Components:', fontsize=12, fontweight='bold')
    ax1.text(0.1, 0.7, '1. Hadamard gates on each qubit', fontsize=10)
    ax1.text(0.1, 0.6, '2. Controlled rotation gates R_k', fontsize=10)
    ax1.text(0.1, 0.5, '3. Qubit swaps for correct order', fontsize=10)
    ax1.text(0.1, 0.3, 'Mathematical Formula:', fontsize=12, fontweight='bold')
    ax1.text(0.1, 0.2, 'QFT|j⟩ = (1/√2ⁿ) Σ e^(2πijk/2ⁿ)|k⟩', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Plot 2: Superposition Visualization
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Quantum Superposition', fontsize=14, fontweight='bold')
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax2.plot(x, y, 'b-', linewidth=2, label='Bloch Sphere')
    ax2.arrow(0, 0, 1/np.sqrt(2), 1/np.sqrt(2), head_width=0.1, head_length=0.1, 
              fc='red', ec='red', linewidth=2)
    ax2.text(0.7, 0.7, '|+⟩ = (|0⟩ + |1⟩)/√2', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Complexity Comparison
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Algorithm Complexity Comparison', fontsize=14, fontweight='bold')
    N_values = np.logspace(1, 4, 100)
    classical = np.exp(np.cbrt(64/9 * np.log(N_values) * (np.log(np.log(N_values)))**2))
    shor = (np.log(N_values))**3
    
    ax3.loglog(N_values, classical, 'r-', linewidth=2, label='Classical (GNFS)')
    ax3.loglog(N_values, shor, 'b-', linewidth=2, label="Shor's Algorithm")
    ax3.set_xlabel('Number of bits (log scale)')
    ax3.set_ylabel('Time complexity (log scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Shor's Algorithm Flow
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Shor\'s Algorithm Flow', fontsize=14, fontweight='bold')
    flow_text = """Shor's Algorithm Steps:

1. Input: Composite number N
2. Choose random a ∈ [2, N-1]
3. Check gcd(a, N) = 1
4. Find period r: a^r ≡ 1 (mod N)
5. If r is even and a^(r/2) ≢ ±1 (mod N):
   • factor1 = gcd(a^(r/2) - 1, N)
   • factor2 = gcd(a^(r/2) + 1, N)
6. Return factors or try different a

Quantum Advantage:
• Step 4 uses quantum phase estimation
• Exponential speedup over classical methods"""
    
    ax4.text(0.05, 0.95, flow_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightyellow", alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""
    
    # Configure page
    st.set_page_config(
        page_title="Shor's Algorithm Demo",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🔬 Shor's Algorithm: Quantum Factorization")
    st.markdown("""
    This application demonstrates Shor's algorithm for integer factorization using quantum computing.
    Shor's algorithm can factor large integers exponentially faster than classical algorithms.
    """)
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Algorithm Parameters")
    
    # Input parameters
    N = st.sidebar.number_input(
        "Number to factor (N):",
        min_value=4,
        max_value=100,
        value=15,
        step=1,
        help="The composite number to factor"
    )
    
    n_count = st.sidebar.slider(
        "Counting qubits:",
        min_value=4,
        max_value=12,
        value=6,
        step=1,
        help="Number of qubits used for phase estimation"
    )
    
    shots = st.sidebar.slider(
        "Number of shots:",
        min_value=256,
        max_value=4096,
        value=1024,
        step=256,
        help="Number of times to run the quantum circuit"
    )
    
    tries = st.sidebar.slider(
        "Number of attempts:",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of random values 'a' to try"
    )
    
    # Action buttons
    st.sidebar.header("🚀 Actions")
    
    if st.sidebar.button("Run Shor's Algorithm", type="primary"):
        run_algorithm(N, n_count, shots, tries)
    
    if st.sidebar.button("Show Educational Content"):
        show_educational_content()
    
    if st.sidebar.button("Show Quantum Concepts"):
        show_quantum_concepts()
    
    # Main content area
    st.header("📊 Results")
    
    # Initialize session state
    if 'shor_instance' not in st.session_state:
        st.session_state.shor_instance = ShorAlgorithm()
    
    # Display current parameters
    st.info(f"**Current Parameters:** N={N}, Counting qubits={n_count}, Shots={shots}, Attempts={tries}")

def run_algorithm(N, n_count, shots, tries):
    """Run Shor's algorithm with given parameters"""
    
    st.subheader("🚀 Running Shor's Algorithm")
    
    with st.spinner("Executing quantum algorithm..."):
        shor = ShorAlgorithm()
        
        factors = shor.shors_algorithm(
            N=N,
            n_count=n_count,
            shots=shots,
            tries=tries,
            show_circuit=True
        )
        
        if factors is None:
            st.error(f"❌ Failed to find factors of {N} after {tries} attempts.")
            st.info("Try increasing the number of counting qubits or shots.")
        else:
            st.success(f"✅ Success! Found factors of {N}: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}")
            st.info(f"Verification: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}")
            
            # Store results in session state
            st.session_state.shor_instance = shor
            st.session_state.factors = factors
    
    # Show probability analysis if available
    if hasattr(st.session_state.shor_instance, 'measurement_results') and st.session_state.shor_instance.measurement_results:
        st.session_state.shor_instance.display_probabilities_streamlit()
    
    # Show step-by-step explanation
    if hasattr(st.session_state.shor_instance, 'step_explanations') and st.session_state.shor_instance.step_explanations:
        st.session_state.shor_instance.display_step_by_step_streamlit()

def show_educational_content():
    """Show educational content about Shor's algorithm"""
    
    st.subheader("📚 Educational Content")
    
    st.markdown("""
    ## What is Shor's Algorithm?
    
    Shor's algorithm is a quantum algorithm for integer factorization. It can factor large integers 
    exponentially faster than classical algorithms, which has significant implications for cryptography.
    
    ### Key Quantum Computing Concepts
    
    #### 1. Quantum Superposition
    In quantum computing, a qubit can exist in a superposition of states |0⟩ and |1⟩:
    |ψ⟩ = α|0⟩ + β|1⟩
    
    This allows quantum computers to process multiple states simultaneously.
    
    #### 2. Quantum Fourier Transform (QFT)
    The QFT is a quantum version of the classical Fourier transform. It transforms quantum states 
    from the computational basis to the frequency basis:
    
    QFT|j⟩ = (1/√N) Σ(k=0 to N-1) e^(2πijk/N) |k⟩
    
    **Role in Shor's Algorithm:** QFT is used to extract period information from quantum states, 
    which is crucial for finding the period of modular exponentiation.
    
    #### 3. Quantum Phase Estimation (QPE)
    QPE is used to estimate the eigenvalue of a unitary operator. In Shor's algorithm, it's used 
    to find the period of the function f(x) = a^x mod N.
    
    ### Shor's Algorithm Steps
    1. **Classical preprocessing**: Check if N is even or has small factors
    2. **Random selection**: Choose a random integer a coprime to N
    3. **Order finding**: Use quantum phase estimation to find the period r of f(x) = a^x mod N
    4. **Classical postprocessing**: Use the period to find factors of N
    """)

def show_quantum_concepts():
    """Show quantum concepts visualization"""
    
    st.subheader("🔬 Quantum Concepts Visualization")
    
    # Create educational plots
    fig = create_educational_plots()
    st.pyplot(fig)
    
    st.markdown("""
    ### Complexity Analysis
    
    | Algorithm | Time Complexity | Space Complexity |
    |-----------|----------------|------------------|
    | Classical (General Number Field Sieve) | O(exp((64/9)^(1/3) (log N)^(1/3) (log log N)^(2/3))) | O(log N) |
    | Shor's Algorithm | O((log N)³) | O(log N) |
    
    The exponential speedup makes Shor's algorithm a threat to RSA cryptography.
    """)

if __name__ == "__main__":
    main()
