# Shor's Algorithm - Streamlit Web Application

A comprehensive implementation of Shor's algorithm with an interactive web interface built using Streamlit.

## Features

- 🔬 **Interactive Web Interface**: Beautiful, responsive web application
- 📊 **Real-time Visualization**: Quantum circuit diagrams and probability analysis
- 📚 **Educational Content**: Step-by-step explanations and quantum concepts
- 🎛️ **Parameter Control**: Adjust algorithm parameters in real-time
- 📈 **Probability Analysis**: Interactive charts showing measurement results
- 🔄 **Live Updates**: Real-time results and visualizations

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run shor_streamlit.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

### Main Interface

1. **Set Parameters** in the sidebar:
   - **Number to factor (N)**: The composite number to factor (4-100)
   - **Counting qubits**: Number of qubits for phase estimation (4-12)
   - **Number of shots**: Quantum circuit executions (256-4096)
   - **Number of attempts**: Random values to try (1-10)

2. **Run the Algorithm**:
   - Click "Run Shor's Algorithm" to execute the factorization
   - View real-time results and visualizations
   - See step-by-step explanations

3. **Explore Educational Content**:
   - Click "Show Educational Content" for detailed explanations
   - Click "Show Quantum Concepts" for visualizations

### Features Overview

#### 🚀 Algorithm Execution
- Real-time factorization with progress indicators
- Success/failure feedback with detailed explanations
- Parameter validation and error handling

#### 📊 Visualizations
- **Quantum Circuit Overview**: Interactive circuit diagrams
- **Probability Analysis**: Bar charts of measurement results
- **Educational Plots**: QFT structure, superposition, complexity comparison

#### 📚 Educational Content
- **Step-by-step Process**: Detailed algorithm walkthrough
- **Quantum Concepts**: QFT, superposition, entanglement explanations
- **Mathematical Foundation**: Formulas and complexity analysis
- **Interactive Examples**: Try different numbers and parameters

## Example Usage

### Basic Factorization
1. Set N = 15 in the sidebar
2. Click "Run Shor's Algorithm"
3. View the results: 3 × 5 = 15

### Educational Exploration
1. Click "Show Educational Content"
2. Learn about quantum concepts
3. Click "Show Quantum Concepts" for visualizations
4. Try different parameters to see how they affect results

## Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Qiskit**: Quantum computing library
- **Qiskit-Aer**: Quantum simulator
- **Matplotlib**: Plotting library
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computations

### Architecture
- **ShorAlgorithm Class**: Core algorithm implementation
- **Streamlit Interface**: Web UI and user interactions
- **Visualization Components**: Charts and educational content
- **Session State**: Maintains state between interactions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**: Use a different port
   ```bash
   streamlit run shor_streamlit.py --server.port 8502
   ```

3. **Memory Issues**: Reduce the number of shots or counting qubits for large numbers

### Performance Tips

- **Small Numbers**: Use N < 50 for faster execution
- **Fewer Shots**: Reduce shots for quicker results (but less accurate)
- **Fewer Qubits**: Use 4-6 counting qubits for small numbers

## Educational Value

This application is designed for:
- **Students**: Learning quantum computing concepts
- **Researchers**: Understanding Shor's algorithm implementation
- **Educators**: Teaching quantum algorithms
- **Enthusiasts**: Exploring quantum computing

## Future Enhancements

- [ ] Real quantum hardware integration
- [ ] More quantum algorithms
- [ ] Advanced visualizations
- [ ] Export functionality
- [ ] Batch processing

## Contributing

Feel free to contribute improvements, bug fixes, or new features!

## License

This project is open source and available under the MIT License.
