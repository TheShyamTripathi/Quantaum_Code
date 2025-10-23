# 🧮 Shor’s Algorithm — Quantum Factorization Demo

**Author:** [Shyam Tripathi](https://github.com/TheShyamTripathi)
**Repository:** [Quantum_Code](https://github.com/TheShyamTripathi/Quantaum_Code)
**Deployed App:** [Streamlit Demo](https://shoralgorithmfactor.streamlit.app/)

---

## 🌟 Overview

This project is a **complete implementation of Shor’s Algorithm** — the groundbreaking **quantum algorithm for integer factorization**.
It includes a full-featured **interactive Streamlit GUI** with **quantum circuit visualization**, **probability analysis**, and **step-by-step educational explanations** of the underlying quantum principles.

Shor’s Algorithm is capable of factoring large integers **exponentially faster** than any known classical algorithm, which has profound implications for cryptography (e.g., RSA).

---

## 🚀 Features

### 🧠 Quantum Algorithm Features

* Full implementation of **Shor’s Algorithm**
* Modular multiplication and **order finding** via **Quantum Phase Estimation (QPE)**
* Integration of **Quantum Fourier Transform (QFT)**
* Automatic **factor discovery** for composite numbers

### 🎨 Streamlit Web Application

* Interactive input controls for:

  * Composite number `N`
  * Counting qubits
  * Number of shots
  * Random attempts (`a` values)
* **Quantum circuit visualization** using Qiskit
* **Measurement probability plots**
* **Circuit analysis dashboard** with metrics and gate distributions

### 📚 Educational Features

* Step-by-step guide to each part of Shor’s algorithm
* Mathematical foundations:

  * Modular arithmetic
  * Quantum Fourier Transform (QFT)
  * Quantum Phase Estimation (QPE)
* Visual explanations of:

  * Quantum superposition
  * QFT operations
  * Measurement probabilities
  * Algorithmic flow and complexity comparison
* Multiple visualization modes (IQX, Clifford, Text, Default)
* Export circuit as **QASM**, **JSON**, or text

---

## 🧩 Tech Stack

| Component            | Technology                 |
| -------------------- | -------------------------- |
| Frontend GUI         | **Streamlit**              |
| Quantum Framework    | **Qiskit (IBM Quantum)**   |
| Visualization        | **Matplotlib**, **Plotly** |
| Programming Language | **Python 3.10+**           |

---

## ⚙️ Installation

### Prerequisites

Make sure you have **Python 3.10+** and **pip** installed.

### Steps

```bash
# Clone this repository
git clone https://github.com/TheShyamTripathi/Quantaum_Code.git
cd Quantaum_Code

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Then open your browser at:
👉 `http://localhost:8501/`

---

## 🧪 Usage

### Run Shor’s Algorithm

1. Enter a composite number `N` in the sidebar (e.g., 15).
2. Choose the number of **counting qubits**, **shots**, and **tries**.
3. Click **Run Shor’s Algorithm**.
4. View:

   * Factorization results
   * Quantum circuit visualization
   * Measurement probability histogram
   * Step-by-step algorithm explanation

### Explore Educational Modules

You can also click:

* **Show Educational Content** → To learn the algorithm basics
* **Show Quantum Concepts** → To visualize QFT and superposition
* **Show Mathematical Foundation** → To explore core quantum math
* **Show Circuit Details / Styles** → To view multiple circuit layouts and exports

---

## 🧮 Algorithm Workflow

1. **Input:** Composite number ( N )
2. **Choose:** Random integer ( a ) such that ( \text{gcd}(a, N) = 1 )
3. **Quantum Phase Estimation:** Find period ( r ) such that ( a^r \equiv 1 \pmod{N} )
4. **Classical Post-Processing:**

   * If ( r ) is even and ( a^{r/2} \neq \pm 1 \pmod{N} ):
     [
     \text{factors} = \gcd(a^{r/2} \pm 1, N)
     ]
5. **Output:** Non-trivial factors of ( N )

---

## 📊 Complexity Comparison

| Algorithm        | Time Complexity                                                                  | Space Complexity |
| ---------------- | -------------------------------------------------------------------------------- | ---------------- |
| Classical (GNFS) | ( O\left(\exp\left((64/9)^{1/3}(\log N)^{1/3}(\log \log N)^{2/3}\right)\right) ) | ( O(\log N) )    |
| Shor’s Algorithm | ( O((\log N)^3) )                                                                | ( O(\log N) )    |

Shor’s algorithm achieves **exponential speedup**, making it a potential **threat to RSA encryption** once large-scale quantum computers become practical.

---

## 🧾 Example

| Input  | Output            | Factors |
| ------ | ----------------- | ------- |
| N = 15 | Found order r = 4 | (3, 5)  |
| N = 21 | Found order r = 6 | (3, 7)  |

---

## 🧑‍🏫 Educational Visuals

* **Quantum Superposition:** Demonstrated using Bloch sphere representation
* **QFT Circuit:** Shows rotations, Hadamard gates, and swaps
* **Probability Histogram:** Displays most common measured bitstrings
* **Algorithm Flow:** Classical + Quantum hybrid process diagram

---

## 🧰 Files

```
📁 Quantaum_Code/
├── app.py                # Streamlit main application
├── shor_algorithm.py     # Core Shor’s algorithm class
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── assets/               # Visual and circuit outputs
```

---

## 🧑‍💻 Author

**👤 Shyam Tripathi**
🎓 Computer Science and Engineering (CSE) Student
🔗 [GitHub Profile](https://github.com/TheShyamTripathi)

---

## 📜 License

This project is licensed under the **APACHE 2.0 LICENSE** — feel free to use and modify it for educational and research purposes.

