# Quantum Graph Neural Network (QGNN) – Task 5

This repository implements a Quantum Graph Neural Network (QGNN) to leverage graph-structured data in a quantum computing context. The circuit encodes graph features into qubits and applies entangling gates to capture relationships between nodes.

---

## Structure

- **qgnn_circuit.py** — constructs and visualizes the QGNN quantum circuit  
- **Task5_Report.md** — detailed description of the approach, circuit design, and results  

---

## Description

A QGNN circuit takes advantage of graph representations by:

1. Encoding node features into qubit rotations (Ry, Rz).  
2. Applying controlled gates (e.g., CNOT) along edges to capture graph connectivity.  
3. Measuring qubits to extract graph-informed quantum features.  

The circuit can be expanded with more qubits or layers to represent larger graphs.  

---

## Requirements

- Python 3.10+  
- Cirq  
- TensorFlow Quantum (optional for hybrid experiments)  
- NumPy

---

## Usage

Run:

```bash
python qgnn_circuit.py
to build and visualize the QGNN circuit.
