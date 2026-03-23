# Equivariant Quantum Neural Networks (EQNN) – Task 7

This repository implements **Equivariant Quantum Neural Networks (EQNNs)** for a simple binary classification problem with **Z•2 × Z•2 symmetry**.

## Overview

- A classical dataset with two features (X₁ and X₂) was generated, respecting **Z•2 × Z•2 symmetry**.  
- Both a **normal QNN** and a **symmetry-equivariant QNN** were implemented and trained.  
- The project demonstrates how incorporating symmetry can affect quantum model performance.  

## Features

- Dataset generation with symmetry.  
- QNN and equivariant QNN implementation.  
- Training and evaluation scripts with accuracy comparison.  
- Circuit rotations expressed in π multiples.  

## Requirements

- Python 3.10+  
- TensorFlow Quantum  
- Cirq  
- NumPy, Matplotlib  

## Usage

```bash
# Install dependencies
pip install tensorflow tensorflow-quantum cirq numpy matplotlib

# Train and evaluate
python eqnn_train.py
