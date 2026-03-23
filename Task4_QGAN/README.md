# Quantum Generative Adversarial Network (QGAN) – Task 4

This project implements a **Quantum Generative Adversarial Network (QGAN)** using **Cirq** and **TensorFlow Quantum (TFQ)** to classify high-energy physics data by separating **signal events** from **background events**.

---

## Overview

QGANs combine quantum circuits with classical machine learning to model complex data distributions. In this task, quantum circuits are used to encode classical data and train a generator–discriminator system in an adversarial setting.

---

## Dataset

- 100 training samples  
- 100 test samples  
- Labels:
  - `1` → Signal  
  - `0` → Background  

---

## Project Structure
---

## Methodology

1. **Data Preprocessing**
   - Feature scaling and dimensionality reduction
   - Conversion to quantum-compatible format

2. **Quantum Encoding**
   - Classical features encoded into qubits using rotation gates

3. **QGAN Architecture**
   - Generator produces quantum states from noise
   - Discriminator classifies real vs fake circuits

4. **Training**
   - Adversarial training between generator and discriminator
   - Performance evaluated using **AUC (Area Under ROC Curve)**

---

## Results

- Final Test AUC: **~0.62**
- Model shows improved separation of signal vs background over training

---

## Key Insights

- Hybrid quantum-classical models are effective for small datasets  
- Circuit design and parameter tuning significantly affect performance  
- Noise and limited data impact stability, requiring careful fine-tuning  

---

## Future Improvements

- Increase number of qubits and circuit depth  
- Train on larger datasets  
- Explore real quantum hardware execution  
- Apply QGANs to other domains such as agriculture and security  

---

## Technologies Used

- Python  
- TensorFlow Quantum (TFQ)  
- Cirq  
- NumPy, Scikit-learn  

---

## Author

Samuel Eke
