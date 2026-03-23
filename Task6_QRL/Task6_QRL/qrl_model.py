import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from tensorflow.keras.datasets import mnist

# -------------------------------
# Load MNIST
# -------------------------------
(x_train, y_train), _ = mnist.load_data()

# Normalize
x_train = x_train / 255.0

# Use small subset (keep it light)
x_train = x_train[:200]
y_train = y_train[:200]

# -------------------------------
# Qubit setup
# -------------------------------
n_qubits = 4

# Base qubits (used for encoding template)
base_qubits = cirq.GridQubit.rect(1, n_qubits)

# -------------------------------
# Encoding function
# -------------------------------
def encode_image(image, params):
    circuit = cirq.Circuit()

    flat = image.flatten()[:n_qubits]

    for i, q in enumerate(base_qubits):
        circuit.append(cirq.ry(flat[i] * np.pi)(q))
        circuit.append(cirq.rz(params[i])(q))

    return circuit

# -------------------------------
# SWAP Test (FIXED VERSION)
# -------------------------------
def swap_test(circuit1, circuit2):
    ancilla = cirq.GridQubit(-1, 0)

    # Create two separate registers
    qubits_a = cirq.GridQubit.rect(1, n_qubits, top=0)
    qubits_b = cirq.GridQubit.rect(1, n_qubits, top=1)

    circuit = cirq.Circuit()

    # Hadamard on ancilla
    circuit.append(cirq.H(ancilla))

    # Map circuit1 → qubits_a
    mapping_a = {base_qubits[i]: qubits_a[i] for i in range(n_qubits)}
    circuit += circuit1.transform_qubits(mapping_a)

    # Map circuit2 → qubits_b
    mapping_b = {base_qubits[i]: qubits_b[i] for i in range(n_qubits)}
    circuit += circuit2.transform_qubits(mapping_b)

    # Controlled SWAPs
    for i in range(n_qubits):
        circuit.append(cirq.CSWAP(ancilla, qubits_a[i], qubits_b[i]))

    # Final Hadamard
    circuit.append(cirq.H(ancilla))

    # Measure ancilla
    circuit.append(cirq.measure(ancilla, key='result'))

    return circuit

# -------------------------------
# Example run
# -------------------------------
params = np.random.uniform(0, np.pi, n_qubits)

c1 = encode_image(x_train[0], params)
c2 = encode_image(x_train[1], params)

final_circuit = swap_test(c1, c2)

print("\nSWAP Test Circuit:\n")
print(final_circuit)
