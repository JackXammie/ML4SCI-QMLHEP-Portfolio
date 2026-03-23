# task1_second_circuit_swap_test.py
import cirq
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 1. Define 4 qubits for the two 2-qubit states + 1 ancilla
ancilla, q1, q2, q3, q4 = cirq.LineQubit.range(5)
circuit = cirq.Circuit()

# 2. Prepare states
circuit.append(cirq.H(q1))                 # Hadamard on first qubit
circuit.append(cirq.rx(np.pi/3)(q2))      # X rotation on second qubit
circuit.append(cirq.H(q3))                 # Hadamard on third qubit
circuit.append(cirq.H(q4))                 # Hadamard on fourth qubit

# 3. SWAP Test
circuit.append(cirq.H(ancilla))
circuit.append(cirq.CSWAP(ancilla, q1, q3))  # SWAP first qubit of each pair
circuit.append(cirq.CSWAP(ancilla, q2, q4))  # SWAP second qubit of each pair
circuit.append(cirq.H(ancilla))
circuit.append(cirq.measure(ancilla, key='m'))

# 4. Display the circuit
print("Task I: Second circuit with 2-qubit SWAP test")
print(circuit)

# 5. Simulate SWAP test
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)
counts = Counter(result.measurements['m'].flatten())
print("\nSWAP test measurement counts (ancilla):")
print(counts)

# Optional: plot histogram
plt.bar(counts.keys(), counts.values())
plt.xlabel('Ancilla measurement')
plt.ylabel('Counts')
plt.title('SWAP Test Measurement Histogram')
plt.show()
