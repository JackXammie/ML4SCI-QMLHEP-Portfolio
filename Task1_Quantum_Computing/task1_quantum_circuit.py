# task1_quantum_circuit.py
import cirq
import numpy as np
import matplotlib.pyplot as plt

# 1. Create 5 qubits
qubits = [cirq.LineQubit(i) for i in range(5)]

# 2. Build the circuit
circuit = cirq.Circuit()

# a) Apply Hadamard to all qubits
for q in qubits:
    circuit.append(cirq.H(q))

# b) Apply CNOT operations
cnot_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
for control, target in cnot_pairs:
    circuit.append(cirq.CNOT(qubits[control], qubits[target]))

# c) Apply SWAP between qubits 0 and 4
circuit.append(cirq.SWAP(qubits[0], qubits[4]))

# d) Apply X rotation by pi/2 on qubit 2
circuit.append(cirq.rx(np.pi / 2)(qubits[2]))

# 3. Display the circuit
print("Task I: 5-qubit quantum circuit")
print(circuit)

# Plot the circuit using matplotlib
fig, ax = plt.subplots(figsize=(8, 4))
circuit_diagram = circuit.to_text_diagram(transpose=False)
ax.text(0.05, 0.95, circuit_diagram, family='monospace', fontsize=10, va='top')
ax.axis('off')
plt.title("Task I: 5-qubit Circuit")
plt.show()

# 4. Simulate the circuit
simulator = cirq.Simulator()
result = simulator.simulate(circuit)

print("\nFinal state vector:")
print(result.final_state_vector)
