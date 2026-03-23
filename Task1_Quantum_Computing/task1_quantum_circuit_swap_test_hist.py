# task1_quantum_circuit_swap_test_hist.py
import cirq
import matplotlib.pyplot as plt

# -----------------------------
# Part 2: SWAP Test
# -----------------------------
# Create 3 qubits: ancilla + 2 target qubits
ancilla, qA, qB = cirq.LineQubit.range(3)
swap_circuit = cirq.Circuit()

# Prepare some states for testing
swap_circuit.append(cirq.H(qA))  # |ψ> = H|0>
swap_circuit.append(cirq.X(qB))  # |φ> = X|0>

# SWAP Test
swap_circuit.append(cirq.H(ancilla))
swap_circuit.append(cirq.CSWAP(ancilla, qA, qB))
swap_circuit.append(cirq.H(ancilla))
swap_circuit.append(cirq.measure(ancilla, key='m'))

print("Part 2: SWAP Test circuit")
print(swap_circuit)

# Simulate SWAP test
simulator = cirq.Simulator()
swap_result = simulator.run(swap_circuit, repetitions=100)

# Count results
ancilla_counts = swap_result.histogram(key='m')
print("\nSWAP test measurement counts (ancilla):")
print(ancilla_counts)

# Plot histogram
plt.bar(ancilla_counts.keys(), ancilla_counts.values(), color='skyblue')
plt.xticks([0, 1])
plt.xlabel("Ancilla measurement outcome")
plt.ylabel("Counts")
plt.title("SWAP Test Results (Ancilla Qubit)")
plt.show()
