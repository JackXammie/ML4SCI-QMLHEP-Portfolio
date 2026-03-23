import cirq
import numpy as np

# Define a simple graph (edges between nodes)
graph_edges = [(0, 1), (0, 2), (1, 3)]

# Number of nodes
n_nodes = 4

# Create qubits (one per node)
qubits = cirq.GridQubit.rect(1, n_nodes)

def create_qgnn_circuit(features):
    circuit = cirq.Circuit()

    # Encode node features
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(features[i])(q))

    # Apply entanglement based on graph structure
    for (i, j) in graph_edges:
        circuit.append(cirq.CNOT(qubits[i], qubits[j]))

    # Message passing layer (extra rotations)
    for i, q in enumerate(qubits):
        circuit.append(cirq.rz(features[i])(q))

    return circuit


# Example feature vector
features = np.random.uniform(0, np.pi, n_nodes)

# Create circuit
circuit = create_qgnn_circuit(features)

# Print circuit
print("QGNN Circuit:\n")
print(circuit)
