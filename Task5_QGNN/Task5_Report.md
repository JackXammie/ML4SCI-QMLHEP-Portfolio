# Task 5 Report: Quantum Graph Neural Network (QGNN)

## 1. Introduction  
A Quantum Graph Neural Network (QGNN) extends classical Graph Neural Networks (GNNs) by using quantum circuits to process graph-structured data. This approach leverages quantum properties such as superposition and entanglement to model relationships between nodes in a graph.

---

## 2. QGNN Circuit Design  
The QGNN circuit is constructed by mapping graph components into quantum elements:

- Nodes are represented as qubits  
- Edges are represented as entangling operations (e.g., CNOT gates)  
- Node features are encoded using parameterized rotation gates (Ry, Rz)  

Each node’s features are encoded into a qubit using rotation gates, while edges define how qubits interact through entanglement. This allows information to propagate across the graph.

---

## 3. Implementation  
The QGNN circuit was implemented using Cirq. The process includes:

- Encoding node features into qubits  
- Applying entanglement based on graph connectivity  
- Adding additional rotation layers to simulate message passing  

This structure mirrors the behavior of classical GNNs, where node representations are updated using information from neighboring nodes.

---

## 4. Circuit Visualization  
The constructed circuit was visualized using Cirq’s circuit printing functionality. The diagram shows:

- Rotation gates for feature encoding  
- CNOT gates representing graph edges  
- Additional rotations representing feature updates  

This provides a clear view of how the graph structure is translated into a quantum circuit.

---

## 5. Discussion  
The QGNN demonstrates how quantum circuits can naturally model graph data through entanglement. Unlike classical GNNs that explicitly aggregate neighbor information, quantum entanglement allows implicit information sharing between connected nodes.

The design can be extended by stacking multiple layers of rotations and entanglement to improve model expressiveness.

---

## 6. Conclusion  
A QGNN circuit was successfully designed and implemented. The mapping of nodes to qubits and edges to entangling gates enables quantum systems to process relational data effectively. This highlights the potential of quantum machine learning for graph-based problems.
