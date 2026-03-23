import cirq
import numpy as np

# -------------------------------
# Qubit setup
# -------------------------------
qubits = cirq.GridQubit.rect(1, 2)


# -------------------------------
# Dataset (Z2 x Z2 symmetry)
# -------------------------------
def generate_data(n=100):
    X = []
    y = []

    for _ in range(n):
        x1, x2 = np.random.uniform(-1, 1, 2)

        # Label rule
        label = 1 if x1 * x2 > 0 else 0

        # Apply symmetry transformations
        X.extend([
            [ x1,  x2],
            [-x1,  x2],
            [ x1, -x2],
            [-x1, -x2]
        ])

        y.extend([label] * 4)

    return np.array(X), np.array(y)


# -------------------------------
# Normal QNN (no symmetry)
# -------------------------------
def normal_qnn(x, params):
    circuit = cirq.Circuit()

    # Encode features
    circuit.append(cirq.ry(x[0] * np.pi)(qubits[0]))
    circuit.append(cirq.ry(x[1] * np.pi)(qubits[1]))

    # Independent parameters
    circuit.append(cirq.rz(params[0])(qubits[0]))
    circuit.append(cirq.rz(params[1])(qubits[1]))

    # Entanglement
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))

    return circuit


# -------------------------------
# Equivariant QNN (Z2 x Z2)
# -------------------------------
def equivariant_qnn(x, theta):
    circuit = cirq.Circuit()

    # Encode features
    circuit.append(cirq.ry(x[0] * np.pi)(qubits[0]))
    circuit.append(cirq.ry(x[1] * np.pi)(qubits[1]))

    # Shared parameter (KEY IDEA)
    circuit.append(cirq.rz(theta)(qubits[0]))
    circuit.append(cirq.rz(theta)(qubits[1]))

    # Symmetric entanglement
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))

    return circuit


# -------------------------------
# Measurement helper
# -------------------------------
def measure_expectation(circuit):
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    # Measure expectation of Z on first qubit
    state = result.final_state_vector
    expectation = np.real(np.vdot(state, cirq.Z(qubits[0])._unitary_() @ state))

    return expectation
