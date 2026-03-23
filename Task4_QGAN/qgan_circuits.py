import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

# 4 qubits
n_qubits = 4
qubits = cirq.GridQubit.rect(1, n_qubits)

# Encoding circuit: classical features -> qubits
def create_encoding_circuit(features):
    circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(features[i])(q))
    return circuit

# Generator circuit: parametric rotations + entanglement
def create_generator_circuit(params):
    circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(params[i])(q))
        circuit.append(cirq.rz(params[i + n_qubits])(q))
    # Entanglement
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit

# Readout
readout = cirq.Z(qubits[0])

# Example: PQC discriminator model
params = sympy.symbols('x0:8')
discriminator_model = tf.keras.Sequential([
    tfq.layers.PQC(create_generator_circuit(params), readout),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
discriminator_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
