import numpy as np
import tensorflow as tf
import cirq
import sympy
import tensorflow_quantum as tfq
from sklearn.model_selection import train_test_split

# ---------------------------
# Reproducibility (IMPORTANT)
# ---------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 1. Generate symmetric dataset
# ---------------------------
def generate_data(n=500):
    X = np.random.uniform(-1, 1, (n, 2))

    # Easier symmetry-aware rule
    y = (X[:, 0] > X[:, 1]).astype(int)

    return X, y


X, y = generate_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale to [0, pi]
X_train = (X_train + 1) * np.pi / 2
X_test = (X_test + 1) * np.pi / 2

# ---------------------------
# 2. Qubits
# ---------------------------
qubits = cirq.GridQubit.rect(1, 2)

# ---------------------------
# 3. Encoding circuit (STRONGER)
# ---------------------------
def encode(x):
    circuit = cirq.Circuit()

    for i in range(2):
        circuit.append(cirq.ry(x[i])(qubits[i]))
        circuit.append(cirq.rz(x[i])(qubits[i]))  # extra expressiveness

    return circuit

# ---------------------------
# 4. Standard QNN (deeper)
# ---------------------------
def normal_qnn():
    params = sympy.symbols('w0:8')
    circuit = cirq.Circuit()

    # Layer 1
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(params[i])(q))
        circuit.append(cirq.rz(params[i+2])(q))

    circuit.append(cirq.CNOT(qubits[0], qubits[1]))

    # Layer 2
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(params[i+4])(q))
        circuit.append(cirq.rz(params[i+6])(q))

    return circuit, params

# ---------------------------
# 5. Equivariant QNN (shared weights)
# ---------------------------
def equivariant_qnn():
    params = sympy.symbols('w0:4')
    circuit = cirq.Circuit()

    # Layer 1 (shared)
    for q in qubits:
        circuit.append(cirq.ry(params[0])(q))
        circuit.append(cirq.rz(params[1])(q))

    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[0]))

    # Layer 2 (shared)
    for q in qubits:
        circuit.append(cirq.ry(params[2])(q))
        circuit.append(cirq.rz(params[3])(q))

    return circuit, params

# ---------------------------
# 6. Prepare quantum data
# ---------------------------
train_circuits = tfq.convert_to_tensor([encode(x) for x in X_train])
test_circuits = tfq.convert_to_tensor([encode(x) for x in X_test])

# ---------------------------
# 7. Build model
# ---------------------------
def build_model(qnn_func):
    circuit, params = qnn_func()

    # MULTI-QUBIT READOUT (important fix)
    readout = [cirq.Z(q) for q in qubits]

    model = tf.keras.Sequential([
        tfq.layers.PQC(circuit, readout),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------------------
# 8. Build both models
# ---------------------------
normal_model = build_model(normal_qnn)
equivariant_model = build_model(equivariant_qnn)

# ---------------------------
# 9. Train models
# ---------------------------
print("Training Normal QNN...")
normal_model.fit(train_circuits, y_train, epochs=50, verbose=0)

print("Training Equivariant QNN...")
equivariant_model.fit(train_circuits, y_train, epochs=50, verbose=0)

# ---------------------------
# 10. Evaluate
# ---------------------------
normal_acc = normal_model.evaluate(test_circuits, y_test, verbose=0)[1]
equiv_acc = equivariant_model.evaluate(test_circuits, y_test, verbose=0)[1]

print(f"Normal QNN Accuracy: {normal_acc:.4f}")
print(f"Equivariant QNN Accuracy: {equiv_acc:.4f}")
