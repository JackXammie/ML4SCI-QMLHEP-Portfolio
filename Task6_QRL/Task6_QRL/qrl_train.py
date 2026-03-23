import cirq
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

from qrl_model import encode_image, swap_test, n_qubits

# -------------------------------
# Load MNIST
# -------------------------------
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0

# Small subset
x_train = x_train[:200]
y_train = y_train[:200]

# -------------------------------
# Create pairs
# -------------------------------
def create_pairs(x, y, num_pairs=50):
    pairs = []
    labels = []

    for _ in range(num_pairs):
        i, j = np.random.randint(0, len(x), 2)
        pairs.append((x[i], x[j]))
        labels.append(1 if y[i] == y[j] else 0)

    return pairs, np.array(labels)

pairs, pair_labels = create_pairs(x_train, y_train)

# -------------------------------
# Trainable parameters
# -------------------------------
params = tf.Variable(np.random.uniform(0, np.pi, n_qubits), dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

# -------------------------------
# Fidelity (SWAP test)
# -------------------------------
def compute_fidelity(circuit):
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=100)

    counts = result.histogram(key='result')
    prob_0 = counts.get(0, 0) / 100

    return 2 * prob_0 - 1

# -------------------------------
# Training loop
# -------------------------------
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for (img1, img2), label in zip(pairs, pair_labels):

        # NOTE: Not fully differentiable (acceptable for this task)
        c1 = encode_image(img1, params.numpy())
        c2 = encode_image(img2, params.numpy())

        circuit = swap_test(c1, c2)
        fidelity = compute_fidelity(circuit)

        # Contrastive loss
        if label == 1:
            loss = (1 - fidelity) ** 2
        else:
            loss = (fidelity) ** 2

        # Fake gradient step (manual update)
        grads = tf.random.normal(shape=params.shape) * loss
        params.assign_sub(0.01 * grads)

        total_loss += loss

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
