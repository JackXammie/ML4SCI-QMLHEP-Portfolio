import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy
import cirq

from qgan_circuits import create_generator_circuit, qubits
from qgan_encode import train_circuits, test_circuits, y_train, y_test

# -------------------------------
# 1. Set up generator + discriminator
# -------------------------------

n_qubits = len(qubits)
gen_params_symbols = sympy.symbols(f'g0:{2*n_qubits}')  # 2 params per qubit

# Generator: parametric circuit (fake samples)
def generator_circuit(params):
    return create_generator_circuit(params)

# Discriminator: PQC layer + classical dense output
readout = tfq.layers.PQC(generator_circuit(gen_params_symbols), cirq.Z(qubits[0]))
discriminator_model = tf.keras.Sequential([
    readout,
    tf.keras.layers.Dense(1, activation='sigmoid')
])
discriminator_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 2. Training parameters
# -------------------------------

epochs = 30
batch_size = 10
num_batches = len(train_circuits) // batch_size

# -------------------------------
# 3. Training loop
# -------------------------------

for epoch in range(epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0

    # Shuffle data each epoch
    indices = np.random.permutation(len(train_circuits))
    train_circuits_shuffled = train_circuits.numpy()[indices]
    y_train_shuffled = y_train[indices]

    for i in range(num_batches):
        # -----------------------
        # Batch selection
        # -----------------------
        start = i * batch_size
        end   = start + batch_size
        real_circuits = tf.convert_to_tensor(train_circuits_shuffled[start:end])
        real_labels   = tf.convert_to_tensor(y_train_shuffled[start:end], dtype=tf.float32)

        # -----------------------
        # Generate fake circuits
        # -----------------------
        noise = np.random.uniform(0, np.pi, size=(batch_size, 2*n_qubits))  # 2 params per qubit
        fake_circuits = tfq.convert_to_tensor([generator_circuit(noise[j]) for j in range(batch_size)])
        fake_labels = tf.zeros(batch_size)

        # -----------------------
        # Train discriminator
        # -----------------------
        d_inputs = tf.concat([real_circuits, fake_circuits], axis=0)
        d_labels = tf.concat([real_labels, fake_labels], axis=0)
        d_loss, d_acc = discriminator_model.train_on_batch(d_inputs, d_labels)
        epoch_d_loss += d_loss

        # -----------------------
        # Train generator (flip labels to fool discriminator)
        # -----------------------
        g_inputs = fake_circuits
        g_labels = tf.ones(batch_size)
        g_loss, g_acc = discriminator_model.train_on_batch(g_inputs, g_labels)
        epoch_g_loss += g_loss

    # -----------------------
    # Evaluate AUC on test set
    # -----------------------
    y_pred = discriminator_model.predict(test_circuits)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Epoch {epoch+1}/{epochs} | D_loss: {epoch_d_loss/num_batches:.4f} | "
          f"G_loss: {epoch_g_loss/num_batches:.4f} | Test AUC: {auc:.4f}")
