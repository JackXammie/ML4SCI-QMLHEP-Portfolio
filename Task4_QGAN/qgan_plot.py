import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy
import cirq

from qgan_circuits import create_generator_circuit, qubits
from qgan_encode import train_circuits, test_circuits, y_train, y_test

n_qubits = len(qubits)
gen_params_symbols = sympy.symbols(f'g0:{2*n_qubits}')

def generator_circuit(params):
    return create_generator_circuit(params)

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

epochs = 30
batch_size = 10
num_batches = len(train_circuits) // batch_size

d_losses, g_losses, aucs = [], [], []

for epoch in range(epochs):
    epoch_d_loss, epoch_g_loss = 0, 0

    indices = np.random.permutation(len(train_circuits))
    train_circuits_shuffled = train_circuits.numpy()[indices]
    y_train_shuffled = y_train[indices]

    for i in range(num_batches):
        start, end = i*batch_size, (i+1)*batch_size
        real_circuits = tf.convert_to_tensor(train_circuits_shuffled[start:end])
        real_labels = tf.convert_to_tensor(y_train_shuffled[start:end], dtype=tf.float32)

        # Generate fake circuits
        noise = np.random.uniform(0, np.pi, size=(batch_size, 2*n_qubits))
        fake_circuits = tfq.convert_to_tensor([generator_circuit(noise[j]) for j in range(batch_size)])
        fake_labels = tf.zeros(batch_size)

        # Train discriminator
        d_inputs = tf.concat([real_circuits, fake_circuits], axis=0)
        d_labels = tf.concat([real_labels, fake_labels], axis=0)
        d_loss, _ = discriminator_model.train_on_batch(d_inputs, d_labels)
        epoch_d_loss += d_loss

        # Train generator (flip labels)
        g_inputs = fake_circuits
        g_labels = tf.ones(batch_size)
        g_loss, _ = discriminator_model.train_on_batch(g_inputs, g_labels)
        epoch_g_loss += g_loss

    # Evaluate on test set
    y_pred = discriminator_model.predict(test_circuits)
    auc = roc_auc_score(y_test, y_pred)

    d_losses.append(epoch_d_loss/num_batches)
    g_losses.append(epoch_g_loss/num_batches)
    aucs.append(auc)

    print(f"Epoch {epoch+1}/{epochs} | D_loss: {d_losses[-1]:.4f} | "
          f"G_loss: {g_losses[-1]:.4f} | Test AUC: {auc:.4f}")

# -------------------------------
# Plot results
# -------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('QGAN Loss per Epoch')

plt.subplot(1,2,2)
plt.plot(aucs, color='green', label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.ylim(0,1)
plt.title('QGAN Test AUC per Epoch')
plt.legend()
plt.show()
