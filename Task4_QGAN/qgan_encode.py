import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from qgan_circuits import create_encoding_circuit
import tensorflow_quantum as tfq

# --- Load dataset ---
data = np.load("data/dataset.npz", allow_pickle=True)
training_input = data["training_input"].item()
test_input = data["test_input"].item()

signal_train = training_input['1']
background_train = training_input['0']
signal_test = test_input['1']
background_test = test_input['0']

# --- PCA ---
pca = PCA(n_components=4)
X_train = pca.fit_transform(np.vstack([signal_train, background_train]))
X_test  = pca.transform(np.vstack([signal_test, background_test]))

# --- Labels (optional, just for reference) ---
y_train = np.hstack([np.ones(len(signal_train)), np.zeros(len(background_train))])
y_test  = np.hstack([np.ones(len(signal_test)),  np.zeros(len(background_test))])

# --- Normalize for quantum encoding ---
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Shuffle ---
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test,  y_test  = shuffle(X_test,  y_test,  random_state=42)

# --- Encode as circuits ---
train_circuits = [create_encoding_circuit(x) for x in X_train]
test_circuits  = [create_encoding_circuit(x) for x in X_test]

# --- Convert to tfq tensors ---
train_circuits = tfq.convert_to_tensor(train_circuits)
test_circuits  = tfq.convert_to_tensor(test_circuits)

print("Quantum circuits ready!")
print("Train circuits shape:", train_circuits.shape)
print("Test circuits shape:", test_circuits.shape)
