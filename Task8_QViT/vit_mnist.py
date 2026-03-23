import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# -----------------------------
# Load MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dims (for patches)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# -----------------------------
# Patch creation
# -----------------------------
PATCH_SIZE = 7
NUM_PATCHES = (28 // PATCH_SIZE) ** 2
EMBED_DIM = 64

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# -----------------------------
# Patch embedding
# -----------------------------
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.projection = layers.Dense(embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=NUM_PATCHES, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

# -----------------------------
# Transformer block
# -----------------------------
def transformer_block(x):
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=EMBED_DIM)(x, x)
    x = layers.LayerNormalization()(x + attention)

    ffn = layers.Dense(128, activation="relu")(x)
    ffn = layers.Dense(EMBED_DIM)(ffn)

    x = layers.LayerNormalization()(x + ffn)
    return x

# -----------------------------
# Build ViT model
# -----------------------------
inputs = layers.Input(shape=(28, 28, 1))

patches = Patches(PATCH_SIZE)(inputs)
encoded = PatchEncoder(NUM_PATCHES, EMBED_DIM)(patches)

# Transformer layers
for _ in range(2):
    encoded = transformer_block(encoded)

# Classification head
representation = layers.GlobalAveragePooling1D()(encoded)
outputs = layers.Dense(10, activation="softmax")(representation)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# -----------------------------
# Compile & Train
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=10, batch_size=64)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
