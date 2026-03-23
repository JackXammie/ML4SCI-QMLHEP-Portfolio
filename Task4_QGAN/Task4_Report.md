Task 4 Report: Quantum Generative Adversarial Network (QGAN)
1. Introduction
Quantum Generative Adversarial Network (QGANs) are a type of neural network used for unsupervised type of machine learning, it is an interaction between a generator and a discriminator, which are in constant competition with one another, where a fake data is put up by a generator and the discriminator tries to call the bluff and the wrong entity tries to upgrade for better accuracy. In Task 4 I explored how to classify high-energy physics data, separating signal events from background events. The QGAN was implemented using Google Cirq and TensorFlow Quantum (TFQ), encoding classical data into quantum circuits and training a generator and discriminator network. The dataset from Delphes consisted of 100 training samples and 100 test samples.
2. Data Preparation
The dataset was loaded and split into signal and background events. Features were scaled and reduced in dimensionality to make them compatible with quantum circuits. Shuffling was applied to ensure randomness in the training and test sets.
3. Quantum Circuit Encoding
Each feature vector was encoded into a quantum circuit using parametric rotations on four (4) qubits. These circuits served as the input for both the generator and discriminator in the QGAN.
4. QGAN Model
The QGAN consisted of two main components:
Generator: Produces “fake” quantum circuits from random noise.
Discriminator: Classifies circuits as real (signal/background) or fake.
The generator and discriminator were trained together in an adversarial manner to improve classification accuracy.
5. Training & Fine-Tuning
The model was trained for multiple epochs with small batch sizes. Fine-tuning involved:
(a) Adjusting learning rate and batch size
(b) Modifying circuit depth and parametric rotations
(c) Optimizing generator noise dimension
This process stabilized training and improved the discriminator’s ability to distinguish real from fake circuits.
6. Results
The QGAN learned patterns from the data and gradually improved classification performance. Early epochs showed fluctuating performance due to the small dataset and quantum circuit noise, but fine-tuning achieved a Test AUC of 0.62, indicating better-than-random separation of signal and background events.
7. Conclusion
QGAN successfully separated the signals from background events using hybrid classical-quantum models.
Fine-tuning hyperparameters and circuit design improved performance.
Future work can involve larger datasets, more qubits, and testing on real quantum hardware to further enhance accuracy.
