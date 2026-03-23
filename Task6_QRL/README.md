Quantum Representation Learning (QRL) – Task 6
This repository implements a simple quantum representation learning scheme using the MNIST dataset and a SWAP test to compare fidelity between quantum state embeddings. The contrastive loss trains an embedding function such that images of the same class have high fidelity and images of different classes have low fidelity.

Structure

qrl_model.py — defines the quantum feature embedding and SWAP test circuits
qrl_train.py — performs training using contrastive loss
Task6_Report.md — detailed explanation of design and results
Description
A quantum circuit encodes classical images into quantum states with trainable parameters. A SWAP test is used to measure fidelity between two image embeddings. A contrastive loss is defined to maximize fidelity for similar images and minimize it for dissimilar images.

Requirements

Python 3.10+
Cirq
TensorFlow Quantum
TensorFlow Keras
NumPy
