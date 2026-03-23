Task 8: Vision Transformer & Quantum Vision Transformer

1. Introduction
A Vision Transformer (ViT) was implemented and applied to the MNIST dataset for image classification. The goal was to evaluate transformer-based architectures on image data and explore how such models can be extended into quantum machine learning through a Quantum Vision Transformer (QViT).
2. Classical Vision Transformer Implementation
The MNIST dataset was preprocessed and normalized before training. Images were divided into fixed-size patches, which were then flattened and projected into an embedding space. Positional encoding was added to preserve spatial structure.
The model consisted of:
Patch extraction and embedding
Transformer encoder blocks with multi-head self-attention
Feedforward neural networks
Global pooling and classification head
The model was trained using the Adam optimizer over multiple epochs.
3. Results
The Vision Transformer achieved:
Test Accuracy: 0.9712 (baseline model)
Test Accuracy: 0.9756 (enhanced model)
The enhanced model introduced deeper transformer layers and improved embedding, but resulted in significantly longer training time with only marginal performance improvement.
4. Performance Analysis & Fine-Tuning Insight
Increasing model complexity led to higher computational cost without significant gains in accuracy. This indicates that the MNIST dataset is relatively simple and does not require deep transformer architectures.
This experiment highlights an important trade-off:
Simpler models → faster training, sufficient accuracy
Complex models → slower training, marginal improvement
The baseline Vision Transformer provides an optimal balance between efficiency and performance for this task.
5. Quantum Vision Transformer (QViT) Concept
A Quantum Vision Transformer can be constructed by integrating quantum circuits into the architecture.
Proposed Architecture
Image → Patch extraction
Each patch → encoded into a quantum state using rotation gates (e.g., Ry, Rz)
Quantum circuits process patches individually
Entanglement between qubits replaces classical attention mechanisms
Measurement extracts expectation values
Classical dense layer performs final classification
6. Key Advantages of QViT
Quantum entanglement enables modeling of complex correlations
Hybrid quantum-classical design improves representational power
Potential for reduced parameterization in future large-scale models
7. Conclusion
The Vision Transformer successfully achieved high accuracy on MNIST without convolutional layers. Increasing architectural complexity showed diminishing returns, emphasizing the importance of model efficiency.
The Quantum Vision Transformer presents a promising extension by leveraging quantum encoding and entanglement for feature extraction, providing a pathway toward advanced hybrid quantum machine learning models.
