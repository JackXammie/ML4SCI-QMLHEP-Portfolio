Task 7 Report: Equivariant Quantum Neural Networks (EQNN)
1. Introduction
Equivariant Quantum Neural Networks based on the symmetry group Z × Z were implemented and evaluated. The goal was to compare a standard Quantum Neural Network (QNN) with a symmetry-aware EQNN on a classification task where the data respects a reflection symmetry.
2. Dataset
A synthetic dataset with two features (x_one and x_two) was generated. The dataset follows a Z × Z symmetry, meaning it is invariant under swapping the two features (reflection along the line y = x).
Two classes were defined based on this symmetric structure.
3. Quantum Circuit Design
Standard QNN
The standard QNN encodes classical features into a quantum circuit using parameterized rotations:
Rotation-Y gates (Ry) for encoding input features
Rotation-Z gates (Rz) for trainable parameters
This model does not explicitly enforce symmetry.
Equivariant QNN (EQNN)
The EQNN enforces symmetry by design:
Parameters are shared across qubits
The circuit structure respects the symmetry between x_one and x_two
Swapping inputs leads to consistent outputs
This ensures the model is aligned with the underlying structure of the dataset.
4. Training
Both models were trained using:
Binary classification objective
Adam optimizer
Same number of epochs and training samples
The goal was to compare performance under identical conditions.
5. Results
Standard QNN Accuracy: 0.96
Equivariant QNN Accuracy: 0.74
6. Analysis
The standard QNN achieved higher accuracy, likely due to greater flexibility and lack of constraints.
The EQNN, while slightly less accurate, benefits from:
Better generalization in theory
Reduced parameter space
Built-in symmetry awareness
The lower performance may be due to:
Small dataset size
Limited circuit depth
Underfitting caused by strong symmetry constraints
7. Conclusion
Equivariant Quantum Neural Networks successfully incorporate symmetry into quantum models.
While the standard QNN achieved higher accuracy in this experiment, EQNNs provide a principled way to embed prior knowledge into model design.
Future improvements may include:
Increasing circuit depth
Using larger datasets
Combining symmetry with more expressive quantum layers
