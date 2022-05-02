# Verifying Fairness in Quantum Machine Learning #

This repository contains two parts:
- An implementation for computing the Lipschitz constant of a quantum decision model (See Algorithm 1 in the paper).
- Experiment codes and data for CAV2022 Artifact Evaluation (See Section 6 in the paper).

## Requirements ##

- [Python3.8](https://www.python.org/).
- Python libraries: 
    * [Cirq](https://quantumai.google/cirq) for representing (noisy) quantum circuits.
    * [Tensornetwork](https://github.com/google/tensornetwork) for manipulating tensor networks.
    * [Numpy](https://numpy.org/) for linear algebra computations.
    * [Jax](https://github.com/google/jax) for  just-in-time (JIT) compilation in Python.
    * [Tensorflow Quantum](https://www.tensorflow.org/quantum) for training quantum decision models.
    * [Project Jupyter](https://jupyter.org/) for demonstrating the workflow of artifact.

## Installation (for Linux) ##

We recommend the users to use [Conda](https://docs.conda.io/en/latest/) to configure the Python environment.

### Install with Conda (Miniconda) ###
1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Clone this repository and cd to it.
    ```bash
    git clone https://github.com/Veri-Q/Fairness.git && cd Fairness
    ```
3. Use Conda to create a new Conda environment:
    ```bash
    conda create -n qfairness python=3.8.12
    ```
4. Activate the above environment and use pip to install required libraries in `requirements.txt`.
    ```bash
    conda activate qfairness
    pip install -r requirements.txt
    ```

## Computing the Lipschitz Constant ##

The file `qlipschitz.py` in this repository is the implementation of Algorithm 1 in the paper. It provides a function `lipschitz` that accepts a quantum decision model and outputs the model's Lipschitz constant as defined in the paper. The usage of `lipschitz` in Python is as follows:
```python
from qlipschitz import lipschitz

# ...

# model_circuit: the (noisy) quantum circuit written with cirq; It expresses the super-operator $\mathcal{E}$ in a quantum decision model.
# qubits: all (cirq) qubits used in this model; usually, qubits = model_circuit.all_qubits()
# measurement: a one qubit measurement (2x2 matrix) on the last qubit of qubits; It expresses the measurement $M$ in a quantum decision model.
k = lipschitz(model_circuit, qubits, measurement)

# ...
```

## Experiments (Artifact Evaluations) ##

### GC ###

### DiCE ###

###  QCNN models ###

We provide a script `evaluate_qcnn_model.py` to reproduce Table 2 in the paper.

```bash
python evaluate_qcnn_model.py <qubits_num> <noise_type>
```
where `<qubits_num>` is the number of qubits (integer), `<noise_type>` is the type of noise that can be the following four values: `phase_flip` for phase flip noise, `depolarize` for depolarize noise, `bit_flip` for bit flip noise and `mixed` for mixed noise.

e.g. run command `python evaluate_qcnn_model.py 25 depolarize` can reproduce the results of **25 Qubits** and **Depolarize Noise** in Table 2.