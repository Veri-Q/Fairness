import tensorflow as tf
import tensorflow_quantum as tfq

import sympy
import cirq
import numpy as np
import time
import sys

import pandas as pd

df = pd.read_csv("./data/german_credit.csv")
train_f = pd.read_csv("./data/Training50.csv")
test_f = pd.read_csv("./data/Test50.csv")
features = ['Account.Balance',
            'Payment.Status.of.Previous.Credit',
            'Purpose',
            'Value.Savings.Stocks',
            'Length.of.current.employment',
            'Sex...Marital.Status',
            'Guarantors',
            'Concurrent.Credits',
            'No.of.Credits.at.this.Bank']
features_MAX = [3., 3., 4., 4., 4., 3., 2., 2., 2.]

X_train = tf.convert_to_tensor(train_f[features],).numpy()
Y_train = tf.convert_to_tensor(train_f['Creditability'],).numpy()
X_test = tf.convert_to_tensor(test_f[features],).numpy()
Y_test = tf.convert_to_tensor(test_f['Creditability'],).numpy()

NUM_QUBITS = X_train.shape[1]
WORKING_QUBITS = cirq.GridQubit.rect(1,NUM_QUBITS)

def generate_data_circuit(data):
    qubits = WORKING_QUBITS
    output = []
    for j in range(data.shape[0]):
        circuit = cirq.Circuit()
        for k in range(data.shape[1]):
            circuit += cirq.X(qubits[k]) ** (data[j,k]/features_MAX[k])
        output.append(circuit)
    return tfq.convert_to_tensor(output)

def generate_model_circuit(variables, p=0., noise_op=cirq.depolarize, mixed=False):
    qubits = WORKING_QUBITS
    symbols = variables
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    
    if p > 1e-5:
        if mixed:
            circuit += cirq.bit_flip(p).on_each(*qubits[::3])
            circuit += cirq.depolarize(p).on_each(*qubits[1::3])
            circuit += cirq.phase_flip(p).on_each(*qubits[2::3])
        else:
            circuit += noise_op(p).on_each(*qubits)
        
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
        
    circuit += cirq.X(qubits[-1]) ** next(symbols)
    circuit += cirq.Y(qubits[-1]) ** next(symbols)
    circuit += cirq.X(qubits[-1]) ** next(symbols)
    
    return circuit

def circuit2M(p, variables, noise_op=cirq.depolarize, mixed=False):
    qubits = WORKING_QUBITS
    num = len(qubits)
    variables = iter(variables)
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    U1 = cirq.unitary(circuit)

    if p > 1e-5:
        if mixed:
            noisy_kraus = [cirq.kraus(cirq.bit_flip(p)(q)) for q in qubits[::3]]
            noisy_kraus += [cirq.kraus(cirq.depolarize(p)(q)) for q in qubits[1::3]]
            noisy_kraus += [cirq.kraus(cirq.phase_flip(p)(q)) for q in qubits[2::3]]
        else:
            noisy_kraus = [cirq.channel(noise_op(p)(q)) for q in qubits] 
    
    circuit = cirq.Circuit()
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    U2 = cirq.unitary(circuit)

    M = U2.conj().T @ np.kron(np.eye(2 ** (num - 1)), np.array([[1.,0.], [0.,0.]])) @ U2
    
    if p > 1e-5:
        for j in range(num):
            N = 0
            for E in noisy_kraus[j]:
                F = np.kron(np.eye(2 ** j), np.kron(E, np.eye(2 ** (num-j-1))))
                N = F.conj().T @ M @ F + N
        
            M = N
    
    M = U1.conj().T @ M @ U1
    return M

def make_quantum_model(p, noise_op, mixed):
    qubits = WORKING_QUBITS
    num = len(qubits)
    num_para = num * 8 + 3
    symbols = iter(sympy.symbols('qgenerator0:%d'%(num_para)))
    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    if p > 1e-5:
        quantum_layer = tfq.layers.NoisyPQC(
            generate_model_circuit(symbols, p, noise_op, mixed),
            cirq.Z(qubits[-1]),
            repetitions=100,
            sample_based=False
        )(circuit_input)
    else:
        quantum_layer = tfq.layers.PQC(
            generate_model_circuit(symbols),
            cirq.Z(qubits[-1])
        )(circuit_input)
    
    return tf.keras.Model(inputs=[circuit_input], outputs=[0.5 * (quantum_layer + tf.constant(1.))])

X_train_input = generate_data_circuit(X_train)
X_test_input = generate_data_circuit(X_test)

noise_type = str(sys.argv[1])
noisy_p = float(sys.argv[2])
noise_op = {
        "phase_flip": cirq.phase_flip,
        "depolarize": cirq.depolarize,
        "bit_flip": cirq.bit_flip,
        "mixed": cirq.depolarize
}
if noise_type == "mixed":
    mixed = True
else:
    mixed = False

noisy_model = make_quantum_model(noisy_p, noise_op[noise_type], mixed)
noisy_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    #loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)

print("================Training Start=================")
train_history = noisy_model.fit(
    x=X_train_input,
    y=Y_train,
    batch_size=100,
    epochs=100,
    verbose=1,
    validation_data=(X_test_input, Y_test)
)
print("=================Training End==================")

tstart = time.time()
print("\n===========Lipschitz Constant Start============")
a,_= np.linalg.eig(circuit2M(noisy_p,noisy_model.layers[1].get_weights()[0],noise_op[noise_type], mixed))
k = np.real(max(a) - min(a))
if k != -1:
    print("Lipschitz K = ", k)
else:
    print("Lipschitz K = -")
print(f"Elapsed time = {(time.time() - tstart):.4f}s")
print("============Lipschitz Constant End=============")
