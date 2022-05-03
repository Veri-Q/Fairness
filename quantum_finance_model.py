import sympy
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq


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