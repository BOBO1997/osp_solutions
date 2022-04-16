import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def trotter_gate(dt, to_instruction = True):
    qc = QuantumCircuit(2)
    qc.rx(2 * dt, 0)
    qc.rz(2 * dt, 1)
    qc.h(1)
    qc.cx(1, 0)
    qc.rz(-2 * dt, 0)
    qc.rx(-2 * dt, 1)
    qc.rz(2 * dt, 1)
    qc.cx(1, 0)
    qc.h(1)
    qc.rz(2 * dt, 0)
    return qc.to_instruction() if to_instruction else qc


def make_initial_state(qc, initial_state):
    """
    logical qubit index
    little endian
    """
    for i, state in enumerate(initial_state):
        if state == "1":
            qc.x(i)


def subspace_encoder(qc, targets):
    """
    naive method, can be optimized for init state |110>
    little endian
    """
    n = qc.num_qubits
    qc.cx(targets[2],targets[1])
    qc.cx(targets[1],targets[0])
    
    
def subspace_encoder_init110(qc, targets):
    """
    optimized encoder for init state |110>
    endian: |q_0, q_1, q_2> (little endian)
    encode |110> to |0>|10>
    """
    n = qc.num_qubits
    qc.x(targets[0])
    
    
def subspace_decoder(qc, targets):
    """
    naive method
    little endian
    """
    n = qc.num_qubits
    qc.cx(targets[1], targets[0])
    qc.cx(targets[2], targets[1])
    
    
def subspace_decoder_init110(qc, targets):
    """
    optimized decoder for init state |110>
    endian: |q_0, q_1, q_2> (little endian)
    decode |0>|10> to |110>
    """
    n = qc.num_qubits
    qc.x(targets[0])
    
    
def trotterize(qc, trot_gate, num_steps, targets):
    for _ in range(num_steps):
        qc.append(trot_gate, qargs = targets)