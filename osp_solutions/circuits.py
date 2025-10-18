from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter # , QuantumRegister


def gate_U(dt, 
           option: str = "a",
           to_instruction = True,
          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """
    qc = QuantumCircuit(2)

    if option == "a":
        qc.cx(0, 1)

        qc.rx(2 * dt, 0)
        qc.h(0)
        qc.rz(- 2 * dt, 1)

        qc.cx(0, 1)

        qc.s(0)
        qc.h(0)
        qc.rz(- 2 * dt, 1)
        
        qc.cx(0, 1)

        qc.rx(- np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)

    elif option == "b":
        qc.cx(1, 0)

        qc.rz(- 2 * dt, 0)
        qc.rx(2 * dt, 1)
        qc.h(1)

        qc.cx(1, 0)

        qc.rz(- 2 * dt, 0)
        qc.s(1)
        qc.h(1)
        
        qc.cx(1, 0)

        qc.rx(np.pi / 2, 0)
        qc.rx(- np.pi / 2, 1)

    elif option == "c":
        qc.rx(np.pi / 2, 0)
        qc.rx(- np.pi / 2, 1)

        qc.cx(0, 1)

        qc.h(0)
        qc.sdg(0)
        qc.rz(- 2 * dt, 1)
        
        qc.cx(0, 1)

        qc.h(0)
        qc.rx(2 * dt, 0)
        qc.rz(- 2 * dt, 1)

        qc.cx(0, 1)

    else:
        raise Exception

    return qc.to_instruction(label="Trotter") if to_instruction else qc


def gate_U_prime(dt, 
                 to_instruction = True,
                ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """
    qc = QuantumCircuit(2)

    qc.rx(np.pi / 2, 0)
    qc.rx(- np.pi / 2, 1)

    qc.cx(0, 1)

    qc.h(0)
    qc.sdg(0)
    qc.rz(- 2 * dt, 1)

    qc.cx(0, 1)

    qc.h(0)
    qc.rx(2 * dt, 0)
    qc.rz(- 2 * dt, 1)

    return qc.to_instruction(label="Trotter") if to_instruction else qc


def gate_U_prime_prime(dt, 
                       to_instruction = True,
                      ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """
    qc = QuantumCircuit(2)
    qc.rz(- 2 * dt, 0)
    qc.rx(2 * dt, 1)
    qc.h(1)

    qc.cx(1, 0)

    qc.rz(- 2 * dt, 0)
    qc.s(1)
    qc.h(1)

    qc.cx(1, 0)
    
    qc.rx(np.pi / 2, 0)
    qc.rx(- np.pi / 2, 1)

    return qc.to_instruction(label="Trotter") if to_instruction else qc


def gate_Heff(dt, 
              to_instruction = True
             ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """
    qc = QuantumCircuit(2)

    qc.rx(2 * dt, 0)
    qc.rz(2 * dt, 1)
    qc.h(1)

    qc.cx(1, 0)

    qc.rz(- 2 * dt, 0)
    qc.rx(- 2 * dt, 1)
    qc.rz(2 * dt, 1)

    qc.cx(1, 0)
    qc.h(1)
    qc.rz(2 * dt, 0)

    return qc.to_instruction(label="Trotter") if to_instruction else qc


def gate_proposed_2t(dt,
                     to_instruction = True
                    ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    """
    qc = QuantumCircuit(5)

    qc.append(gate_U_prime(dt), qargs = [1, 2])
    qc.append(gate_U_prime(2 * dt), qargs = [3, 4])

    qc.cx(3, 2)
    qc.cx(2, 1)
    qc.cx(3, 4)
    qc.cx(2, 3)

    qc.append(gate_Heff(dt), qargs = [1, 3])

    qc.cx(2, 1)
    qc.cx(1, 0)
    qc.cx(2, 3)
    qc.cx(1, 2)

    qc.append(gate_U_prime_prime(2 * dt), qargs = [0, 1])
    qc.append(gate_U_prime_prime(dt), qargs = [2, 3])

    return qc.to_instruction(label="Trotter") if to_instruction else qc


### ================================================== ###


def append_block_trotter(qc: QuantumCircuit, 
                         dt,
                         num_steps: int, 
                        ) -> None:
    """
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter block for \Delta t
    """
    num_qubits = qc.num_qubits
    
    if not num_qubits & 1: ### even
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits):
                if not ith_qubit & 1: ### even
                    qc.append(gate_U(dt), qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits - 1):
                if ith_qubit & 1: ### odd
                    qc.append(gate_U(dt), qargs = [ith_qubit, ith_qubit + 1])
    if num_qubits & 1: ### odd
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits - 1):
                if not ith_qubit & 1: ### even
                    qc.append(gate_U(dt), qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits):
                if ith_qubit & 1: ### odd
                    qc.append(gate_U(dt), qargs = [ith_qubit, ith_qubit + 1])


def append_block_trotter_proposed(qc: QuantumCircuit, 
                                  dt,
                                  num_steps: int, 
                                 ) -> None:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 2\Delta t
    """
    assert not (num_steps & 1)
    num_qubits = qc.num_qubits
    assert (num_qubits - 1) % 4 == 0

    for _ in range(num_steps // 2):
        for ith_qubit in range(num_qubits - 2):
            if ith_qubit % 4 == 0: ### even
                qc.append(gate_proposed_2t(dt), qargs = list(range(ith_qubit, ith_qubit + 4)))


def trotterize(qc, 
               trot_gate, 
               num_steps, 
               targets,
              ) -> None:
    """
    old function for contest
    """
    for _ in range(num_steps):
        qc.append(trot_gate, qargs = targets)


### ================================================== ###


def append_initial_state(qc: QuantumCircuit, 
                         state_initial_str: str,
                        ) -> None:
    """
    The state_initial_str is the string in little endian.
    """
    for i, state in enumerate(state_initial_str):
        if state == "1":
            qc.x(i)


def general_subspace_encoder(qc, targets) -> None:
    """
    Generalized method for any initial state
    little endian
    """
    n = qc.num_qubits
    qc.cx(targets[0],targets[1])
    qc.cx(targets[2],targets[1])
    qc.cx(targets[1],targets[2])
    qc.cx(targets[0],targets[1])
    qc.cx(targets[1],targets[0])


def general_subspace_decoder(qc, targets) -> None:
    """
    generalized method for any initial state
    little endian
    """
    n = qc.num_qubits
    qc.cx(targets[1],targets[0])
    qc.cx(targets[0],targets[1])
    qc.cx(targets[1],targets[2])
    qc.cx(targets[2],targets[1])
    qc.cx(targets[0],targets[1])


def subspace_encoder(qc, targets) -> None:
    """
    naive method, can be optimized for init state |110>
    little endian
    """
    n = qc.num_qubits
    qc.cx(targets[2],targets[1])
    qc.cx(targets[1],targets[0])
    
    
def subspace_decoder(qc, targets) -> None:
    """
    naive method
    little endian
    """
    n = qc.num_qubits
    qc.cx(targets[1], targets[0])
    qc.cx(targets[2], targets[1])
    
    
def subspace_encoder_init110(qc, targets) -> None:
    """
    optimized encoder for init state |110>
    endian: |q_0, q_1, q_2> (little endian)
    encode |110> to |0>|10>
    """
    n = qc.num_qubits
    qc.x(targets[0])
    
    
def subspace_decoder_init110(qc, targets) -> None:
    """
    optimized decoder for init state |110>
    endian: |q_0, q_1, q_2> (little endian)
    decode |0>|10> to |110>
    """
    n = qc.num_qubits
    qc.x(targets[0])