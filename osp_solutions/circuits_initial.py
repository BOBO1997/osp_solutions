from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter # , QuantumRegister


### ================================================== ###


def gate_initial_state(state_initial: str,
                       endian_state_initial: str = "big",
                       to_instruction: bool = True,
                       add_barrier: bool = False,
                      ) -> Union[QuantumCircuit, Instruction]:
    """
    The state_initial should be the string in big endian.

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    if endian_state_initial == "little":
        state_initial = state_initial[::-1]

    qc = QuantumCircuit(len(state_initial))
    
    for i, state in enumerate(state_initial):
        if state == "1":
            qc.x(i)

    if add_barrier:
        qc.barrier(label=state_initial)

    return qc.to_instruction(label=state_initial) if to_instruction else qc