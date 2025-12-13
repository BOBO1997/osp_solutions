from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter # , QuantumRegister
from osp_solutions.circuits_util import (
    gate_block_trotter_qiskit,
    gate_block_trotter_6cnot,
    gate_block_trotter_3cnot,
)
from osp_solutions.circuits_1d_util import (
    gate_block_trotter_hybrid_1d,
    gate_block_trotter_triangle_1d,
)


### ================================================== ###


def gate_ring_conventional(num_qubits: int, 
                           num_steps: int, 
                           dt: float, 
                           type_block: str = "3cnot",
                           to_instruction: bool = True,
                           add_barrier: bool = False,
                          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter iterations for ring structure
    """

    ### choose the type of Trotter block
    if type_block == "qiskit":
        gate_block_trotter = gate_block_trotter_qiskit
    elif type_block == "6cnot":
        gate_block_trotter = gate_block_trotter_6cnot
    elif type_block == "3cnot":
        gate_block_trotter = gate_block_trotter_3cnot
    else:
        raise Exception("specify a valid type for the Trotter block")
    
    qc = QuantumCircuit(num_qubits,
                        name="ring_conventional")
    
    for ith_step in range(num_steps):

        ### Trotter blocks among direct neighbours
        for ith_qubit in range(0, num_qubits, 2): ### even
            qc.compose(gate_block_trotter(dt=dt,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        for ith_qubit in range(1, num_qubits, 2): ### odd
            qc.compose(gate_block_trotter(dt=dt,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
                    
        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="ring_conventional") if to_instruction else qc


### ================================================== ###


def gate_ring_hybrid(num_qubits: int,
                     num_steps: int, ###! note that this is based on counting the number of proposed Trotter iterations
                     dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                     type_H_eff: str = None,
                     connectivity: str = "complete",
                     to_instruction: bool = True,
                     add_barrier: bool = False,
                    ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 2\Delta t
    """
    assert num_qubits % 4 == 0 ### 0 mod 4

    qc = QuantumCircuit(num_qubits,
                        name="ring_hybrid")

    for ith_step in range(num_steps): ###! note that this is based on counting the number of proposed Trotter iterations
        ### U
        for ith_qubit in range(3, num_qubits, 4): ### 3 mod 4
            qc.compose(gate_block_trotter_3cnot(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        ### H_eff
        for ith_qubit in range(1, num_qubits, 4): ### 1 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                        type_H_eff=type_H_eff,
                                                        connectivity=connectivity,
                                                        to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ### U
        for ith_qubit in range(0, num_qubits, 4): ### 0 mod 4
            qc.compose(gate_block_trotter_3cnot(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
                
        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="ring_hybrid") if to_instruction else qc


### ================================================== ###


def gate_ring_triangle(num_qubits: int,
                       num_steps: int, ###! note that this is based on counting the number of proposed Trotter iterations
                       dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                       connectivity: str = "complete",
                       type_H_eff: str = None,
                       to_instruction: bool = True,
                       add_barrier: bool = False,
                      ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 2\Delta t
    """
    assert num_qubits % 4 == 0 ### 0 mod 4

    qc = QuantumCircuit(num_qubits,
                        name="ring_triangle")

    for ith_step in range(num_steps):
        ###
        for ith_qubit in range(2, num_qubits, 4): ### 2 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                        type_H_eff=type_H_eff,
                                                        connectivity=connectivity,
                                                        to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits, 4): ### 0 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                        type_H_eff=type_H_eff,
                                                        connectivity=connectivity,
                                                        to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                        inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="ring_triangle") if to_instruction else qc