from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter
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
### from osp_solutions ###


def gate_path_hybrid(num_qubits: int,
                     num_steps: int, ###! note that this is based on counting the number of proposed Trotter iterations
                     dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                     type_H_eff: str = None,
                     to_instruction: bool = True,
                     add_barrier: bool = False,
                    ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 2\Delta t
    """
    assert num_qubits % 4 == 1

    qc = QuantumCircuit(num_qubits,
                        name="path_hybrid")

    for ith_step in range(num_steps): ###! note that this is based on counting the number of proposed Trotter iterations
        for ith_qubit in range(0, num_qubits - 2, 4):
            qc.compose(gate_block_trotter_hybrid_1d(dt=dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                                                type_H_eff=type_H_eff,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier),
                        qubits=list(range(ith_qubit, ith_qubit + 5)),
                        inplace=True,)
        
        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="path_hybrid") if to_instruction else qc


### ================================================== ###


def gate_path_conventional(num_qubits: int, 
                           num_steps: int, 
                           dt: float,
                           type_block: str = "3cnot",
                           to_instruction: bool = True,
                           add_barrier: bool = False,
                          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter iterations for path structure
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
                        name="path_conventional")
    
    for ith_step in range(num_steps):
    
        if not (num_qubits & 1): ### even
            for ith_qubit in range(0, num_qubits, 2): ### even indices
                qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits = [ith_qubit, ith_qubit + 1],
                            inplace=True,)
            for ith_qubit in range(1, num_qubits - 1, 2): ### odd indices
                qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits = [ith_qubit, ith_qubit + 1],
                            inplace=True,)
            # if add_barrier:
            #     qc.barrier([ith_qubit, ith_qubit + 1])

        if num_qubits & 1: ### odd
            for ith_qubit in range(0, num_qubits - 1, 2):  ### even indices
                qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits = [ith_qubit, ith_qubit + 1],
                            inplace=True,)
            for ith_qubit in range(1, num_qubits, 2): ### odd indices
                qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits = [ith_qubit, ith_qubit + 1],
                            inplace=True,)
            # if add_barrier:
            #     qc.barrier([ith_qubit, ith_qubit + 1])

        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="path_conventional") if to_instruction else qc


### ================================================== ###


def gate_path_triangle(num_qubits: int,
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
    assert num_qubits % 2 == 1 ###! 1 mod 2: It is ok even for 7-qubits

    qc = QuantumCircuit(num_qubits,
                        name="path_triangle")

    for ith_step in range(num_steps):
        ###
        for ith_qubit in range(2, num_qubits - 2, 4): ### encoder for blue H_eff ### 2 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                        type_H_eff=type_H_eff,
                                                        connectivity=connectivity,
                                                        to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                        qubits=[jth_qubit for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits - 2, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                        type_H_eff=type_H_eff,
                                                        connectivity=connectivity,
                                                        to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                        qubits=[jth_qubit for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="path_triangle") if to_instruction else qc


### ================================================== ###


def gate_path_triangle_parallel(num_qubits: int,
                                num_steps: int, ###! note that this is based on counting the number of proposed Trotter iterations
                                dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                                connectivity: str = "complete",
                                type_H_eff: str = None,
                                to_instruction: bool = True,
                                add_barrier: bool = False,
                               ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 4\Delta t
    """
    assert num_qubits % 4 == 1 ###! for patch
    # assert num_qubits % 2 == 1 ###! 1 mod 2: It is ok even for 7-qubits

    qc = QuantumCircuit(num_qubits,
                        name="path_triangle")

    for ith_step in range(num_steps):
        ###
        for ith_qubit in range(2, num_qubits - 2, 4): ### encoder for blue H_eff ### 2 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                    type_H_eff=type_H_eff,
                                                    type_enc_dec="H", ###!
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits - 2, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                    type_H_eff=type_H_eff,
                                                    type_enc_dec=None,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits - 2, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                    type_H_eff=type_H_eff,
                                                    type_enc_dec="H", ###!
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(3, num_qubits - 2, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_block_trotter_triangle_1d(dt=dt,
                                                    type_H_eff=type_H_eff,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###! patch
        qc.compose(gate_block_trotter_3cnot(dt=dt,
                                            # option="d",
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier),
                    qubits=[0, 1],
                    inplace=True,)
        qc.compose(gate_block_trotter_3cnot(dt=dt,
                                            option="d", ###! to further save two CNOT gates
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier),
                    qubits=[num_qubits - 2, num_qubits - 1],
                    inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="path_triangle") if to_instruction else qc