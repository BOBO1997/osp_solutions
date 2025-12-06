from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter
from osp_solutions.circuits_util import (
    gate_block_trotter_qiskit,
    gate_block_trotter_6cnot,
    gate_block_trotter_3cnot,
    gate_proposed_2t
)

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
                if to_instruction:
                    qc.append(gate_block_trotter(dt=dt,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier), 
                                qargs=[ith_qubit, ith_qubit + 1])
                else:
                    qc.compose(gate_block_trotter(dt=dt,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier), 
                                qubits = [ith_qubit, ith_qubit + 1],
                                inplace=True,)
            for ith_qubit in range(1, num_qubits - 1, 2): ### odd indices
                if to_instruction:
                    qc.append(gate_block_trotter(dt=dt,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier), 
                                qargs=[ith_qubit, ith_qubit + 1])
                else:
                    qc.compose(gate_block_trotter(dt=dt,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier), 
                                qubits = [ith_qubit, ith_qubit + 1],
                                inplace=True,)
            # if add_barrier:
            #     qc.barrier([ith_qubit, ith_qubit + 1])

        if num_qubits & 1: ### odd
            for ith_qubit in range(0, num_qubits - 1, 2):  ### even indices
                if to_instruction:
                    qc.append(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier),
                                qargs=[ith_qubit, ith_qubit + 1])
                else:
                    qc.compose(gate_block_trotter(dt=dt,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier), 
                                qubits = [ith_qubit, ith_qubit + 1],
                                inplace=True,)
            for ith_qubit in range(1, num_qubits, 2): ### odd indices
                if to_instruction:
                    qc.append(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                                qargs=[ith_qubit, ith_qubit + 1])
                else:
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


def gate_path_proposed(num_qubits: int,
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
    ### assert not (num_steps & 1) ###! we no longer have to check this
    assert (num_qubits - 1) % 4 == 0

    qc = QuantumCircuit(num_qubits,
                        name="path_proposed")

    for ith_step in range(num_steps): ###! note that this is based on counting the number of proposed Trotter iterations
        for ith_qubit in range(0, num_qubits - 2, 4):
            if to_instruction:
                qc.append(gate_proposed_2t(dt=dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                                            type_H_eff=type_H_eff,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                            qargs=list(range(ith_qubit, ith_qubit + 5)))
            else:
                qc.compose(gate_proposed_2t(dt=dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                                            type_H_eff=type_H_eff,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier),
                            qubits=list(range(ith_qubit, ith_qubit + 5)),
                            inplace=True,)
        
        if add_barrier:
            qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="path_proposed") if to_instruction else qc

