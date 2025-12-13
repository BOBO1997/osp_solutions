from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter # , QuantumRegister
from osp_solutions.circuits_util import (
    gate_block_trotter_3cnot,
    gate_U_enc, 
    gate_U_dec,
    gate_U_enc_H, 
    gate_U_dec_H,
)


### ================================================== ###
### library of different H_eff funcitons ###

def gate_H_eff_1d_old(dt,
                   to_instruction: bool = True,
                   add_barrier: bool = False,
                  ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    function for arXiv paper: 2505.04552
    """

    qc = QuantumCircuit(2,
                        name="H_eff_1d_old")

    qc.rx(2 * dt, 0)
    qc.rz(2 * dt, 1)
    qc.h(1)

    qc.cx(1, 0)

    qc.rz(- 2 * dt, 0)
    qc.rx(- 2 * dt, 1)
    qc.rz(2 * dt, 1)

    qc.cx(1, 0)

    qc.rz(2 * dt, 0)
    qc.h(1)

    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="H_eff_old") if to_instruction else qc


def gate_H_eff_1d_new(dt,
                   to_instruction: bool = True,
                   add_barrier: bool = False,
                  ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    function for arXiv paper: 2505.04552
    """

    qc = QuantumCircuit(2,
                        name="H_eff_1d_new")

    qc.ry(- np.pi / 4, 0)
    qc.ry(- np.pi / 4, 1)

    qc.rz(np.sqrt(2) * dt, 0)
    qc.rz(np.sqrt(2) * dt, 1)

    qc.cx(1, 0)

    qc.rz(- 2 * dt, 0)
    qc.rx(2 * dt, 1)

    qc.cx(1, 0)

    qc.rz(np.sqrt(2) * dt, 0)
    qc.rz(np.sqrt(2) * dt, 1)

    qc.ry(np.pi / 4, 0)
    qc.ry(np.pi / 4, 1)

    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="H_eff_new") if to_instruction else qc


###! wrong!!!
def gate_H_eff_1d_exact(dt,
                        to_instruction: bool = True,
                        add_barrier: bool = False,
                       ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    function for arXiv paper: 2505.04552, but with exact implementation without Trotter error
    """

    theta = - np.arctan(2 * np.sqrt(2)) / 2

    qc = QuantumCircuit(2,
                        name="H_eff_1d_exact")

    ### V_enc
    qc.ry(- np.pi / 4, 0)
    qc.ry(- np.pi / 4, 1)

    qc.cx(1, 0)

    qc.rz(- 2 * dt, 0)
    qc.x(0)
    qc.ry(- theta, 1)

    qc.cx(0, 1)

    qc.ry(theta, 1)
    qc.rx(4 * dt, 1)
    qc.h(1)

    qc.cx(0, 1)

    qc.rz(- 2 * dt, 1)

    qc.cx(0, 1)

    qc.h(1)
    qc.ry(- theta, 1)

    qc.cx(0, 1)

    qc.x(0)
    qc.ry(theta, 1)

    qc.cx(1, 0)

    ### V_dec
    qc.ry(np.pi / 4, 0)
    qc.ry(np.pi / 4, 1)

    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="H_eff_exact") if to_instruction else qc


### ================================================== ###
### from osp_solutions ###


def gate_block_trotter_hybrid_1d(dt, ###! dt follows the same scale as the arXiv paper
                              type_H_eff: str = None,
                              to_instruction: bool = True,
                              add_barrier: bool = False,
                             ) -> Union[QuantumCircuit, Instruction]:
    """
    This function is a copy of gate_triangle_2t in osp_solutions: https://github.com/BOBO1997/osp_solutions/
    dt: qiskit._accelerate.circuit.Parameter, this corresponds to Delta t' in the arXiv paper: 2505.04552
    """

    if type_H_eff == "new":
        gate_H_eff_1d = gate_H_eff_1d_new
    elif type_H_eff == "old":
        gate_H_eff_1d = gate_H_eff_1d_old
    elif type_H_eff == "exact":
        gate_H_eff_1d = gate_H_eff_1d_exact
    else:
        raise Exception("specify new or old")

    qc = QuantumCircuit(5)

    qc.compose(gate_block_trotter_3cnot(dt=dt, ###!
                                        option="b", # "d",
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier), 
                qubits=[3, 4],
                inplace=True,)
        
    if add_barrier:
        qc.barrier()

    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 1)

    if add_barrier:
        qc.barrier()

    qc.compose(gate_H_eff_1d(dt=dt, ###!
                            to_instruction=to_instruction,
                            add_barrier=add_barrier,
                            ),
                qubits=[1, 2],
                inplace=True,
                )

    qc.cx(3, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    if add_barrier:
        qc.barrier()

    qc.compose(gate_block_trotter_3cnot(dt=dt, ### !
                                        option="b", # "c",
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier), 
                qubits=[0, 1],
                inplace=True,)
        
    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="block_trotter_hybrid") if to_instruction else qc


### ================================================== ###


def gate_block_trotter_triangle_1d(dt: float,
                                    type_H_eff: str = None,
                                    type_enc_dec: str = None,
                                    connectivity: str = "complete",
                                    to_instruction: bool = True,
                                    add_barrier: bool = False,
                                  ) -> Union[QuantumCircuit, Instruction]:
    if type_H_eff == "new":
        gate_H_eff_1d = gate_H_eff_1d_new
    elif type_H_eff == "old":
        gate_H_eff_1d = gate_H_eff_1d_old
    elif type_H_eff == "exact":
        gate_H_eff_1d = gate_H_eff_1d_exact
    else:
        raise Exception("specify new or old")
    
    if type_enc_dec == "H":
        gate_U_enc_temp = gate_U_enc_H
        gate_U_dec_temp = gate_U_dec_H
    else:
        gate_U_enc_temp = gate_U_enc
        gate_U_dec_temp = gate_U_dec
    
    qc = QuantumCircuit(3,
                        name="block_trotter_triangle_1d")
    qc.compose(gate_U_enc_temp(connectivity=connectivity,
                            to_instruction=to_instruction,
                            add_barrier=add_barrier),
                qubits=[jth_qubit for jth_qubit in range(0, 3)],
                inplace=True,)
    qc.compose(gate_H_eff_1d(dt=dt,
                            to_instruction=to_instruction,
                            add_barrier=add_barrier),
                qubits=[jth_qubit for jth_qubit in range(0, 2)],
                inplace=True,)
    qc.compose(gate_U_dec_temp(connectivity=connectivity,
                            to_instruction=to_instruction,
                            add_barrier=add_barrier),
                qubits=[jth_qubit for jth_qubit in range(0, 3)],
                inplace=True,)
    if add_barrier:
        qc.barrier(label="block_trotter_triangle_1d")

    return qc.to_instruction(label="block_trotter_triangle_1d") if to_instruction else qc

