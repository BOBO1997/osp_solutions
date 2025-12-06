from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter # , QuantumRegister


def gate_block_trotter_qiskit(dt: float, 
                              to_instruction: bool = True,
                              add_barrier: bool = False,
                             ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    qc = QuantumCircuit(2,
                        name="block_trotter_qiskit")
    qc.rzz(theta=2 * dt,
           qubit1=0,
           qubit2=1)
    qc.ryy(theta=2 * dt,
           qubit1=0,
           qubit2=1)
    qc.rxx(theta=2 * dt,
           qubit1=0,
           qubit2=1)

    if add_barrier:
        qc.barrier(label="block_trotter_qiskit")

    return qc.to_instruction(label="block_trotter_qiskit") if to_instruction else qc


def gate_block_trotter_6cnot(dt: float, 
                             to_instruction: bool = True,
                             add_barrier: bool = False,
                            ) -> Union[QuantumCircuit, Instruction]:
    """
    ### ! NOT RECOMMENDED ! ### use gate_block_trotter_qiskit instead.
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    based on https://github.com/qiskit-community/open-science-prize-2021
    """
    
    # Build a subcircuit for XX(t) two-qubit gate
    qc_XX = QuantumCircuit(2, name='XX')
    qc_XX.ry(np.pi/2,[0,1])
    qc_XX.cx(0,1)
    qc_XX.rz(2 * dt, 1)
    qc_XX.cx(0,1)
    qc_XX.ry(-np.pi/2,[0,1])

    # Build a subcircuit for YY(t) two-qubit gate
    qc_YY = QuantumCircuit(2, name='YY')
    qc_YY.rx(np.pi/2,[0,1])
    qc_YY.cx(0,1)
    qc_YY.rz(2 * dt, 1)
    qc_YY.cx(0,1)
    qc_YY.rx(-np.pi/2,[0,1])

    # Build a subcircuit for ZZ(t) two-qubit gate
    qc_ZZ = QuantumCircuit(2, name='ZZ')
    qc_ZZ.cx(0,1)
    qc_ZZ.rz(2 * dt, 1)
    qc_ZZ.cx(0,1)

    qc = QuantumCircuit(2,
                        name="block_trotter_6cnot")
    qc.append(instruction=qc_ZZ.to_instruction(), qargs=[0,1])
    qc.append(instruction=qc_YY.to_instruction(), qargs=[0,1])
    qc.append(instruction=qc_XX.to_instruction(), qargs=[0,1])

    if add_barrier:
        qc.barrier(label="block_trotter_6cnot")

    return qc.to_instruction(label="block_trotter_6cnot") if to_instruction else qc


def gate_block_trotter_3cnot(dt: float,
                             option: str = "b",
                             to_instruction: bool = True,
                             add_barrier: bool = False,
                            ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    theta = np.pi / 2 - 2 * dt
    phi = - theta

    theta_negative_t = np.pi / 2 + 2 * dt
    phi_negative_t = - theta_negative_t
    
    qc = QuantumCircuit(2,
                        name="block_trotter_3cnot")

    if option == "a":
        qc.cx(1, 0)

        qc.rz(- theta, 0)
        qc.rz(- np.pi / 2, 1)
        qc.ry(- phi, 1)

        qc.cx(0, 1)

        qc.ry(- theta, 1)
        
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)

    elif option == "b":
        qc.cx(0, 1)

        qc.rz(- np.pi / 2, 0)
        qc.ry(- phi, 0)
        qc.rz(- theta, 1)

        qc.cx(1, 0)

        qc.ry(- theta, 0)
        
        qc.cx(0, 1)

        qc.rz(np.pi / 2, 1)

    elif option == "c":
        qc.cx(1, 0)

        qc.rz(- np.pi / 2, 1)
        qc.ry(- phi, 1)
        qc.rz(- theta, 0)

        qc.cx(0, 1)

        qc.ry(- theta, 1)
        
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)

    elif option == "d":
        qc.rz(- np.pi / 2, 1)

        qc.cx(0, 1)

        qc.ry(theta_negative_t, 0)

        qc.cx(1, 0)

        qc.rz(theta_negative_t, 1)
        qc.ry(phi_negative_t, 0)
        qc.rz(np.pi / 2, 0)

        qc.cx(0, 1)
        
    else:
        raise Exception
    
    if add_barrier:
        qc.barrier(label="block_trotter_3cnot")

    return qc.to_instruction(label="block_trotter_3cnot") if to_instruction else qc


def gate_H_eff_old(dt,
                   to_instruction: bool = True,
                   add_barrier: bool = False,
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

    qc.rz(2 * dt, 0)
    qc.h(1)

    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="H_eff_old") if to_instruction else qc


def gate_H_eff_new(dt,
                   to_instruction: bool = True,
                   add_barrier: bool = False,
                  ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    qc = QuantumCircuit(2)

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


def gate_proposed_2t(dt, ###! dt follows the same scale as the arXiv paper
                     type_H_eff: str = None,
                     to_instruction: bool = True,
                     add_barrier: bool = False,
                    ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter, this corresponds to Delta t' in the arXiv paper: 2505.04552
    
    """

    if type_H_eff == "new":
        gate_H_eff = gate_H_eff_new
    elif type_H_eff == "old":
        gate_H_eff = gate_H_eff_old
    else:
        raise Exception("specify new or old")

    qc = QuantumCircuit(5)

    if to_instruction:
        qc.append(gate_block_trotter_3cnot(dt=dt * 2, ###!
                                           option="a", # "d",
                                           to_instruction=to_instruction,
                                           add_barrier=add_barrier),
                  qargs=[3, 4])
    else:
        qc.compose(gate_block_trotter_3cnot(dt=dt * 2, ###!
                                            option="a", # "d",
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

    if to_instruction:
        qc.append(gate_H_eff(dt=dt, # (1 + lmd) * dt,
                             to_instruction=to_instruction,
                             add_barrier=add_barrier,
                            ),
                qargs=[1, 2],
                )
    else:
        qc.compose(gate_H_eff(dt=dt, # (1 + lmd) * dt,
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

    if to_instruction:
        qc.append(gate_block_trotter_3cnot(dt=dt * 2, ###!
                                           option="a", # "c",
                                           to_instruction=to_instruction,
                                           add_barrier=add_barrier), 
                  qargs=[0, 1])
    else:
        qc.compose(gate_block_trotter_3cnot(dt=dt * 2, ### !
                                            option="a", # "c",
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                   qubits=[0, 1],
                   inplace=True,)
        
    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="proposed_2t") if to_instruction else qc


### ================================================== ###


###! unused
def append_path_qiskit(qc: QuantumCircuit, 
                       dt,
                       num_steps: int, 
                       to_instruction: bool = True,
                      ) -> None:
    """
    ###! unused
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter block for \Delta t
    """
    num_qubits = qc.num_qubits
    
    if not (num_qubits & 1): ### even
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits):
                if not ith_qubit & 1: ### even
                    qc.append(gate_block_trotter_qiskit(dt=dt,
                                        to_instruction=to_instruction), 
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits - 1):
                if ith_qubit & 1: ### odd
                    qc.append(gate_block_trotter_qiskit(dt=dt,
                                        to_instruction=to_instruction), 
                              qargs = [ith_qubit, ith_qubit + 1])
    if num_qubits & 1: ### odd
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits - 1):
                if not ith_qubit & 1: ### even
                    qc.append(gate_block_trotter_qiskit(dt=dt,
                                        to_instruction=to_instruction),
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits):
                if ith_qubit & 1: ### odd
                    qc.append(gate_block_trotter_qiskit(dt=dt,
                                        to_instruction=to_instruction), 
                              qargs = [ith_qubit, ith_qubit + 1])


###! old -> added gate_path_qiskit
def append_path_qiskit(qc: QuantumCircuit, 
                             dt,
                             num_steps: int, 
                             option: str = None,
                             to_instruction: bool = True,
                             add_barrier: bool = False,
                            ) -> None:
    """
    ###! old -> added gate_path_qiskit
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter block for \Delta t
    """
    num_qubits = qc.num_qubits
    
    if not num_qubits & 1: ### even
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits):
                if not ith_qubit & 1: ### even
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "c" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits - 1):
                if ith_qubit & 1: ### odd
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "d" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            if add_barrier:
                qc.barrier()

    if num_qubits & 1: ### odd
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits - 1):
                if not ith_qubit & 1: ### even
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "c" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier),
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits):
                if ith_qubit & 1: ### odd
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "d" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            if add_barrier:
                qc.barrier()


def gate_path_qiskit(num_qubits: int,
                           num_steps: int,
                           dt: float,
                           option: str = None,
                           to_instruction: bool = True,
                           add_barrier: bool = False,
                          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter block for \Delta t
    """
    qc = QuantumCircuit(num_qubits,
                        name="path_qiskit")
    
    if not (num_qubits & 1): ### even
        for ith_step in range(num_steps):
            for ith_qubit in range(num_qubits):
                if not ith_qubit & 1: ### even
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "c" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits - 1):
                if ith_qubit & 1: ### odd
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "d" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            if add_barrier:
                qc.barrier(label=str(ith_step+1)+"-th iteration")

    if num_qubits & 1: ### odd
        for ith_step in range(num_steps):
            for ith_qubit in range(num_qubits - 1):
                if not ith_qubit & 1: ### even
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "c" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier),
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits):
                if ith_qubit & 1: ### odd
                    qc.append(gate_block_trotter_3cnot(dt=dt,
                                             option="a", # "d" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            if add_barrier:
                qc.barrier(label=str(ith_step+1)+"-th iteration")

    return qc.to_instruction(label="path_qiskit") if to_instruction else qc


###! old -> added gate_path_proposed
def append_path_proposed(qc: QuantumCircuit, 
                         dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                         num_steps: int, ###! note that this is based on the counting of the original Trotter iteration
                         type_H_eff: str = None,
                         to_instruction: bool = True,
                         add_barrier: bool = False,
                        ) -> None:
    """
    ###! old -> added gate_path_proposed
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 2\Delta t
    """
    assert not (num_steps & 1)
    num_qubits = qc.num_qubits
    assert (num_qubits - 1) % 4 == 0

    for _ in range(num_steps // 2): ###! halved here: since gate_proposed_2t merges two original Trotter iterations
        for ith_qubit in range(num_qubits - 2):
            if ith_qubit % 4 == 0: ### even
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


def gate_path_proposed(num_qubits: int,
                       num_steps: int, ###! note that this is based on the counting of the original Trotter iteration
                       dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                       type_H_eff: str = None,
                       to_instruction: bool = True,
                       add_barrier: bool = False,
                      ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Proposed Trotter block for 2\Delta t
    """
    assert not (num_steps & 1)
    assert (num_qubits - 1) % 4 == 0

    qc = QuantumCircuit(num_qubits,
                        name="path_proposed")

    for ith_step in range(num_steps // 2): ###! halved here: since gate_proposed_2t merges two original Trotter iterations
        for ith_qubit in range(num_qubits - 2):
            if ith_qubit % 4 == 0: ### even
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


###! old -> added gate_initial_state
def append_initial_state(qc: QuantumCircuit, 
                         state_initial: str,
                         endian_state_initial: str = "big",
                         add_barrier: bool = False,
                        ) -> None:
    """
    ###! old -> added gate_initial_state
    The state_initial is the string in big endian.
    """
    if endian_state_initial == "little":
        state_initial = state_initial[::-1]

    for i, state in enumerate(state_initial):
        if state == "1":
            qc.x(i)
    
    if add_barrier:
        qc.barrier()


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


###! ================================================== ###
###! from here: unused !###
###! ================================================== ###


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