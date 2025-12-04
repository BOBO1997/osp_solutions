from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter # , QuantumRegister


def gate_U_aj(dt, 
           option: str = "a",
           to_instruction: bool = True,
          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    based on https://github.com/qiskit-community/open-science-prize-2021
    """
    # Build a subcircuit for XX(t) two-qubit gate
    XX_qc = QuantumCircuit(2, name='XX')
    #
    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cx(0,1)
    XX_qc.rz(2 * dt, 1)
    XX_qc.cx(0,1)
    XX_qc.ry(-np.pi/2,[0,1])
    # Convert custom quantum circuit into a gate
    XX = XX_qc.to_instruction()

    # Build a subcircuit for YY(t) two-qubit gate
    YY_qc = QuantumCircuit(2, name='YY')
    #
    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cx(0,1)
    YY_qc.rz(2 * dt, 1)
    YY_qc.cx(0,1)
    YY_qc.rx(-np.pi/2,[0,1])
    # Convert custom quantum circuit into a gate
    YY = YY_qc.to_instruction()

    # Build a subcircuit for ZZ(t) two-qubit gate
    ZZ_qc = QuantumCircuit(2, name='ZZ')
    #
    ZZ_qc.cx(0,1)
    ZZ_qc.rz(2 * dt, 1)
    ZZ_qc.cx(0,1)
    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc.to_instruction()

    qc = QuantumCircuit(2)
    qc.append(instruction=ZZ, qargs=[0,1])
    qc.append(instruction=YY, qargs=[0,1])
    qc.append(instruction=XX, qargs=[0,1])

    return qc.to_instruction(label="U_aj") if to_instruction else qc


def gate_U_negishi(dt, 
                   option: str = "a",
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
    
    qc = QuantumCircuit(2)

    if option == "a":
        qc.cx(0, 1)

        qc.rz(- np.pi / 2, 0)
        qc.ry(- phi, 0)
        qc.rz(- theta, 1)

        qc.cx(1, 0)

        qc.ry(- theta, 0)
        
        qc.cx(0, 1)

        qc.rz(np.pi / 2, 1)

    elif option == "b":
        qc.cx(1, 0)

        qc.rz(- np.pi / 2, 1)
        qc.ry(- phi, 1)
        qc.rz(- theta, 0)

        qc.cx(0, 1)

        qc.ry(- theta, 1)
        
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)

    elif option == "c":
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
        qc.barrier()

    return qc.to_instruction(label="U_negishi") if to_instruction else qc


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
                     lmd: float = 0.0,
                     type_H_eff: str = None,
                     to_instruction: bool = True,
                     add_barrier: bool = False,
                    ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter, this corresponds to Delta t' in the arXiv paper: 2505.04552
    
    """

    gate_H_eff = gate_H_eff_new if type_H_eff is None else gate_H_eff_old

    qc = QuantumCircuit(5)

    if to_instruction:
        # qc.append(gate_U_negishi(dt=(1 - lmd) * dt,
        #                          option="c",
        #                          to_instruction=to_instruction,
        #                          add_barrier=add_barrier),
        #           qargs=[1, 2])
        qc.append(gate_U_negishi(dt=2 * dt,
                                 option="c",
                                 to_instruction=to_instruction,
                                 add_barrier=add_barrier),
                  qargs=[3, 4])
    else:
        # qc.compose(gate_U_negishi(dt=(1 - lmd) * dt,
        #                           option="c",
        #                           to_instruction=to_instruction,
        #                           add_barrier=add_barrier),
        #            qubits=[1, 2],
        #            inplace=True,)
        qc.compose(gate_U_negishi(dt=2 * dt,
                                  option="c",
                                  to_instruction=to_instruction,
                                  add_barrier=add_barrier), 
                   qubits=[3, 4],
                   inplace=True,)
        
    if add_barrier:
        qc.barrier()

    # qc.cx(3, 2)
    # qc.cx(2, 1)
    # qc.cx(3, 4)
    # qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 1)

    if add_barrier:
        qc.barrier()

    if to_instruction:
        qc.append(gate_H_eff(dt=(1 + lmd) * dt,
                             to_instruction=to_instruction,
                             add_barrier=add_barrier,
                            ),
                qargs=[1, 2],
                )
    else:
        qc.compose(gate_H_eff(dt=(1 + lmd) * dt,
                              to_instruction=to_instruction,
                              add_barrier=add_barrier,
                             ),
                   qubits=[1, 2],
                   inplace=True,
                  )

    # qc.cx(2, 1)
    # qc.cx(1, 0)
    # qc.cx(2, 3)
    # qc.cx(1, 2)
    qc.cx(3, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)

    if add_barrier:
        qc.barrier()

    if to_instruction:
        qc.append(gate_U_negishi(dt=2 * dt,
                                option="b",
                                to_instruction=to_instruction,
                                add_barrier=add_barrier), 
                  qargs=[0, 1])
        # qc.append(gate_U_negishi(dt=(1 - lmd) * dt,
        #                         option="b",
        #                         to_instruction=to_instruction,
        #                         add_barrier=add_barrier), 
        #           qargs=[2, 3])
    else:
        qc.compose(gate_U_negishi(dt=2 * dt,
                                  option="b",
                                  to_instruction=to_instruction,
                                  add_barrier=add_barrier), 
                   qubits=[0, 1],
                   inplace=True,)
        # qc.compose(gate_U_negishi(dt=(1 - lmd) * dt,
        #                           option="b",
        #                           to_instruction=to_instruction,
        #                           add_barrier=add_barrier),
        #            qubits=[2, 3],
        #            inplace=True,)
        
    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="proposed_2t") if to_instruction else qc


### ================================================== ###


def append_block_trotter_aj(qc: QuantumCircuit, 
                         dt,
                         num_steps: int, 
                         to_instruction: bool = True,
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
                    qc.append(gate_U_aj(dt=dt,
                                        to_instruction=to_instruction), 
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits - 1):
                if ith_qubit & 1: ### odd
                    qc.append(gate_U_aj(dt=dt,
                                        to_instruction=to_instruction), 
                              qargs = [ith_qubit, ith_qubit + 1])
    if num_qubits & 1: ### odd
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits - 1):
                if not ith_qubit & 1: ### even
                    qc.append(gate_U_aj(dt=dt,
                                        to_instruction=to_instruction),
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits):
                if ith_qubit & 1: ### odd
                    qc.append(gate_U_aj(dt=dt,
                                        to_instruction=to_instruction), 
                              qargs = [ith_qubit, ith_qubit + 1])
                    

def append_block_trotter_negishi(qc: QuantumCircuit, 
                                 dt,
                                 num_steps: int, 
                                 option: str = None,
                                 to_instruction: bool = True,
                                 add_barrier: bool = False,
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
                    qc.append(gate_U_negishi(dt=dt,
                                             option="b" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits - 1):
                if ith_qubit & 1: ### odd
                    qc.append(gate_U_negishi(dt=dt,
                                             option="c" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            if add_barrier:
                qc.barrier()

    if num_qubits & 1: ### odd
        for _ in range(num_steps):
            for ith_qubit in range(num_qubits - 1):
                if not ith_qubit & 1: ### even
                    qc.append(gate_U_negishi(dt=dt,
                                             option="b" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier),
                              qargs = [ith_qubit, ith_qubit + 1])
            for ith_qubit in range(num_qubits):
                if ith_qubit & 1: ### odd
                    qc.append(gate_U_negishi(dt=dt,
                                             option="c" if option is None else option,
                                             to_instruction=to_instruction,
                                             add_barrier=add_barrier), 
                              qargs = [ith_qubit, ith_qubit + 1])
            if add_barrier:
                qc.barrier()


def append_block_trotter_proposed(qc: QuantumCircuit, 
                                  dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                                  lmd: float,
                                  num_steps: int, ###! note that this is based on the counting of the original Trotter iteration
                                  type_H_eff: str = None,
                                  to_instruction: bool = True,
                                  add_barrier: bool = False,
                                 ) -> None:
    """
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
                                               lmd=lmd,
                                               type_H_eff=type_H_eff,
                                               to_instruction=to_instruction,
                                               add_barrier=add_barrier), 
                              qargs=list(range(ith_qubit, ith_qubit + 5)))
                else:
                    qc.compose(gate_proposed_2t(dt=dt, ###! dt follows the same scale as the arXiv paper: 2505.04552
                                                lmd=lmd,
                                                type_H_eff=type_H_eff,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier),
                               qubits=list(range(ith_qubit, ith_qubit + 5)),
                               inplace=True,)


### ================================================== ###
###! from here: unused !###
### ================================================== ###


def append_initial_state(qc: QuantumCircuit, 
                         state_initial_str: str,
                         add_barrier: bool = False,
                        ) -> None:
    """
    The state_initial_str is the string in little endian.
    """
    for i, state in enumerate(state_initial_str):
        if state == "1":
            qc.x(i)
    
    if add_barrier:
        qc.barrier()


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