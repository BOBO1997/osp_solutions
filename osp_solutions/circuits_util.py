from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter


### ================================================== ###

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


### ================================================== ###


def gate_U_enc(connectivity: str = "complete",
               to_instruction: bool = True,
               add_barrier: bool = False,
              ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for blue and red
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
    elif connectivity == "path":
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(1, 2)
    else:
        raise Exception("invalid connectivity")

    if add_barrier:
        qc.barrier(label="U_enc")

    return qc.to_instruction(label="U_enc") if to_instruction else qc


def gate_U_dec(connectivity: str = "complete",
               to_instruction: bool = True,
               add_barrier: bool = False,
              ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for blue and red
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(2, 0)
        qc.cx(1, 2)
        qc.cx(0, 1)
    elif connectivity == "path":
        qc.cx(1, 2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(0, 1)
    else:
        raise Exception("invalid connectivity")
    
    if add_barrier:
        qc.barrier(label="U_dec")

    return qc.to_instruction(label="U_dec") if to_instruction else qc


def gate_U_enc_H(connectivity: str = "complete",
                 to_instruction: bool = True,
                 add_barrier: bool = False,
                ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for yellow and green
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(0, 2)
    elif connectivity == "path":
        qc.cx(1, 0)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 1)
    else:
        raise Exception("invalid connectivity")

    if add_barrier:
        qc.barrier(label="U_enc_H")

    return qc.to_instruction(label="U_enc_H") if to_instruction else qc


def gate_U_dec_H(connectivity: str = "complete",
                 to_instruction: bool = True,
                 add_barrier: bool = False,
                ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for yellow and green
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(1, 0)
    elif connectivity == "path":
        qc.cx(2, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(1, 0)
    else:
        raise Exception("invalid connectivity")
    
    if add_barrier:
        qc.barrier(label="U_dec_H")

    return qc.to_instruction(label="U_dec_H") if to_instruction else qc


### ================================================== ###


def gate_2xx_zz(dt: float,
                to_instruction: bool = True,
                add_barrier: bool = False,
               ) -> Union[QuantumCircuit, Instruction]:
    """
    exp(-i(2XX+ZZ)dt), used in the proposed method
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rx(4 * dt, 0)
    qc.rz(2 * dt, 1)
    qc.cx(0, 1)

    ###! force not to add barrier
    # if add_barrier:
    #     qc.barrier()

    return qc.to_instruction(label="2xx_zz") if to_instruction else qc


def gate_H_eff(dt: float,
               to_instruction: bool = True,
               add_barrier: bool = False,
              ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    qc = QuantumCircuit(2)

    ### V_enc
    qc.ry(- 1 * np.pi / 4, 0) ###! cos(\pi/8) + i sin (\pi/8 Y)
    qc.ry(- 1 * np.pi / 4, 1) ###! cos(\pi/8) + i sin (\pi/8 Y)
    
    ### further trotter, exp(-i (Z_1 + Z_2) t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 0) ### exp(-i Z_1 t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 1) ### exp(-i Z_2 t / sqrt(2))

    qc.sxdg(0) ### sqrt(X)^{dag}
    qc.sxdg(1) ### sqrt(X)^{dag}

    ###! force not to add barrier
    # if add_barrier:
    #     qc.barrier()
    
    if to_instruction:
        qc.append(gate_2xx_zz(dt=dt,
                              to_instruction=to_instruction,
                              add_barrier=add_barrier),
                  qargs=[0, 1])
    else:
        qc.compose(gate_2xx_zz(dt=dt,
                               to_instruction=to_instruction,
                               add_barrier=add_barrier),
                   qubits=[0, 1],
                   inplace=True,)
    
    qc.sx(0) ### sqrt(X)
    qc.sx(1) ### sqrt(X)

    ### further trotter, exp(-i (Z_1 + Z_2) t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 0) ### exp(-i Z_1 t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 1) ### exp(-i Z_2 t / sqrt(2))
    
    ### V_dnc = V_enc^{dag}
    qc.ry(1 * np.pi / 4, 0) ###! cos(\pi/8) - i sin (\pi/8 Y)
    qc.ry(1 * np.pi / 4, 1) ###! cos(\pi/8) - i sin (\pi/8 Y)

    if add_barrier:
        qc.barrier(label="H_eff")

    return qc.to_instruction(label="H_eff") if to_instruction else qc


### ================================================== ###


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
        qc.append(gate_block_trotter_3cnot(dt=dt, #! * 2, ###!
                                           option="a", # "d",
                                           to_instruction=to_instruction,
                                           add_barrier=add_barrier),
                  qargs=[3, 4])
    else:
        qc.compose(gate_block_trotter_3cnot(dt=dt, #! * 2, ###!
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
        qc.append(gate_block_trotter_3cnot(dt=dt, #! * 2, ###!
                                           option="a", # "c",
                                           to_instruction=to_instruction,
                                           add_barrier=add_barrier), 
                  qargs=[0, 1])
    else:
        qc.compose(gate_block_trotter_3cnot(dt=dt, #! * 2, ### !
                                            option="a", # "c",
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                   qubits=[0, 1],
                   inplace=True,)
        
    if add_barrier:
        qc.barrier()

    return qc.to_instruction(label="proposed_2t") if to_instruction else qc

