# Solutions for IBM Quantum Open Science Prize 2021

## Abstract

We simulate the time evolution of the $N=3$ Heisenberg model by Tritterization on `ibmq_jakarta`.
Our approach is to encode the $N$ spin site into $N-1$-qubit space by constructing an equivalent effective Hamiltonian from the original Hamiltonian.
Under the encoded space, the time evolution is simulated by the Trotterization of the effective Hamitonian, and decoded to $N$-qubit system at the end.
This embedding framework makes it possible to reduce the number of CNOT gates in the quantum circuit from $6$ to $4$ for one Trotter step.
In particular, when $N$ is 3, the circuit depth and the number of CNOT gates becomes constant for any trotter steps.
Combining with several error mitigation techniques, we finally achieve fidelity $0.94$ for the time evolution from $t=0$ to $t=\pi$ of the $N=3$ Heisenberg model.

## Methods

Please refer to our [report](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf)

## Directory Structure

