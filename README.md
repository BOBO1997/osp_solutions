# Solutions for IBM Quantum Open Science Prize 2021

## Abstract

We simulate the time evolution of the N = 3 Heisenberg model on ibmq_jakarta with a modified Trotterization scheme based on the circuit level approach (without pulse). 
Focusing on the symmetry of the given Heisenberg Hamiltonian, we construct an effective Hamiltonian which acts on the smaller subspace. 
We show the Trotterization of this effective Hamiltonian is equivalent to changing the axis of the standard Trotterization of the original Hamiltonian. 
In the given problem setting with N = 3, this encoding framework makes it possible to drastically reduce the number of CNOT gates and the circuit depth into a constant through circuit optimization, regardless of the number of Trotter iterations. 
Combining with several error mitigation techniques, we finally achieve fidelity 0.9928 ± 0.0013 on ibmq_jakarta real quantum device, for the given problem setting.

## Methods

Please first refer to our [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

## To see the methods to output the figures in [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf)

Please see the [README.md](https://github.com/BOBO1997/osp_solutions/blob/main/experiments/README.md) in the [experiments](https://github.com/BOBO1997/osp_solutions/tree/main/experiments) directory.

## To Re-execute the programs or to check of previous results

Please see the [README.md](https://github.com/BOBO1997/osp_solutions/blob/main/solutions/README.md) in the [solutions](https://github.com/BOBO1997/osp_solutions/tree/main/solutions) directory.
The files in [solutions](https://github.com/BOBO1997/osp_solutions/tree/main/solutions) are the source of the table of results from the real quantum backedn in [report.pdf](https://github.com/BOBO1997/osp_solutions/blob/main/report.pdf).

## Authors

- Bo Yang: PhD candidate at Graduate School of Information Science and Technology, The University of Tokyo
- Naoki Negishi: PhD candidate at Graduate School of Arts and Sciences, The University of Tokyo

Many thanks to all the organizing staffs of for holding this event!
We are pleased to join this [IBM Quantum Open Science Prize 2021](https://ibmquantumawards.bemyapp.com/#/event).
We are happy to propose our efficient method to solve the simulation of time evolution on 3-qubit Heisenberg model.
This repository is the replication of our final submission with change logs.
Hope you enjoy our solution!
If you find any questions, please contact us!

## Links

- [Official GitHub repository](https://github.com/qiskit-community/open-science-prize-2021)
