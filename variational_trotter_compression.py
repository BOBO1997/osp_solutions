from typing import Union, List, Any
from cirq import fidelity

import numpy as np
import itertools
import cma

from qiskit import QuantumCircuit, QuantumRegister
import mitiq

class VTC:
    def __init__(self, args) -> None:
        """
        This class implements the variational Trotter compression: https://arxiv.org/abs/2112.12654
        args: dictionary of arguments
        """

        # unfold arguments
        self.num_qubits: int = args["num_qubits"]
        self.system_size: int = args["system_size"]
        self.target_time: float = args["target_time"]
        self.trotter_steps: int = args["trotter_steps"]
        self.time_interval: float = self.target_time / self.trotter_steps
        self.ansatz_depth: int = args["ansatz_depth"]
        self.shots: int = args["shots"]
        self.init_state: Union[str, QuantumCircuit] = args["init_state"]
        self.initial_layout = args["initial_layout"]


    # とりあえずOK
    def evolve(self, 
               alpha: float, 
               q0: Union[int, QuantumRegister], 
               q1: Union[int, QuantumRegister]) -> QuantumCircuit:
        """
        The implementation of Fig. 4 in https://arxiv.org/abs/2112.12654 
        """
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, q1)
        qc.cnot(q1, q0)
        qc.rz(alpha - np.pi / 2, q0)
        qc.ry(np.pi / 2 - alpha, q1)
        qc.cnot(q0, q1)
        qc.ry(alpha - np.pi / 2, q1)
        qc.cnot(q1, q0)
        qc.rz(np.pi / 2, q0)
        return qc


    # とりあえずOK
    def make_ansatz_circuit(self, parameters: np.array) -> QuantumCircuit:
        """
        Prepare ansatz circuit
        code reference: https://gitlab.com/QANED/heis_dynamics
        method reference: https://arxiv.org/abs/1906.06343
        Args:
            parameters: 1d array (for 2d parameters on circuit)
        """
        qc = QuantumCircuit(self.num_qubits)
        for l in range(self.ansatz_depth):
            if self.num_qubits & 1:
                for i in range(0, self.num_qubits, 2):  # linear condition
                    qc.compose(self.evolve(parameters[l * self.ansatz_depth + i] / 4), [i, i + 1], inplace = True) #! we do not have to divide the angle by 4
                for i in range(1, self.num_qubits - 1, 2):  # linear condition
                    # ! we do not have to divide the angle by 4
                    qc.compose(self.evolve(parameters[l * self.ansatz_depth + i] / 4), [i, i + 1], inplace=True)
            else:
                for i in range(0, self.num_qubits - 1, 2):  # linear condition
                    # ! we do not have to divide the angle by 4
                    qc.compose(self.evolve(parameters[l * self.ansatz_depth + i]), [i, i + 1], inplace=True)
                for i in range(1, self.num_qubits - 1, 2):  # linear condition
                    # ! we do not have to divide the angle by 4
                    qc.compose(self.evolve(parameters[l * self.ansatz_depth + i]), [i, i + 1], inplace=True)
        return qc
        

    # とりあえずOK
    def make_trotter_circuit(self) -> QuantumCircuit:
        """ 
        Prepare Trotter circuit
        code reference: https://gitlab.com/QANED/heis_dynamics
        method reference: https://arxiv.org/abs/1906.06343
        """
        qc = QuantumCircuit(self.num_qubits)
        for n in range(self.trotter_steps): #! self.time_interval の符号に注意
            if self.num_qubits & 1:
                for i in range(0, self.num_qubits, 2):  # linear condition
                    qc.compose(self.evolve(self.time_interval / 4), [i, i + 1], inplace = True) #! we do not have to divide the angle by 4
                for i in range(1, self.num_qubits - 1, 2):  # linear condition
                    qc.compose(self.evolve(self.time_interval / 4), [i, i + 1], inplace = True)  # ! we do not have to divide the angle by 4
            else:
                for i in range(0, self.num_qubits - 1, 2):  # linear condition
                    qc.compose(self.evolve(self.time_interval / 4), [i, i + 1], inplace = True) #! we do not have to divide the angle by 4
                for i in range(1, self.num_qubits - 1, 2):  # linear condition
                    qc.compose(self.evolve(self.time_interval / 4), [i, i + 1], inplace = True)  # ! we do not have to divide the angle by 4
        return qc


    def zne_encoder(self, circuits):
        """
        """
        scale_factors = [1.0, 2.0, 3.0]  # ! ZNEのノイズスケーリングパラメタ
        folded_circuits = []  # ! ZNE用の回路
        for circuit in circuits:
            folded_circuits.append([mitiq.zne.scaling.fold_gates_at_random(circuit, scale) for scale in scale_factors])  # ! ここでmitiqを使用
        folded_circuits = list(itertools.chain(*folded_circuits))  # ! folded_circuitsを平坦化
        # ! 後からPauli Twirlingを施す!
        folded_circuits = [TwirlCircuit(circ) for circ in folded_circuits]
    
    def zne_decoder(self):
        """
        """
    
    def qrem_encoder():
        """
        """

    def qrem_decoder():
        """
        """

    def execute():
        """
        """



    def compute_fidelity(self, source_parameters, target_parameters):
        """
        """


    
    def train_one_step(self, source_parameters):
        """
        1時間間隔分: self.time_interval だけ次の時間の状態に進める
        """
        init_parameters = np.random.uniform(0, 2 * np.pi, self.num_qubits * self.ansatz_depth)
        es = cma.CMAEvolutionStrategy(init_parameters, np.pi / 2)
        es.opts.set({'ftarget': 5e-3, 'maxiter': 1000})
        while not es.stop():  # ! 最適化パート
            # ! 25 = number of returned solutions
            target_parameters = es.ask(25)
            sth = self.execute(source_parameters, target_parameters)
            fidelities = self.compute_fidelity()
            es.tell(target_parameters, fidelities)  # ! 実行パート
            es.disp()
            open(f'./results_{L}/optimizer_dump','wb').write(es.pickle_dumps())
        return es.result_pretty()

    def run(self, backend, shots=None):
        """
        """
        self.backend = backend
        if shots is not None:
            self.shots = shots
        
        parameters_list = []
        
        for step in range(self.trotter_steps): #! temporary value
            trained_ansatz_parameters = self.train_one_step(np.zeros(self.num_qubits * self.ansatz_depth)) #! random parameterにするべき
            parameters_list.append(trained_ansatz_parameters)

