### import libaray ###
from typing import *
import time
from pprint import pprint

### import qiskit library ###
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerOptions
from qiskit_ibm_runtime import SamplerV2 as Sampler ###! commented out for simulator, 2024.06.27 !###
from qiskit_aer.primitives import SamplerV2 as AerSampler
# from qiskit_ibm_runtime import Sampler
from qiskit_aer.noise import NoiseModel #! deprecated? qiskit.aer
# from qiskit_aer import AerSimulator #! deprecated? qiskit.aer
from qiskit_aer.backends.aer_simulator import AerSimulator #! deprecated? qiskit.aer
from qiskit.providers.fake_provider import *
from qiskit.providers import Backend, Job #! deprecated

# from osp_solutions.backend_simulator import DMExtended, qc_to_dm

def choose_backend_from_str(name_backend: str,
                            service: QiskitRuntimeService = QiskitRuntimeService(),
                            method_aer_simulator: str = "density_matrix",
                            options: SamplerOptions = None,
                           ) -> Any:
    ### set backend ###
    backend = None
    noise_model = None
    if name_backend[:4] == "Fake":
        fake_backend = globals()[name_backend]()
        noise_model = NoiseModel.from_backend(fake_backend)
        options.simulator = {
            "noise_model": noise_model,
            # "basis_gates": fake_backend.configuration().basis_gates,
            # "coupling_map": fake_backend.configuration().coupling_map,
            "seed_simulator": 42,
        }
        backend = AerSimulator(method=method_aer_simulator,
                               noise_model=noise_model) # service.backend("ibmq_qasm_simulator")
        return backend, noise_model
    elif name_backend == "aer_simulator" or "AerSimulator":
        backend = AerSimulator(method=method_aer_simulator)
        return backend, None
    else:
        backend = service.backend(name_backend)
        return backend, None


def send_jobs(list_qcs: List[List[QuantumCircuit]],
              service: QiskitRuntimeService = QiskitRuntimeService(),
              backend: Backend = None,
              options: SamplerOptions = None,
              reutrn_jobs: bool = False,
              num_qcs_max: int = 5000,
             ) -> Tuple[List[str], Any]:
    """
    adapted only for the finite shot counts.
    For inifinite shot counts you should you the function: .
    """
    ids_jobs = []
    jobs = []

    ###! patch: not using this case !###
    if isinstance(backend, AerSimulator): ### if using local simulator ###
        ### set sampler and run quantum circuits ###
        if options.default_shots > 0:
            sampler = AerSampler() ###? here, backend is not used ###
            for qcs in list_qcs:
                if len(qcs) > num_qcs_max:
                    print("CAUTION: You are going to run "+str(len(qcs))+">"+str(num_qcs_max)+" quantum circuits.")
                job = sampler.run(qcs) ###? sampler ###
                jobs.append(job)
                ids_jobs.append(job.job_id())
        else: ### denisty matrix simulator ###
            for qcs in list_qcs:
                if len(qcs) > num_qcs_max:
                    print("CAUTION: You are going to run "+str(len(qcs))+">"+str(num_qcs_max)+" quantum circuits.")
                job = backend.run(qcs) ###? here, backend is actually used ###
                jobs.append(job)
                ids_jobs.append(job.job_id())

    ###! always this case !###
    else: ### if using remote service ###
        with Session(service=service, backend=backend) as session:
            print("\nStart session\n")
            ### set sampler and run quantum circuits ###
            sampler = Sampler(session=session,
                              backend=backend,
                              options=options)
            for qcs in list_qcs:
                if len(qcs) > num_qcs_max:
                    print("CAUTION: You are going to run "+str(len(qcs))+">"+str(num_qcs_max)+" quantum circuits.")
                job = sampler.run(qcs)
                jobs.append(job)
                ids_jobs.append(job.job_id())

    if reutrn_jobs:
        return ids_jobs, jobs
    else:
        return ids_jobs, None # type: Tuple[List[str], List[Job]]


def make_qcs_qrem_YRU2022(num_qubits: int) -> List[QuantumCircuit]:
    qcs_qrem = []
    ### all zero circuits ###
    qr_qrem = QuantumRegister(num_qubits)
    cr_qrem = ClassicalRegister(num_qubits)
    qc_qrem = QuantumCircuit(qr_qrem, cr_qrem, name="qrem_all_zero")
    qc_qrem.measure(qr_qrem, cr_qrem)
    qcs_qrem.append(qc_qrem)

    ### all one circuits ###
    qr_qrem = QuantumRegister(num_qubits)
    cr_qrem = ClassicalRegister(num_qubits)
    qc_qrem = QuantumCircuit(qr_qrem, cr_qrem, name="qrem_all_one")
    qc_qrem.x(qr_qrem) ### all one circuits ###
    qc_qrem.measure(qr_qrem, cr_qrem)
    qcs_qrem.append(qc_qrem)
    
    return qcs_qrem


def initialise_parameters_backend(**kwargs):
    parameters_backend = dict(sorted(kwargs.items()))
    return parameters_backend