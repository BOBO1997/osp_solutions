from typing import *
import numpy as np
from pprint import pprint

### import qiskit library ###
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.result import Result, Counts #! deprecated 
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives import BitArray
from qiskit.providers import Job #! deprecated
from qiskit.result import QuasiDistribution
import qiskit.quantum_info as qi

from osp_solutions.backend_simulator import dms_to_hists, result_to_dms


### === retrieve jobs === ###


def get_hist_from_bitarray(bitarray: BitArray) -> Any:
    for field, value in bitarray.__dict__.items():
        if isinstance(value, BitArray):
            hist = value.get_counts()
            num_shots = sum(hist.values())
            return hist, num_shots
    return None


def get_qd_from_bitarray(bitarray: BitArray) -> Any:
    for field, value in bitarray.__dict__.items():
        if isinstance(value, BitArray):
            int_counts = value.get_int_counts()
            num_shots = sum(int_counts.values())
            qd = {key: val / num_shots for key, val in int_counts.items()}
            return qd, num_shots
    return None


def retrieve_jobs(service: QiskitRuntimeService = None,
                  ids_jobs: List[str] = None,
                  jobs: List[Job] = None,
                  results_of_jobs: List[PrimitiveResult] = None,
                 ) -> Dict[str, Any]:

    assert not ((ids_jobs is None) and (jobs is None) and (results_of_jobs is None))

    records_retrieve = dict()

    ### retrieve results ###
    # service = QiskitRuntimeService()
    list_of_hists = []
    list_of_qds = []
    list_of_nums_shots = []
    for ith_job, id_job in enumerate(ids_jobs):
        print("Retriving job (id:", id_job, ")")
        if jobs is not None:
            job = jobs[ith_job]
            results_of_each_job = job.result()
        elif results_of_jobs is None:
            job = service.job(id_job)
            results_of_each_job = job.result()
        else:
            pass

        hists_in_each_job = []
        qds_in_each_job = []
        nums_shots_in_each_job = []
        for result in results_of_each_job:
            hist, num_shots = get_hist_from_bitarray(result.data)
            qd, _ = get_qd_from_bitarray(result.data)
            hists_in_each_job.append(hist)
            qds_in_each_job.append(qd)
            nums_shots_in_each_job.append(num_shots)
        list_of_hists.append(hists_in_each_job)
        list_of_qds.append(qds_in_each_job)
        list_of_nums_shots.append(nums_shots_in_each_job)
    records_retrieve["list_of_hists"] = list_of_hists ### 2d array in the current implementation
    records_retrieve["list_of_qds"] = list_of_qds ### 2d array in the current implementation
    records_retrieve["list_of_nums_shots"] = list_of_nums_shots ### 2d array in the current implementation
    return records_retrieve


def retrieve_jobs_dm(ids_jobs: List[str] = None,
                     jobs: List[Job] = None,
                     paulis_shadow: List[qi.Pauli] = None,
                    ) -> Dict[str, Any]:
    
    assert paulis_shadow is not None

    records_retrieve = dict()

    ### retrieve results ###
    list_of_hists = []
    list_of_qds = []
    list_of_nums_shots = []
    for ith_job, id_job in enumerate(ids_jobs):
        print("Retriving job (id:", id_job, ")")
        job = jobs[ith_job]
        results_of_each_job = job.result()

        dms = result_to_dms(result=results_of_each_job,
                            endian_dm="big")
        hists_in_each_job = dms_to_hists(dms=dms,
                                         observables=paulis_shadow)
        qds_in_each_job = [QuasiDistribution(data=hist) for hist in hists_in_each_job] ###! check endian !###
        nums_shots_in_each_job = [1 for _ in range(len(qds_in_each_job))]

        list_of_hists.append(hists_in_each_job)
        list_of_qds.append(qds_in_each_job)
        list_of_nums_shots.append(nums_shots_in_each_job)
    records_retrieve["list_of_hists"] = list_of_hists ### 2d array in the current implementation
    records_retrieve["list_of_qds"] = list_of_qds ### 2d array in the current implementation
    records_retrieve["list_of_nums_shots"] = list_of_nums_shots ### 2d array in the current implementation
    return records_retrieve


### === process probability distribtions === ###


def quasi_distribution_to_counts(qd: QuasiDistribution,
                                 num_clbits: int,
                                 shots: int = 1,
                                ) -> Counts:
    """
    Equivalent to QuasiDistribution.binary_probabilities() method.
    So this function will be deprecated in the future.
    But before deprecating this, we have to check 
    whether the sum of distribution from QuasiDistribution.binary_probabilities() == 1 or == shots.
    """
    hist = Counts(dict())
    for key, value in qd.items():
        hist[format(key, "0"+str(num_clbits)+"b")] = value * shots
    return hist


def quasi_distribution_to_list_of_int_and_float(qd: QuasiDistribution,
                                                num_clbits: int,
                                                shots: int = 1,
                                                endian_input: str = "big",
                                                endian_output: str = "little"):
    """
    Each histogram is stored in List[Tuple[List[int], float]].
    """
    hist = []
    for key_int, value in qd.items():
        if endian_input == "big" and endian_output == "little" or endian_input == "little" and endian_output == "big":
            key_int_flipped = 0
            for i in range(num_clbits):
                key_int_flipped += (key_int >> i & 1) << (num_clbits - 1 - i)
            hist.append((key_int_flipped, value * shots)) ### flip endian ###
        else:
            hist.append((key_int, value * shots))
    return hist


def quasi_distribution_to_list_of_list_and_float(qd: QuasiDistribution,
                                                 num_clbits: int,
                                                 shots: int = 1,
                                                 endian_input: str = "big",
                                                 endian_output: str = "little") -> List[Tuple[List[int], float]]:
    """
    Each histogram is stored in List[Tuple[List[int], float]].
    """
    hist = []
    for key_int, value in qd.items():
        state_list = []
        for i in range(num_clbits):
            if endian_input == "big" and endian_output == "little" or endian_input == "little" and endian_output == "big":
                state_list.append(key_int >> i & 1)
            else:
                state_list.append(key_int >> (num_clbits - 1 - i) & 1)
        hist.append((state_list, value * shots))
    return hist


def reduce_labels(hist: List[Tuple[List[int], float]], 
                  num_bins_max=100):
    """
    sparsify the histogram
    """
    print("length of histogram: ", len(hist))
    hist_reduced = sorted(hist, key = lambda x: x[1])[::-1][:num_bins_max]
    sum_hist_reduced = sum([item[1] for item in hist_reduced])
    hist_reduced_normalised = [(item[0], item[1] / sum_hist_reduced) for item in hist_reduced]

    return hist_reduced_normalised


def flip_endian_of_hist_label(hist: Union[Counts, QuasiDistribution]) -> Union[Counts, QuasiDistribution]:
    if isinstance(hist, QuasiDistribution):
        return QuasiDistribution({k[::-1]: v for k, v in hist.items()})
    else:
        return Counts({k[::-1]: v for k, v in hist.items()})


def remove_space_from_hist_label(hist: Union[Counts, QuasiDistribution]) -> Union[Counts, QuasiDistribution]:
    """
    ### correctness is verified
    remove the space from the labels of histogram
    """
    if isinstance(hist, QuasiDistribution):
        hist_replaced = QuasiDistribution(dict())
    else:
        hist_replaced = Counts(dict())
    for state, count in hist.items():
        hist_replaced[state.replace(" ", "")] = count
    return hist_replaced


def reshape_hist(hist: Union[Counts, QuasiDistribution]) -> Union[Counts, QuasiDistribution]:
    """
    flip endian and remove space from labels
    """
    if isinstance(hist, QuasiDistribution):
        return QuasiDistribution({k[::-1]: v for k, v in remove_space_from_hist_label(hist).items()})
    else:
        return Counts({k[::-1]: v for k, v in remove_space_from_hist_label(hist).items()})


### used in main process ###
###! unused !###
def reshape_hists(result: Union[Result, List[Union[Counts, QuasiDistribution]]], 
                  from_wf: bool = False,
                 ) -> List[Union[Counts, QuasiDistribution]]:
    """
    ### correctness is verified
    Given the Result object, this function returns the list of "reshaped" histograms by removing the space in the state labels and reversing their order (endian).
    """
    # retrieve histograms and remove space
    hists = [remove_space_from_hist_label(hist) for hist in result.quasi_dists] if isinstance(result, Result) else result
    if from_wf:
        num_qcs = len(hists)
        n = len(next(iter(hists[0])))
        hists = []
        for i in range(num_qcs):
            wf = result.get_statevector(i)
            hist = dict()
            for i, prob in zip(range(1 << n), np.abs(wf) ** 2):
                hist[format(i, "0"+str(n)+"b")] = prob
            hists.append(hist)

    # reverse endian
    hists_little_endian = []
    for hist in hists:
        hists_little_endian.append({k[::-1]: v for k, v in hist.items()})
    return hists_little_endian


def make_hist_conditioned(hist: Union[Counts, QuasiDistribution],
                          condition_state: str, 
                          qr_poses: List[int],
                         ) -> Union[Counts, QuasiDistribution]:
    """
    Return the histogram conditioned by the conditioned state and its position
    all data here are supposed to be in little endian
    """
    num_clbits = len(next(iter(hist)))
    N_total = sum(hist.values())
    hist_conditioned = dict()
    # strlen = len(next(iter(hist)))
    for state, count in hist.items():
        state_conditioning = "".join([state[qr_pos] for qr_pos in qr_poses]).replace(" ", "")
        state_conditioned = "".join([state[qr_pos] for qr_pos in range(num_clbits) if qr_pos not in qr_poses]).replace(" ", "")
        if state_conditioning == condition_state:
            # only works for 1-qubit case. the ancillary qubit is assumed to be in the head of string
            hist_conditioned[state_conditioned] = count
    N_total_conditioned = sum(hist_conditioned.values())
    return hist_conditioned, N_total_conditioned / N_total



#############################################################################################
###
### functions for usual quantum circuits
###
#############################################################################################

#! this should not be here
def compute_expval_and_variance(hist: Union[Counts, QuasiDistribution]) -> Tuple[float, float]:
    N_total = sum(hist.values()) # .shots()
    sum_with_weight = 0
    for key, value in hist.items():
        if key.count("1") & 1 == 1:
            sum_with_weight -= value
        else:
            sum_with_weight += value
    expval = sum_with_weight / N_total
    variance = 1 - expval ** 2
    return expval, variance