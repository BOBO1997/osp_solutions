from typing import *
from qiskit.circuit import QuantumCircuit

def rename_qcs_StateTomography_to_ignis(circuits: List[QuantumCircuit]):
    """
    Rename Qiskit Experiments-style StateTomography circuits to
    Ignis-compatible names for StateTomographyFitter.
    
    Input:
        circuits (list[QuantumCircuit])
    Output:
        same list, with .name updated to Ignis literal format ('Z','X','Y')
    """
    import re
    
    pauli_map = {0: 'Z', 1: 'X', 2: 'Y'}
    
    for circ in circuits:
        name = circ.name
        
        # Check for the typical Experiments format: "StateTomography_(...)" or similar
        m = re.search(r'\((.*?)\)', name)
        if m:
            try:
                # "(0, 1, 2)" → (0, 1, 2)
                index_tuple = tuple(int(x.strip()) for x in m.group(1).split(','))
                # Convert index → Pauli label
                pauli_tuple = tuple(pauli_map[i] for i in index_tuple)
                # Set Ignis-compatible name: e.g. "('Z','X','Y')"
                circ.name = str(pauli_tuple)
            except Exception:
                # if unexpected format, keep original name
                pass
    
    return circuits