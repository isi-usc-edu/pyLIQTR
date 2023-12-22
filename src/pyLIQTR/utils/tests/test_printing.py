"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

import cirq
import pkg_resources
import pytest
from pathlib import Path
import numpy as np

#lets generate some circuits

from    pyLIQTR.ProblemInstances.getInstance         import   getInstance
from    pyLIQTR.clam.lattice_definitions             import   CubicLattice, SquareLattice, TriangularLattice
from    pyLIQTR.BlockEncodings.getEncoding           import   getEncoding, VALID_ENCODINGS

from    pyLIQTR.qubitization.qsvt_dynamics           import   qsvt_dynamics, simulation_phases
from    pyLIQTR.qubitization.qubitized_gates         import   QubitizedWalkOperator
from    pyLIQTR.qubitization.phase_estimation        import   QubitizedPhaseEstimation

from    pyLIQTR.utils.printing                       import   openqasm
from    openfermion.chem                             import  MolecularData
import os
# from    openfermionpyscf                             import  run_pyscf - not available on windows, uncomment for other platforms


@pytest.fixture(scope="module")
def script_loc(request):
    '''Return the directory of the currently running test script'''

    return request.path.parent

class TestPrinting:
    @pytest.fixture(scope="class")
    def fermi_hubbard_dynamics(self):
        J      = -1.0;          N      =     3     
        U      =  4.0;          shape  =  (N,N)

        model  =  getInstance('FermiHubbard',shape=shape, J=J, U=U, cell=SquareLattice)

        times       =  1
        # times       =  10 * N
        eps         =  1e-1
        # eps         =  1e-3
        phases      =  simulation_phases(times,eps=eps)

        gate_qsvt   =   qsvt_dynamics( encoding=getEncoding(VALID_ENCODINGS.FermiHubbardSquare),
                               instance=model,
                               phase_sets=phases )
        yield gate_qsvt.circuit

    @pytest.fixture(scope="class")
    def heisenberg_phase_estimation(self):
        N=3
        N_prec = 4

        J_x = J_y = -0.5;                J_z = -1.0
        h_x = 1.0;      h_y = 0.0;       h_z = 0.5

        model  =  getInstance( "Heisenberg", 
                            shape=(N,N), 
                            J=(J_x,J_y,J_z), 
                            h=(h_x,h_y,h_z), 
                            cell=SquareLattice )

        gate_gsee = QubitizedPhaseEstimation( getEncoding(VALID_ENCODINGS.PauliLCU),
                                      instance=model,prec=N_prec )
        
        yield gate_gsee.circuit

    @pytest.fixture(scope="class")
    def electronic_structure_LinearT_encoding(self,script_loc):
        example_ham_filename = script_loc.joinpath('example.ham.hdf5')
        example_grid_filename = script_loc.joinpath('example.grid.hdf5')

        model  =  getInstance('ElectronicStructure',
                                filenameH=example_ham_filename,
                                filenameG=example_grid_filename)

        encoding = getEncoding(VALID_ENCODINGS.LinearT, 
                                instance=model, 
                                approx_error=0.001,
                                control_val=1)
        
        yield encoding.circuit
        
    # Marking these as skipped until we figure out how to protect the Windows users
    @pytest.mark.skip
    @pytest.fixture(scope="class")
    def chemical_phase_estimation(self):
        mol_data = MolecularData([('H', (0.0, 0.0, 0.63164)), ('H', (0.0, 0.0, 1.76836))],\
                                 'sto-3g', 1.0, 0, 'H2')
        mol = run_pyscf(mol_data, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0, verbose=0)
        mol_ham = mol.get_molecular_hamiltonian()
        
        model = getInstance("ChemicalHamiltonian",mol_ham=mol_ham,mol_name="H2")
        lam       =   model.lam
        delta_e   =   0.001
        m_base = (np.sqrt(2) * np.pi * lam) / (2 * delta_e)
        m = int(np.ceil(np.log(m_base)))
        
        encoding = getEncoding(VALID_ENCODINGS.PauliLCU,instance=model,prec=m)
        
        yield encoding.circuit
        
    def test_electronic_structure(self, electronic_structure_LinearT_encoding):
        gqm = cirq.GreedyQubitManager(prefix="_ancilla", maximize_reuse=False)
        context = cirq.DecompositionContext(gqm)

        for line in openqasm(electronic_structure_LinearT_encoding, context=context):
            pass
        for line in openqasm(electronic_structure_LinearT_encoding, rotation_allowed=False, context=context):
            pass

    def test_electronic_structure_context(self,electronic_structure_LinearT_encoding):
        gqm = cirq.GreedyQubitManager(prefix="_ancilla", maximize_reuse=False)
        context = cirq.DecompositionContext(gqm)

        qasm = openqasm(electronic_structure_LinearT_encoding, rotation_allowed=True, context=context)
        assert qasm is not None


    def test_heisenberg(self, heisenberg_phase_estimation):
        for line in openqasm(heisenberg_phase_estimation):
            pass
        for line in openqasm(heisenberg_phase_estimation,rotation_allowed=False):
            pass
    
    def test_fermi_hubbard(self, fermi_hubbard_dynamics):
        for line in openqasm(fermi_hubbard_dynamics,rotation_allowed=True):
            pass
        for line in openqasm(fermi_hubbard_dynamics,rotation_allowed=False):
            pass
        
    @pytest.mark.skip
    def test_chemical_phase_estimation(self, chemical_phase_estimation):
        for line in openqasm(chemical_phase_estimation,rotation_allowed=True):
            pass
        for line in openqasm(chemical_phase_estimation,rotation_allowed=False):
            pass

    def test_fermi_hubbard_context(self, fermi_hubbard_dynamics):
        gqm = cirq.GreedyQubitManager(prefix="_ancilla", maximize_reuse=True)
        context = cirq.DecompositionContext(gqm)

        qasm = openqasm(fermi_hubbard_dynamics, rotation_allowed=True, context=context)
        assert qasm is not None

    def test_fermi_hubbard_simple_context(self, fermi_hubbard_dynamics):
        sqm = cirq.SimpleQubitManager(prefix="_ancilla")
        context = cirq.DecompositionContext(sqm)

        qasm = openqasm(fermi_hubbard_dynamics, rotation_allowed=True, context=context)
        assert qasm is not None
        
    @pytest.mark.skip
    def test_chemical_phase_estimation_context(self, chemical_phase_estimation):
        gqm = cirq.GreedyQubitManager(prefix="_ancilla", maximize_reuse=True)
        context = cirq.DecompositionContext(gqm)
        
        qasm = openqasm(chemical_phase_estimation, rotation_allowed=True, context=context)
        assert qasm is not None

    def test_fermi_hubbard_context_error(self, fermi_hubbard_dynamics):
        assert openqasm(fermi_hubbard_dynamics, rotation_allowed=True) is not None


    def test_get_attr_qasm(self, fermi_hubbard_dynamics):
        from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, Iterable
        from pyLIQTR.utils.printing import circuit_decompose_multi, _build_qasm_qubit_map

        og_circuit = cirq.align_left(fermi_hubbard_dynamics)
        decomposed_circuit = circuit_decompose_multi(og_circuit,1)
        circuit = decomposed_circuit
        qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit)
        RaiseTypeErrorIfNotProvided: Any = ([],)
        default = RaiseTypeErrorIfNotProvided
        for moment in circuit:
            for op in moment:
                method = getattr(op, '_qasm_', None)
                assert method is not None
                result = NotImplementedError
                if method is not None:
                    kwargs: Dict[str, Any] = {}
                    if qasm_args is not None:
                        kwargs['args'] = qasm_args
                    # if qubits is not None:
                    #     kwargs['qubits'] = tuple(qubits)
                    # pylint: disable=not-callable
                    result = method(**kwargs)
                    # pylint: enable=not-callable
                assert method is not None
                assert result is not None
                assert result is not NotImplementedError
                assert result is not NotImplemented
