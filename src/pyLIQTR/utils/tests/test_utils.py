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
import pytest
from openfermion.circuits import trotterize_exp_qubop_to_qasm
from openfermion import jordan_wigner
from openfermion.chem import MolecularData
from cirq.contrib import qasm_import
import random

#qsp
import pyLIQTR.QSP.gen_qsp                 as qspFuncs
import pyLIQTR.QSP.QSP                     as pQSP
import pyLIQTR.sim_methods.quantum_ops  as vs        
from pyLIQTR.QSP.Hamiltonian             import Hamiltonian as pyH
from pyLIQTR.QSP.qsp_helpers             import qsp_decompose_once
#utils
from pyLIQTR.utils.utils import open_fermion_to_qasm
from pyLIQTR.utils.printing import to_openqasm
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform

class TestUtils:
    @pytest.fixture(scope="class")
    def trotterized_hamiltonian(self):
        """Generate the hamiltonian for a system (H2 in this case),
        apply jordan-wigner transform and create the trotterized hamiltonian cirq-qasm generator"""
        diatomic_bond_length = 0.7414
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, diatomic_bond_length))]
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        description = str(diatomic_bond_length)

        molecule = MolecularData(geometry, basis, multiplicity, charge, description)
        molecule.load()

        mol_ham = molecule.get_molecular_hamiltonian()
        jw_ham = jordan_wigner(mol_ham)
        trotterized_hamiltonian = trotterize_exp_qubop_to_qasm(
            jw_ham * 2, evolution_time=1, trotter_number=1, trotter_order=1
        )
        yield trotterized_hamiltonian
        del trotterized_hamiltonian

    @pytest.fixture(scope="class")
    def vlasov_n8_qsp(self):
        random.seed(0)
        N      = 8
        k      =  2.0
        nu     =  0.0
        alpha  =  0.6

        required_precision   = 1e-2
        timestep_of_interest = 0.05 # sim_time

        ham_string = vs.hamiltonian_wfn_vlasov_hermite_linear_sym_string(k, alpha, nu, N)

        qsp_H = pyH(ham_string)

        print('\n: --- Visual Check Hamiltonian ---')
        print(f': qsp_H = {qsp_H}\n')

        angles, tolerances = qspFuncs.compute_hamiltonian_angles(qsp_H,\
                                        simtime=timestep_of_interest,req_prec=required_precision,mode="legacy")

        qsp_generator = pQSP.QSP(phis=angles, hamiltonian=qsp_H, target_size=qsp_H.problem_size)
        qsp_circ      = qsp_generator.circuit()
        decomposed_once_circuit = cirq.align_left(qsp_decompose_once(qsp_circ))
        decomposed_circuit      = cirq.align_left(qsp_decompose_once(decomposed_once_circuit))
        cliff_plus_T_circuit    = cirq.align_left(clifford_plus_t_direct_transform(decomposed_circuit))
        yield cliff_plus_T_circuit
        del cliff_plus_T_circuit


    def test_open_fermion_to_qasm(self, trotterized_hamiltonian):
        """This will test the util function that translates OpenFermion to OpenQASM 2.0.
        It does this by simply passing in the same input every time, expecting to get the same output."""
        jw_ham_trotterized_circuit = open_fermion_to_qasm(
            4, trotterized_hamiltonian, reg_name="reg"
        )
        circuit = qasm_import.circuit_from_qasm(jw_ham_trotterized_circuit)
        assert circuit is not None
        # Let's check and make sure the circuit is the expected length
        assert len(circuit) == 53
        # Lets looks a little closer at a couple of the circuit entries, we'll use the first and last for now
        assert len(circuit.moments[0].qubits) == 4
        # Grab the cirq.NamedQubit objects from the frozenset, so we can look a little closer...
        o1, o2, o3, o4 = circuit.moments[0].operations
        # We should have operations in this moment in the order [H, H, Rx, Rx] so lets walk through these
        assert isinstance(o1.gate, cirq.HPowGate)
        assert isinstance(o2.gate, cirq.HPowGate)
        assert isinstance(o3.gate, cirq.Rx)
        assert isinstance(o4.gate, cirq.Rx)
        # for this circuit, the last element should be [52] and it should be a frozenset of 1
        # we made sure the circuit was only 53 entries above, and now we will check the last entry
        # to make sure it is as expected.
        assert len(circuit.moments[52].qubits) == 1
        # Grab the cirq.NamedQubit object from the frozenset, so we can look a little closer...
        qubit, = circuit.moments[52].qubits
        assert qubit.name == "reg_3"
        assert qubit.dimension == 2

    def test_to_openqasm(self, vlasov_n8_qsp):
        circuit_output = "".join([line for line in to_openqasm(circuit_in=vlasov_n8_qsp)])
        assert(circuit_output is not "")
        #first 1000 chars
        test_output = circuit_output[0:1000]
        true_output = '// Generated from Cirq, Openfermion, and MIT LL\n\nOPENQASM 2.0;\ninclude "qelib1.inc";\n\n// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7), ctl_q9, ctl_q10, ctl_q11, ctl_q12, phs_q8, z_anc_q13, z_anc_q14, z_anc_q15, z_anc_q16]\nqreg q[17];\n\nreset q[8];\nreset q[9];\nreset q[10];\nreset q[11];\nreset q[12];\nreset q[13];\nreset q[14];\nreset q[15];\nreset q[16];\nsdg q[0];\nsdg q[1];\nsdg q[2];\nsdg q[3];\nsdg q[4];\nsdg q[5];\nsdg q[6];\nsdg q[7];\nx q[8];\nx q[9];\nx q[10];\nx q[11];\nh q[12];\nreset q[13];\nreset q[14];\nreset q[15];\nreset q[16];\nsdg q[12];\nsdg q[8];\nh q[13];\nh q[14];\nh q[15];\nh q[16];\nh q[12];\nh q[8];\nsdg q[12];\ns q[8];\nt q[12];\nx q[8];\nh q[12];\nt q[8];\ns q[12];\nh q[8];\nt q[12];\nt q[8];\nh q[12];\nh q[8];\nt q[12];\nt q[8];\nh q[12];\nh q[8];\nt q[12];\ns q[8];\nh q[12];\nt q[8];\nt q[12];\nh q[8];\nh q[12];\nt q[8];\nt q[12];\nh q[8];\nh q[12];\ns q[8];\ns q[12];\nt q[8];\nt q[12];\nh q[8];\nh q[12];\nt q[8];\ns q[12];\nh q[8];\nt q[12];\nt q[8];\nh q[12];\nh q[8];\nt q[12];\nt q[8];\nh q[12];\nh q[8];\ns q[12];\ns q[8\n'
        
        test_output = test_output.splitlines()
        true_output = true_output.splitlines()
        idx=0
        for truth,test in zip(true_output,test_output):
            idx+=1
            truth = f"l{idx}: {truth}"
            test = f"l{idx}: {test}"
            assert(truth==test)

    @pytest.mark.skip
    def test_invalid_trotterized_hamiltonian(self):
        """Generate an invalid hamiltonian for a system (H2 in this case),
        apply jordan-wigner transform and create the trotterized hamiltonian cirq-qasm generator"""
        diatomic_bond_length = -0.7414
        geometry = [("H", (1.0, 0.0, 0.0)), ("H", (0.0, 0.0, diatomic_bond_length))]
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        description = str(diatomic_bond_length)

        molecule = MolecularData(geometry, basis, multiplicity, charge, description)
        molecule.load()

        mol_ham = molecule.get_molecular_hamiltonian()
        jw_ham = jordan_wigner(mol_ham)
        invalid_trotterized_hamiltonian = trotterize_exp_qubop_to_qasm(
            jw_ham * 2, evolution_time=1, trotter_number=1, trotter_order=1
        )
        assert invalid_trotterized_hamiltonian is None