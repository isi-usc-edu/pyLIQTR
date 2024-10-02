import pytest
import math
import numpy as np
import cirq
import rustworkx as rx
from pyLIQTR.scheduler.scheduler import Scheduler, schedule_circuit
from pyLIQTR.scheduler.Instruction import Instruction
from pyLIQTR.scheduler.DAG import DAG
from qualtran.bloqs.arithmetic.comparison import LessThanConstant as ltc
from pyLIQTR.circuits.operators                     import *
from pyLIQTR.BlockEncodings.CarlemanLinearization   import Carleman_Linearization
from pyLIQTR.ProblemInstances.NonlinearODE          import FOperators
from    pyLIQTR.ProblemInstances.getInstance                  import   getInstance
from    pyLIQTR.clam.lattice_definitions                      import   CubicLattice, SquareLattice, TriangularLattice
from    pyLIQTR.BlockEncodings.getEncoding                    import   getEncoding, VALID_ENCODINGS
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran import Register, Signature, QAny

class TestScheduler:

    @pytest.fixture(scope="class")
    def canonical_basic_instance(self):
        qbs = cirq.LineQubit.range(7)
        canonical = [cirq.CX(qbs[0],qbs[1]),cirq.CX(qbs[0],qbs[2]),\
                            cirq.CX(qbs[3],qbs[0]),cirq.CX(qbs[4],qbs[0]),\
                            cirq.CX(qbs[0],qbs[5]),cirq.CX(qbs[0],qbs[6])]
        yield canonical, qbs
    
    def test_proof_of_concept(self,canonical_basic_instance):
        """
        Test our canonical exmaple
        """
        canonical_insts = canonical_basic_instance[0]
        canonical_circuit = cirq.Circuit(canonical_insts)
        results = schedule_circuit(canonical_circuit)

        assert results['Total time for execution'] == 12, results['Circuit T-depth'] == 0

    
    def test_t_depth(self):
        """
        Test T-depth functionality
        """
        test_q = cirq.LineQubit.range(5)
        t_circuit = cirq.Circuit()
        t_circuit.append([cirq.T(test_q[0]), cirq.T(test_q[1]), cirq.T(test_q[2])])
        t_circuit.append([cirq.T(test_q[0]), cirq.T(test_q[1])])
        t_circuit.append([cirq.T(test_q[0]), cirq.T(test_q[1]), cirq.T(test_q[2]), cirq.T(test_q[3]), cirq.T(test_q[4])])
        t_circuit.append([cirq.T(test_q[0])])

        results = schedule_circuit(t_circuit)

        assert results['Circuit T-depth'] == 4


    def test_time_assignment(self):
        
        qubits = cirq.LineQubit.range(2)
        resources = {qubits[0]: 1, qubits[1]: 1}
        inst1 = Instruction(cirq.X(qubits[0]))
        inst2 = Instruction(cirq.CX(qubits[1], qubits[0]))
        scheduler = Scheduler(arch_description=resources)
        time1 = scheduler.assign_execution_time(inst1)
        time2 = scheduler.assign_execution_time(inst2)
        assert time1 == 1, time2 == 2

    def test_populate_ready(self, canonical_basic_instance):
        
        dag = DAG(max_moments=100)
        
        canonical_insts = canonical_basic_instance[0]
        qbs = canonical_basic_instance[1]
        insts = []
        for inst in canonical_insts:
            instruction = Instruction(inst)
            insts.append(instruction)
            dag.add_dependency(instruction)
        dag.finish()

        resources = {}
        for qbit in qbs:
            resources[qbit] = 1

        expected_ready_q = [insts[0], insts[1]]

        scheduler = Scheduler(arch_description=resources)

        #first cycle, get through all qubit nodes
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)
        scheduler.free_instructions(dag)

        #second cycle, now actual instructions should be added
        scheduler.populate_ready(dag)

        assert scheduler.ready_q == expected_ready_q

    def test_move_ready_to_ex(self, canonical_basic_instance):

        dag = DAG(max_moments=100)
        
        canonical_insts = canonical_basic_instance[0]
        qbs = canonical_basic_instance[1]
        insts = []
        for inst in canonical_insts:
            instruction = Instruction(inst)
            insts.append(instruction)
            dag.add_dependency(instruction)
        dag.finish()

        resources = {}
        for qbit in qbs:
            resources[qbit] = 1

        scheduler = Scheduler(arch_description=resources)

        expected_execution_q = [insts[0]]

        #first cycle, get through all qubit nodes
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)
        scheduler.free_instructions(dag)

        #second cycle, now actual instructions should be added
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)

        assert scheduler.execution_q == expected_execution_q 

    def test_execution(self, canonical_basic_instance):

        dag = DAG(max_moments=100)
        
        canonical_insts = canonical_basic_instance[0]
        qbs = canonical_basic_instance[1]
        for inst in canonical_insts:
            instruction = Instruction(inst)
            dag.add_dependency(instruction)
        dag.finish()

        resources = {}
        for qbit in qbs:
            resources[qbit] = 1

        scheduler = Scheduler(arch_description=resources)

        #first cycle, get through all qubit nodes
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)
        scheduler.free_instructions(dag)

        #second cycle, now actual instructions should be added
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)
        scheduler.free_instructions(dag)

        assert (scheduler.global_time == 2 and len(scheduler.execution_q) == 0)

    def test_recursion_gate_counts_simple(self):

        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(7)

        op = ltc(6, 4).on(qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5], qubits[6])
        rot = cirq.Rz(rads=1.435).on(qubits[1])

        circuit.append(op)
        circuit.append(rot)

        full_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level='Full')
        no_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level=0)

        full_decomp_gate_counts = full_decomp_result['Gate profile']
        no_decomp_gate_counts = no_decomp_result['Gate profile']

        actual_gate_counts = {
            'T': 24,
            'H': 18,
            'S': 6, 
            'CX': 49,
            'CZ': 6,
            'Pauli (X, Y, Z)': 4,
            'Rotation': 1
        }

        assert full_decomp_gate_counts == no_decomp_gate_counts == actual_gate_counts

    def test_recursion_gate_counts_carleman(self):

        n = 2
        K = 4

        a0_in = n
        a1 = n
        a2_in = math.ceil(n/2)
        a2_out = math.ceil(n)

        alpha0 = 1
        alpha1 = 1
        alpha2 = 1

        ancilla_register = Register("ancilla", QAny(bitsize = 7 + max(a0_in, a1, a2_in, a2_out) + math.ceil(np.log2(K))))
        data_register = Register("data", QAny(bitsize = n*K+1))

        signature = Signature([ancilla_register, data_register])
        registers = get_named_qubits(signature)

        carlemanLinearization = Carleman_Linearization(FOperators(n, K, (a0_in, a1, a2_in, a2_out), (alpha0, alpha1, alpha2)), K)
        circuit = cirq.Circuit(carlemanLinearization.on_registers(**registers))

        no_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level=0)
        no_decomp_gate_counts = no_decomp_result['Gate profile']

        full_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level='Full')
        full_decomp_gate_counts = full_decomp_result['Gate profile']


        assert full_decomp_gate_counts == no_decomp_gate_counts

    def test_recursion_gate_counts_fh(self):

        J      = -1.0;          N      =     2    
        U      =  4.0;          shape  =  (N,N)

        model  =  getInstance('FermiHubbard',shape=shape, J=J, U=U, cell=SquareLattice)
        block_encoding    =  getEncoding(VALID_ENCODINGS.FermiHubbardSquare)(model)
        circuit = block_encoding.circuit

        no_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level=0)
        no_decomp_gate_counts = no_decomp_result['Gate profile']

        full_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level='Full')
        full_decomp_gate_counts = full_decomp_result['Gate profile']

        assert full_decomp_gate_counts == no_decomp_gate_counts

    def test_recursion_qubit_counts(self):

        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(7)

        op = ltc(6, 4).on(qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5], qubits[6])
        rot = cirq.Rz(rads=1.435).on(qubits[1])

        circuit.append(op)
        circuit.append(rot)

        no_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level=0)
        no_decomp_qubit_count = no_decomp_result['Number of qubits used']

        full_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level='Full')
        full_decomp_qubit_count = no_decomp_result['Number of qubits used']

        assert full_decomp_qubit_count == no_decomp_qubit_count == 14




    



    

