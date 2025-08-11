import pytest
import cirq
from   pyLIQTR.scheduler.scheduler            import Scheduler, schedule_circuit, decomposition_protocol
from   pyLIQTR.scheduler.Instruction          import CirqInstruction
from   pyLIQTR.scheduler.DAG                  import DAG
from   qualtran.bloqs.arithmetic.comparison   import LessThanConstant as ltc
from   pyLIQTR.circuits.operators             import *
from   pyLIQTR.ProblemInstances.getInstance   import getInstance
from   pyLIQTR.clam.lattice_definitions       import SquareLattice
from   pyLIQTR.BlockEncodings.getEncoding     import getEncoding, VALID_ENCODINGS
from   pyLIQTR.scheduler.scheduler_utils      import architecture, state_factory
import pyLIQTR.scheduler.sets as sets
from   pyLIQTR.qubitization.phase_estimation  import   QubitizedPhaseEstimation

#Cant disable UserWarning caused by scheduler, so putting this in for now
import warnings
warnings.filterwarnings("ignore")

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

        assert results['Total time for execution'] == 1200, results['Circuit T-depth'] == 0

    
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
        inst1 = CirqInstruction(cirq.X(qubits[0]))
        inst2 = CirqInstruction(cirq.CX(qubits[1], qubits[0]))
        inst1.get_data_dependencies()
        inst2.get_data_dependencies()
        scheduler = Scheduler(architecture_params=None)
        time1 = scheduler.assign_execution_time(inst1)
        time2 = scheduler.assign_execution_time(inst2)
        assert time1 == 100, time2 == 200
    

    def test_populate_ready(self, canonical_basic_instance):
        
        dag = DAG(max_moments=100)
        
        canonical_insts = canonical_basic_instance[0]
        qbs = canonical_basic_instance[1]
        insts = []
        for inst in canonical_insts:
            instruction = CirqInstruction(inst)
            insts.append(instruction)
            dag.add_dependency(instruction)
        dag.finish()

        expected_ready_q = [insts[0], insts[1]]

        scheduler = Scheduler(architecture_params=None)

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
            instruction = CirqInstruction(inst)
            insts.append(instruction)
            dag.add_dependency(instruction)
        dag.finish()

        scheduler = Scheduler(architecture_params=None)

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
            instruction = CirqInstruction(inst)
            dag.add_dependency(instruction)
        dag.finish()

        scheduler = Scheduler(architecture_params=None)

        #first cycle, get through all qubit nodes
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)
        scheduler.free_instructions(dag)

        #second cycle, now actual instructions should be added
        scheduler.populate_ready(dag)
        scheduler.move_ready_to_execute(dag)
        scheduler.free_instructions(dag)

        assert (scheduler.global_time == 200 and len(scheduler.execution_q) == 0)

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
            'T': 79,
            'H': 129,
            'S': 6, 
            'CX': 49,
            'Toffoli': 0,
            'CZ': 6,
            'Pauli (X, Y, Z)': 4,
            'Measurement/Reset': 12
        }
        print(actual_gate_counts, full_decomp_gate_counts, no_decomp_gate_counts)
        assert full_decomp_gate_counts == no_decomp_gate_counts == actual_gate_counts

    '''
    this works but takes forever and i think the test above is sufficient
    def test_recursion_gate_counts_fh(self):

        J      = -1.0;          N      =     2    
        U      =  4.0;          shape  =  (N,N)

        model  =  getInstance('FermiHubbard',shape=shape, J=J, U=U, cell=SquareLattice)
        block_encoding    =  getEncoding(VALID_ENCODINGS.FermiHubbardSquare)(model)
        circuit = block_encoding.circuit

        no_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level=0, decomposition_protocol=decomposition_protocol.recursive)
        no_decomp_gate_counts = no_decomp_result['Gate profile']

        full_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level='Full')
        full_decomp_gate_counts = full_decomp_result['Gate profile']

        print(full_decomp_gate_counts, no_decomp_gate_counts)

        assert full_decomp_gate_counts == no_decomp_gate_counts
    '''

    def test_recursion_qubit_counts(self):

        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(7)

        op = ltc(6, 4).on(qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5], qubits[6])
        rot = cirq.Rz(rads=1.435).on(qubits[1])

        circuit.append(op)
        circuit.append(rot)

        no_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level=0, decomposition_protocol=decomposition_protocol.recursive)
        no_decomp_qubit_count = no_decomp_result['Number of qubits used']

        full_decomp_result = schedule_circuit(circuit, full_profile=True, decomp_level='Full')
        full_decomp_qubit_count = full_decomp_result['Number of qubits used']

        assert full_decomp_qubit_count == no_decomp_qubit_count == 14

    def test_t_state(self):

        test_q = cirq.LineQubit.range(5)
        t_circuit = cirq.Circuit()
        t_circuit.append([cirq.T(test_q[0]), cirq.T(test_q[1]), cirq.T(test_q[2])])
        t_circuit.append([cirq.T(test_q[0]), cirq.T(test_q[1])])
        t_circuit.append([cirq.T(test_q[0]), cirq.T(test_q[1]), cirq.T(test_q[2]), cirq.T(test_q[3]), cirq.T(test_q[4])])
        t_circuit.append([cirq.T(test_q[0])])

        t_factory_1 = state_factory(production_time=4, consumption_time=1, production_limit=2, success_prob=1)
        my_timings = {sets.T: [t_factory_1], sets.CLIFFORD: 1, sets.MISC: 0}
        my_params = architecture(user_input_timings=my_timings)

        results = schedule_circuit(t_circuit, architecture_config=my_params)

        assert results['Circuit T-depth'] == 8

    def test_multi_run_qubit_count(self):
        model  =  getInstance('FermiHubbard',shape=(3,3), J=-1, U=4, cell=SquareLattice)
        gate_gsee = QubitizedPhaseEstimation( getEncoding(VALID_ENCODINGS.FermiHubbardSquare),
                                      instance=model,prec=4)
        q_counts = []
        for _ in range(3):
            res = schedule_circuit(gate_gsee.circuit, full_profile=True, decomp_level=0, context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
            q_counts.append(res["Number of qubits used"])

        assert q_counts[0] == q_counts[1] == q_counts[2]

    



    

