import pytest
import cirq
import rustworkx as rx

from pyLIQTR.scheduler.DAG import DAG
from pyLIQTR.scheduler.Instruction import Instruction

#testing utils
from pyLIQTR.scheduler.scheduler_utils import get_instruction2node_map, manual_link, dag_creation
from rustworkx.visualization import graphviz_draw

class TestDAG:

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
        auto_dag = dag_creation(cirq.Circuit(canonical_basic_instance[0]))

        canonical_insts = canonical_basic_instance[0]
        qbs = canonical_basic_instance[1]

        manualDAG = rx.PyDAG()
        op2node,q2node = get_instruction2node_map(manualDAG,canonical_basic_instance[0],\
                                                     qubits=canonical_basic_instance[1])
        manual_link(manualDAG,q2node,op2node,\
               qbs[0],[canonical_insts[0],canonical_insts[1]],qbs[0],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[1],canonical_insts[0],qbs[1],'X')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[2],canonical_insts[1],qbs[2],'X')

        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[0],canonical_insts[1]],canonical_insts[2],qbs[0],'X')
        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[0],canonical_insts[1]],canonical_insts[3],qbs[0],'X')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[3],canonical_insts[2],qbs[3],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[4],canonical_insts[3],qbs[4],'Z')

        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[2],canonical_insts[3]],canonical_insts[4],qbs[0],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[2],canonical_insts[3]],canonical_insts[5],qbs[0],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[5],canonical_insts[4],qbs[5],'X')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[6],canonical_insts[5],qbs[6],'X')
        
        
        assert(manualDAG == auto_dag)

    def test_verify_reversed(self,canonical_basic_instance):
        """
        The canonical set where each layer is 'reversed' should not matter from a DAG standpoint.

        Ie, the graphs should be the same.
        """
        canonical_insts = canonical_basic_instance[0]
        qbs = canonical_basic_instance[1]

        canonical_insts_rev = [canonical_insts[1],canonical_insts[0],\
                           canonical_insts[3],canonical_insts[2],\
                           canonical_insts[5],canonical_insts[4]]

        auto_dag = dag_creation(cirq.Circuit(canonical_insts_rev))
        manualDAG = rx.PyDAG()
        op2node,q2node = get_instruction2node_map(manualDAG,canonical_basic_instance[0],\
                                                     qubits=canonical_basic_instance[1])
        manual_link(manualDAG,q2node,op2node,\
               qbs[0],[canonical_insts[0],canonical_insts[1]],qbs[0],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[1],canonical_insts[0],qbs[1],'X')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[2],canonical_insts[1],qbs[2],'X')

        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[0],canonical_insts[1]],canonical_insts[2],qbs[0],'X')
        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[0],canonical_insts[1]],canonical_insts[3],qbs[0],'X')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[3],canonical_insts[2],qbs[3],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[4],canonical_insts[3],qbs[4],'Z')

        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[2],canonical_insts[3]],canonical_insts[4],qbs[0],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    [canonical_insts[2],canonical_insts[3]],canonical_insts[5],qbs[0],'Z')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[5],canonical_insts[4],qbs[5],'X')
        manual_link(manualDAG,q2node,op2node,\
                    qbs[6],canonical_insts[5],qbs[6],'X')
        
        #Here we check the OG construction versus a 'mirrored' one. they should be the same
        assert(manualDAG == auto_dag)

    def test_minimal_cirq_quantum_gates(self):
        """
        This test validates that this minimal set of cirq gates works correctly, both in terms of
        'do i support this gate' and 'do i have the right dependency'
        """
        #here we build a 'dumb' circuit that just tests various gates and their dependencies.
        #we should not fail on these
        q = cirq.LineQubit.range(2)

        gates = [(cirq.CX,('Z','X')),\
                (cirq.CZ,('Z','Z')),\
                (cirq.X,('X',)),\
                (cirq.Z,('Z',)),\
                (cirq.Y,('All',)),\
                (cirq.H,('All',)),\
                (cirq.T,('All',)),\
                (cirq.S,('All',)),\
                ]

        gates_as_circuit = [g[0].on(*q[0:len(g[1])]) for g in gates]
        gac = gates_as_circuit
        auto_dag = dag_creation(cirq.Circuit(gates_as_circuit))
        manualDAG = rx.PyDAG()
        op2node,q2node = get_instruction2node_map(manualDAG,\
                                                gates_as_circuit, q)

        
        idx=0 #CX to q0, x to ccx
        manual_link(manualDAG, q2node,op2node,\
                    q[0], gac[idx], q[0],'Z')
        manual_link(manualDAG, q2node,op2node,\
                    q[1], gac[idx], q[1],'X')
        idx+=1 #CZ to 0, z to cx
        manual_link(manualDAG, q2node,op2node,\
                q[0], gac[idx], q[0],'Z')
        manual_link(manualDAG, q2node,op2node,\
                    gac[idx-1], gac[idx], q[1],'Z')
        idx+=1 #X
        manual_link(manualDAG, q2node,op2node,\
                    [gac[1],gac[0]], gac[idx], q[0],'X')
        idx+=1 #Z
        manual_link(manualDAG, q2node,op2node,\
                    gac[idx-1], gac[idx], q[0],'Z')
        idx+=1 #Y
        manual_link(manualDAG, q2node,op2node,\
                    gac[idx-1], gac[idx], q[0],'All')
        idx+=1 #H
        manual_link(manualDAG, q2node,op2node,\
                    gac[idx-1], gac[idx], q[0],'All')
        idx+=1 #T
        manual_link(manualDAG, q2node,op2node,\
                    gac[idx-1], gac[idx], q[0],'All')
        idx+=1 #S - note that this two of the same dependencies back to back...
        #so the idx-X is different.
        manual_link(manualDAG, q2node,op2node,\
                    gac[idx-1], gac[idx], q[0],'All')
        
        assert(auto_dag == manualDAG)
