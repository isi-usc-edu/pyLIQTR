#Scheduler utils
import cirq
import rustworkx as rx
from pyLIQTR.scheduler.DAG import DAG
from pyLIQTR.scheduler.Instruction import Instruction, DependencyEdge

def dag_creation(circuit):
    dag = DAG(max_moments=100)
    for op in circuit.all_operations():
        inst = Instruction(op)
        dag.add_dependency(inst)
    dag.finish()
    return dag

def get_instruction2node_map(graph,ops,qubits):
    graph.clear()
    ops_to_nodes = {}
    for op in ops:
        ops_to_nodes[op] = graph.add_node(Instruction(op))
    
    qubits_to_nodes = {}
    for q in qubits:
        qubits_to_nodes[q] = graph.add_node(q)

    return ops_to_nodes, qubits_to_nodes

def manual_link(graph,qubit_map, inst_map,\
                start,end,\
                qubit,type):
    if isinstance(start,list) and not isinstance(end,list):
        for s in start:
            manual_link(graph,qubit_map, inst_map,\
                        s, end, 
                        qubit,type)
    elif isinstance(end,list) and not isinstance(start,list):
        for e in end:
            manual_link(graph, qubit_map, inst_map,\
                        start, e, 
                        qubit,type)
    else:
        assert(not isinstance(start,list))
        assert(not isinstance(end,list))

        if isinstance(start,cirq.ops.gate_operation.GateOperation):
            start_node = inst_map[start]
        else:
            start_node = qubit_map[start]

        if isinstance(end,cirq.ops.gate_operation.GateOperation):
            end_node = inst_map[end]
        else:
            end_node = qubit_map[end]

        graph.add_edge(start_node,end_node,DependencyEdge(qubit,type))