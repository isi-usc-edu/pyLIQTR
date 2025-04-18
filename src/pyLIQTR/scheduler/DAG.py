import rustworkx as rx
from rustworkx.visualization import graphviz_draw

from pyLIQTR.scheduler.Instruction import Instruction, DependencyEdge, CirqInstruction, OQInstruction
import time

class DAG:
    """
    A class for storing and manipulating a DAG (directional acyclic graph). Allows for instructions to be linked by dependency type to qubits or previous instructions and removed by a scheduler or other external method. This class wraps around a rustworkx.PyDAG object.
    """
    def __init__(self,max_moments: int):
        """
        :param int max_moments: the maximum number of allowed instructions in the DAG. If this number is exceeded, no more instructions will be added to the DAG until it has been emptied by the scheduler.
        :param dict qubit_node_idx: Dictionary of the form {qX: node index, qY: node index, ...}. Since qubit nodes are not instructions and do not have an index attribute, this dict saves the relevant node index to be referenced later.
        :param list free_instructions: Running list of nodes with no incoming edges. Each time a qubit node is created, it is automatically added to this list, since it has no incoming edges by definition. Once a node has been added to the Scheduler's ready queue, it will be removed from this list. The list will be repopulated each time an instruction has been removed from the DAG.
        :param dict dependency_log: Dictionary of the form {qX: [(second most recent dependency type, [associated instructions]), (most recent dependency type, [associated instructions])], qY: ...}. Stores two most recent dependency types  on a particular qubit and the set of commutable operators associated with each.
        :param list loop_catcher: list keeping track of the ten most recently removed DAG layers. If all of the entries are '0' then the instruction removal is stuck in a loop. The DAG will then initiate a search for free instructions to resolve the loop.
        :param int dag_cycle: Denotes how many times the DAG has reached its max_moment count, cleared, and begun refilling.
    """
        self.insts_in_dag = 0
        self.max_moments = max_moments
        self.dag = rx.PyDAG()
        self.qubit_node_idx = {} 
        self.free_instructions = []
        self.dependency_log = {}
        self.loop_catcher = []
        self.dag_cycle = 0
        
        
    
    def __eq__(self,other):
        """
        Validate whether or not two graphs are equal by checking types and isomorphism.
        """
        if type(other) != type(self):
            if isinstance(other,rx.PyDAG):
                return rx.is_isomorphic(self.dag, other)
            else:
                return False
        else:
            return rx.is_isomorphic(self.dag,other.dag)

    
    def add_dependency(self,inst, forceAdd = True):
        '''
        Add an instruction to the DAG along with each of its dependencies.
        '''
        self.insts_in_dag += 1
        dependencies = inst.get_data_dependencies()
        for dep in dependencies.keys():
            self._link(dep,dependencies[dep],inst) #link current inst and dependency

    
    def _link(self, dependency, dependency_type:str, instruction):
        '''
        For each dependency an instruction has, follow these steps:
        1.  If the instruction does not yet exist as a node, add a node to the DAG with instruction attached. 
            If the instruction already has a node, fetch the index of the existing node.
        2.  If the qubit that the dependency is on does not yet exist in the DAG, create a node with qubit attached.
            Add an edge between the qubit node and the instruction node. Initialize this qubit's entry in the dependency log as:
            qX: [(current dependency_type, [instruction])].
        3.  If the qubit already exists in the DAG, reference its dependency log entry. The entry will be structured as:
                qX: [(second most recent dependency_type, [instructions]), (most recent dependency_type, [instructions])].
            Compare the current dependency_type to the most recent dependency_type in the log. 
            If they match, simply append the current instruction to the list of instructions associated with the most recent dependency_type.
            If they do not match:
                Check the length of the qubit's dependency log entry. 
                    If it is 1, then:
                    qX: [(most recent dependency_type, [instructions])]. 
                    Each instruction in this list is an 'initial instruction', meaning that it must stem directly from a qubit node.
                    Attach each of these initial instructions to the qubit node if an edge does not already exist between them.
                    Now, add a second item to the qubit's dependency log entry, such that:
                    qX: [(most recent dependency_type, [instructions]), (current dependency_type, [current instruction])].

                    If it is 2, then:
                    qX: [(second most recent dependency_type, [instructions]), (most recent dependency_type, [instructions])].
                    For each instruction associated with the second most recent dependency_type, attach it to each instruction associated
                    with the most recent dependency_type. This is to account for the fact that since each list of instructions is commutable 
                    within itself and could occur in any order, each instruction in the later list will depend on each instruction in 
                    the first list.
                    Once this interlinking is complete, modify the qubit's entry in the dependency log such that:
                    qX: [(most recent dependency_type, [instructions]), (current dependency_type, [current instruction])].
                
                    
        '''
        if instruction.index is None:
            inst_idx = self.dag.add_node(instruction)
            instruction.index = inst_idx
        
        else:
            inst_idx = instruction.index
        qubit = dependency
        if qubit in self.dependency_log:
            if (dependency_type == self.dependency_log[qubit][-1][0]) and not (dependency_type == 'All'):
                self.dependency_log[qubit][-1][1].append(instruction)
            else:
                if len(self.dependency_log[qubit]) == 1:
                    for c_inst in self.dependency_log[qubit][0][1]:
                        q_idx = self.qubit_node_idx[qubit]
                        c_idx = c_inst.index
                        if not self.dag.has_edge(q_idx, c_idx): 
                            self.dag.add_edge(q_idx, c_idx, DependencyEdge(qubit, self.dependency_log[qubit][0][0]))
                    self.dependency_log[qubit].append((dependency_type, [instruction]))
                elif len(self.dependency_log[qubit]) == 2:
                    for p_inst in self.dependency_log[qubit][0][1]:
                        for c_inst in self.dependency_log[qubit][1][1]:
                            p_idx = p_inst.index
                            c_idx = c_inst.index
                            #if p_idx in self.dag.node_indices():
                            self.dag.add_edge(p_idx, c_idx, DependencyEdge(qubit, self.dependency_log[qubit][1][0]))
                            #else:
                            #    if self.dag.in_degree(c_idx) == 0:
                            #        self.free_instructions.append(c_inst)
                    new_p_insts = self.dependency_log[qubit][1]
                    self.dependency_log[qubit] = [new_p_insts, (dependency_type, [instruction])]
        else:
            qubit_node = self.dag.add_node(qubit)
            self.qubit_node_idx[qubit] = qubit_node
            self.dag.add_edge(qubit_node, inst_idx, DependencyEdge(qubit, dependency_type))
            self.dependency_log[qubit] = [(dependency_type, [instruction]),]
            self.free_instructions.append(qubit)
            self.insts_in_dag += 1

       


    
    def _unlink(self, instruction):
        """
        When an instruction is ready to be executed and removed from the DAG,
        follow these steps:
        1.  Note each child node of the instruction.
        2.  remove each edge stemming from the instruction.
        3.  remove the instruction node itself.
        4.  iterate through the children nodes. If a child node now has no incoming edges, add it to the 
            list of free instructions. If there are still one or more incoming edges to the child node,
            pass it over.
        """
        if type(instruction) != CirqInstruction and type(instruction) != OQInstruction:
            inst_idx = self.qubit_node_idx[instruction]
        else:
            inst_idx = instruction.index
            
        children = self.dag.successor_indices(inst_idx)

        out_edges = self.dag.out_edges(inst_idx)
        for edge in out_edges:
            child_idx = edge[1]
            self.dag.remove_edge(inst_idx, child_idx)

        self.dag.remove_node(inst_idx)
            
        for child in children:
            in_edges = self.dag.in_edges(child)
            if len(in_edges) == 0:
                child_instruction = self.dag.get_node_data(child)
                if child_instruction.free == False:
                    self.free_instructions.append(child_instruction)
                    child_instruction.free = True
            else:
                pass
        

    def get_available(self):
        """
        Return current list of nodes with no incoming edges.     
        """
        if len(self.dag.node_indices()) <= 0:
            return "End"
        else:
            if len(self.free_instructions) == 0:
                for i in range(self.insts_in_dag):
                    node = self.dag.node_indices()[i]
                    in_degree = self.dag.in_degree(node)
                    if in_degree == 0:
                        self.free_instructions.append(self.dag.get_node_data(node))

            return self.free_instructions
            
    
    
    def _remove_dependency(self, inst):
        '''
        Wrap around _unlink.
        '''
        self.insts_in_dag -= 1
        self._unlink(inst)


    def remove_dependencies(self, insts: list): 
        '''
        Remove multiple dependencies simultaneously.
        '''
        for inst in insts:
            self._remove_dependency(inst)


    def finish(self):
        """
        Once all instructions have been added to the DAG, link sets of commutable operations one last time.
        """
        for qubit in self.dependency_log: 
            if len(self.dependency_log[qubit]) == 2: 
                for p_inst in self.dependency_log[qubit][0][1]:
                    for c_inst in self.dependency_log[qubit][1][1]:
                        parent_idx = p_inst.index
                        child_idx = c_inst.index
                        self.dag.add_edge(parent_idx, child_idx, DependencyEdge(qubit, self.dependency_log[qubit][1][0]))


    def full(self):
        return self.insts_in_dag > self.max_moments

    def empty(self):
        return (self.insts_in_dag == 0)
    
    def draw(self,filename='test.png',image_type='png'):
        def edge_attr(edge, dag):
            edge_type = edge.type
            if edge_type == 'X':
                return {'color': 'blue', 'label': str(edge)}
            elif edge_type == 'Z':
                return {'color': 'red', 'label': str(edge)}
            else:
                return {'color': 'black', 'label': str(edge)}
            
            
        graphviz_draw(self.dag, node_attr_fn=lambda node: {"label": str(node)}, edge_attr_fn=lambda edge: edge_attr(edge, self.dag), filename=filename, image_type=image_type)