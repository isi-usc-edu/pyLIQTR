"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
import cirq
import numpy as np
from qualtran.bloqs.arithmetic.addition import Add as qtAdd
from qualtran.bloqs.arithmetic.addition import AddConstantMod as qtAddConstantMod
from qualtran import bloqs
from qualtran.bloqs import and_bloq
from cirq._compat import cached_property
from typing import List, Tuple, Sequence, Optional
from numpy.typing import NDArray

def two_complement(val,nb):
    if np.sign(val) >= 0:
        return format(val,"0{}b".format(nb))
    else:
        return two_complement(2**nb - np.abs(val),nb)

def debugPrint(name,circuit,initial_state=None):
    if isinstance(circuit,list):
        circuit = cirq.Circuit(circuit)
    qord = list(circuit.all_qubits())
    qord.sort()

    if initial_state is not None:
        assert(len(initial_state)==len(qord))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(name)
    print(circuit)
    print(cirq.Simulator().simulate(circuit,qubit_order=qord,initial_state=initial_state))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

class Add(qtAdd):
    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        input_bits = qubits[: self.bitsize][::-1]
        output_bits = qubits[self.bitsize :][::-1]
        if self.bitsize == 1:
            assert(len(input_bits) == 1)
            assert(len(output_bits) == 1)
            #I dont love this because it drops the carry bit...
            yield cirq.CX(input_bits[0],output_bits[0])
        else:
            ancillas = context.qubit_manager.qalloc(self.bitsize - 1)[::-1]
            # Start off the addition by anding into the ancilla
            yield and_bloq.And().on(input_bits[0], output_bits[0], ancillas[0])
            # Left part of Fig.2
            yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
            yield cirq.CX(ancillas[-1], output_bits[-1])
            yield cirq.CX(input_bits[-1], output_bits[-1])
            # right part of Fig.2
            yield from self._right_building_block(input_bits, output_bits, ancillas, self.bitsize - 2)
            yield and_bloq.And(uncompute=True).on(input_bits[0], output_bits[0], ancillas[0])
            yield cirq.CX(input_bits[0], output_bits[0])
            context.qubit_manager.qfree(ancillas)
        

class AddMod(qtAddConstantMod):
    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        #Following the construction in
        #https://journals.aps.org/pra/pdf/10.1103/PhysRevA.54.147
        #from Fig 4.
        tmpG = []
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

        isControlled =  len(self.cvs)>0
        if isControlled:
            ctlQubits = list(qubits[0:len(self.cvs)])
            targetQubit = context.qubit_manager.qalloc(1)
            yield bloqs.multi_control_multi_target_pauli.MultiControlPauli(self.cvs).on(*(ctlQubits+targetQubit))
        
        #MSB is first qubit
        #if the input register is A, with size a
        #then our tmp register B needs to be of size a+1
        MSB = 0
        
        #Cirq-ft add_mod is semiclassical (ie adds or subtracts a classical value)
        assert(np.abs(self.add_val) <= 2**(self.bitsize-1))
        #Classical value implemented.
        addVal = two_complement(self.add_val,self.bitsize)
        sign = self.add_val >= 0
        B = context.qubit_manager.qalloc(self.bitsize)
        for idx,c in enumerate(addVal):
            if c=='1':
                if isControlled:
                    yield cirq.CX.on(targetQubit[0],B[idx])
                else:
                    yield cirq.X.on(B[idx])
        
        #Need to implement |N>
        modVal = format(self.mod,'0{}b'.format(self.bitsize))
        N = context.qubit_manager.qalloc(self.bitsize)
        for idx,c in enumerate(modVal):
            if c=='1':
                yield cirq.X.on(N[idx])
        
        #Need a temporary qubit.
        t = context.qubit_manager.qalloc(1)
        tmp = t[0]
        
        add = Add(bitsize=self.bitsize)
        sub = cirq.inverse(add)
        
        
        #Input qubits (qubits) should be of the correct size. We make no guarantee about 
        #overflow though...
        assert(len(qubits)-len(self.cvs)==self.bitsize)
        A = list(qubits[len(self.cvs)::])
        #SWITCH IN ORDER TO DO INPLACE
        tmpB = A
        A = B
        B = tmpB

        #A,B,N,|0> = tmp
        #|a>,|b> --> |a>,|a+b>
        #debugPrint("Entry",tmpG)
        #tmpG.append(add.on(*(A+B)))
        yield add.on(*(A+B))
        #debugPrint("1st Add",tmpG)

        #"""
        #"""
        #|N>,|a+b> -> |N> |a+b-N> #(may overflow)
        #tmpG.append(sub.on(*(N+B)))        
        yield sub.on(*(N+B)) #correct to here
        #debugPrint("1st Sub",tmpG)
        #need to check MSB of |a+b-N> for overflow
        #tmpG.append(cirq.X.on(B[MSB]))
        #tmpG.append(cirq.CX.on(B[MSB],tmp))
        #tmpG.append(cirq.X.on(B[MSB]))
        #debugPrint("Control tmp",tmpG) 
        yield cirq.X.on(B[MSB])
        yield cirq.CX.on(B[MSB],tmp)
        yield cirq.X.on(B[MSB])
        #Control invert of |N> register, controlled by tmp
        for idx,c in enumerate(modVal):
            if c=='1':
                #tmpG.append(cirq.CX.on(tmp,N[idx]))
                yield cirq.CX.on(tmp,N[idx])

        #potentially add this |?N> back.
        #debugPrint("Control |N>",tmpG)
        #tmpG.append(add.on(*(N+B)))
        yield add.on(*(N+B))
        #debugPrint("Add |N>",tmpG) 
        for idx,c in enumerate(modVal):
            if c=='1':
                tmpG.append(cirq.CX.on(tmp,N[idx]))
                yield cirq.CX.on(tmp,N[idx])
        #do the subtraction so we can reset the |0> ancilla
        #debugPrint("Uncontrol |N>",tmpG)
        #tmpG.append(sub.on(*(A+B)))
        yield sub.on(*(A+B))
        #debugPrint("Subtract A,B",tmpG)

        #Control invert of |N> register, controlled by tmp
        #tmpG.append(cirq.CX.on(B[MSB],tmp))
        #debugPrint("Undo tmp",tmpG)
        #tmpG.append(add.on(*(A+B)))
        yield cirq.CX.on(B[MSB],tmp)
        yield add.on(*(A+B))
        #debugPrint("Final add",tmpG)
        #Then we want this to be in place... So we will just insert some swaps 
        for idx,c in enumerate(addVal):
            if c=='1':
                if isControlled:
                    yield cirq.CX.on(targetQubit[0],A[idx])
                else:
                    yield cirq.X.on(A[idx])

        #Need to un-implement |N>
        for idx,c in enumerate(modVal):
            if c=='1':
                #tmpG.append(cirq.X.on(N[idx]))
                yield cirq.X.on(N[idx])

        #debugPrint("Reorder",tmpG)
        if isControlled:
            ctlQubits = list(qubits[0:len(self.cvs)])
            yield bloqs.multi_control_multi_target_pauli.MultiControlPauli(self.cvs).on(*(ctlQubits+targetQubit))
        

        context.qubit_manager.qfree(A)
        context.qubit_manager.qfree(N)
        context.qubit_manager.qfree(t)
        if isControlled:
            context.qubit_manager.qfree(targetQubit)
