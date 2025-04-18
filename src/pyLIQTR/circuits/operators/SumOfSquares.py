"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import numpy as np
from functools import cached_property
from typing import Set
from qualtran import GateWithRegisters, Signature, Register, QAny, Side
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.arithmetic import OutOfPlaceAdder

class SumOf3Squares(GateWithRegisters):
    """This gate sums the squares of three :math:`n_p`-bit binary numbers using :math:`3 n_p^2 - n_p - 1` Toffoli gates. This is useful for calculating the norm of a vector, for example.

    Registers:

    .. line-block::
        input_vector: Represents a 3 component vector whose elements are to be squared and summed.
        output: The register the result is output on.
        product_ancilla: Ancilla qubits used to compute bit products.
        carry_ancilla: Ancilla qubits used for carry bits from the arithmetic.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Appendix G, Lemma 8

    :param int num_bits_p: the number of bits :math:`n_p` which is equivalent to the size of one dimension of the input vector.
    """

    def __init__(self,num_bits_p:int):
        self.num_bits_p = num_bits_p
        self.half_n = int(np.ceil(self.num_bits_p/2))
        self._even_n = self.num_bits_p % 2 == 0

    @cached_property
    def signature(self):
        return Signature([
            Register("input_vector", QAny(bitsize=self.num_bits_p), shape=(3,)),
            Register("output", QAny(bitsize=2*self.num_bits_p+2),side=Side.RIGHT),
            Register("product_ancilla", QAny(bitsize=int(3*self.num_bits_p*(self.num_bits_p-1)/2)),side=Side.RIGHT),
            Register("carry_ancilla",QAny(bitsize=int(self.num_bits_p*(3*self.num_bits_p+1)/2-3)),side=Side.RIGHT) #extra -2 since first and last carry go directly to output
        ])

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs,
    ) -> cirq.OP_TREE:

        px, py, pz = quregs['input_vector'][0],quregs['input_vector'][1],quregs['input_vector'][2]
        output = quregs['output']
        product_ancilla, carry_ancilla = quregs['product_ancilla'], quregs['carry_ancilla']

        three_bit_adder = OutOfPlaceAdder(bitsize=1)
        half_n = self.half_n

        # first two levels (2**0 and 2**1) use one Toffoli
        ## copy one to output
        yield CNOT().on(pz[0],output[0])
        ## add other two
        yield three_bit_adder.on_registers(a=px[0],b=py[0],c=[output[1],output[0]])

        output_counter = 1 # tracks element of output register
        carry_counter = 0 # tracks element of carry_ancilla register
        product_counter = 0 # tracks element of product_ancilla register
        carry_ancilla_2 = [] # stores carry bits from sigma2
        for ell in range(1,half_n):

            ### sigma1
            output_counter+=1
            bits_to_sum = [px[ell],py[ell],pz[ell]]
            #### compute bit products
            for j in range(ell):
                yield And().on(px[2*ell-1-j],px[j],product_ancilla[product_counter])
                yield And().on(py[2*ell-1-j],py[j],product_ancilla[product_counter+1])
                yield And().on(pz[2*ell-1-j],pz[j],product_ancilla[product_counter+2])
                bits_to_sum += [*product_ancilla[product_counter:product_counter+3]]
                product_counter+=3
            
            #### collect bits to sum at this level
            bits_to_sum += [*carry_ancilla_2]
            bit_pairs = zip(bits_to_sum[::2], bits_to_sum[1::2])
            carry_ancilla_1 = [] # stores carry bits from sigma 1
            for bit1,bit2 in bit_pairs:
                #### add
                yield three_bit_adder.on_registers(a=bit1,b=bit2,c=[carry_ancilla[carry_counter],output[output_counter]])
                carry_ancilla_1 += [carry_ancilla[carry_counter]]
                carry_counter += 1

            ### sigma2
            output_counter += 1
            bits_to_sum = []
            #### compute bit products
            for j in range(ell):
                yield And().on(px[2*ell-j],px[j],product_ancilla[product_counter])
                yield And().on(py[2*ell-j],py[j],product_ancilla[product_counter+1])
                yield And().on(pz[2*ell-j],pz[j],product_ancilla[product_counter+2])
                bits_to_sum += [*product_ancilla[product_counter:product_counter+3]]
                product_counter+=3
            #### collect bits to sum at this level and break into pairs to pass to the adder
            bits_to_sum += [*carry_ancilla_1]
            bit_pairs = zip(bits_to_sum[::2], bits_to_sum[1::2])
            carry_ancilla_2 = [] # stores carry bits from sigma2
            for bit1,bit2 in bit_pairs:
                #### add
                yield three_bit_adder.on_registers(a=bit1,b=bit2,c=[carry_ancilla[carry_counter],output[output_counter]])
                carry_ancilla_2 += [carry_ancilla[carry_counter]]
                carry_counter += 1

        carry_ancilla_4 = [] # stores carry bits from sigma4
        for ell in range(half_n,self.num_bits_p):
            ### sigma3
            output_counter += 1

            #### handle special cases
            if not self._even_n and ell == half_n:
                # copy one to output
                yield CNOT().on(pz[ell],output[output_counter])
                bits_to_sum = [px[ell],py[ell]]
            elif not self._even_n and ell == half_n+1:
                # copy one to output
                yield CNOT().on(pz[ell],output[output_counter])
                bits_to_sum = [px[ell],py[ell]]
            else:
                bits_to_sum = [px[ell],py[ell],pz[ell]]

            #### compute bit products
            for j in range(2*ell-self.num_bits_p,ell):
                yield And().on(px[2*ell-1-j],px[j],product_ancilla[product_counter])
                yield And().on(py[2*ell-1-j],py[j],product_ancilla[product_counter+1])
                yield And().on(pz[2*ell-1-j],pz[j],product_ancilla[product_counter+2])
                bits_to_sum += [*product_ancilla[product_counter:product_counter+3]]
                product_counter+=3

            #### collect bits to sum at this level and break into pairs to pass to the adder
            if ell == half_n:
                # if this is the first pass for sigma3, we need to include the carry bits from the final pass of sigma2
                bits_to_sum += [*carry_ancilla_2]
            bits_to_sum += [*carry_ancilla_4]
            bit_pairs = zip(bits_to_sum[::2], bits_to_sum[1::2])
            carry_ancilla_3 = [] # stores carry bits from sigma3
            for bit1,bit2 in bit_pairs:
                #### add
                yield three_bit_adder.on_registers(a=bit1,b=bit2,c=[carry_ancilla[carry_counter],output[output_counter]])
                carry_ancilla_3 += [carry_ancilla[carry_counter]]
                carry_counter += 1

            ### sigma4
            output_counter += 1
            carry_ancilla_4 = [] # stores carry bits from sigma4
            if ell < self.num_bits_p-1:
                bits_to_sum = []
                #### compute bit products
                for j in range(2*ell-self.num_bits_p+1,ell):
                    yield And().on(px[2*ell-j],px[j],product_ancilla[product_counter])
                    yield And().on(py[2*ell-j],py[j],product_ancilla[product_counter+1])
                    yield And().on(pz[2*ell-j],pz[j],product_ancilla[product_counter+2])
                    bits_to_sum += [*product_ancilla[product_counter:product_counter+3]]
                    product_counter+=3

                #### collect bits to sum at this level and break into pairs to pass to the adder
                bits_to_sum += [*carry_ancilla_3]
                if self._even_n and ell == half_n:
                    assert len(bits_to_sum) % 2 != 0
                    # copy one to output
                    bit_to_copy = bits_to_sum.pop(-1)
                    yield CNOT().on(bit_to_copy,output[output_counter])
                bit_pairs = zip(bits_to_sum[::2], bits_to_sum[1::2])
                for bit1,bit2 in bit_pairs:
                    #### add
                    yield three_bit_adder.on_registers(a=bit1,b=bit2,c=[carry_ancilla[carry_counter],output[output_counter]])
                    carry_ancilla_4 += [carry_ancilla[carry_counter]]
                    carry_counter += 1
            
        output_counter+=1

        # sum any remaining carries
        bits_to_sum = [*carry_ancilla_3,*carry_ancilla_4]
        if len(bits_to_sum) % 2 != 0:
            # if there's an odd number of bits, copy one directly to output.
            bit_to_copy = bits_to_sum.pop(-1)
            yield CNOT().on(bit_to_copy,output[-3])
        
        # break remaining bits into pairs to pass to the adder
        bit_pairs = zip(bits_to_sum[::2], bits_to_sum[1::2])
        final_carries = []
        for bit1,bit2 in bit_pairs:
            yield three_bit_adder.on_registers(a=bit1,b=bit2,c=[carry_ancilla[carry_counter],output[-3]])
            final_carries += [carry_ancilla[carry_counter]]
            carry_counter += 1
        # one final adder for the last two carry bits
        assert len(final_carries) == 2
        yield three_bit_adder.on_registers(a=final_carries[0],b=final_carries[1],c=[output[-2],output[-1]])


    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        three_bit_adder = OutOfPlaceAdder(bitsize=1)
        if not self._even_n and self.num_bits_p-self.half_n>1:
            num_cnots = 4
        elif self._even_n and self.num_bits_p-self.half_n>1:
            num_cnots = 3
        else:
            num_cnots = 2
        return {(CNOT(),num_cnots), (three_bit_adder,int(self.num_bits_p*(3*self.num_bits_p+1)/2-1)), (And(),int(3*self.num_bits_p*(self.num_bits_p-1)/2))}