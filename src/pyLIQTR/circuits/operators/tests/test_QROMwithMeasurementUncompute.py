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
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from pyLIQTR.circuits.operators.QROMwithMeasurementUncompute import QROMwithMeasurementUncompute

class TestQROMwithMeasurementUncompute:

    @pytest.mark.parametrize("data",[[2,3,4,8],[15,13,14,5,4,5,12],[6,2,8,3,1]])
    def test_QROMwithMeasurementUncompute(self,data):
        '''
        Tests the method measurement_uncompute correctly uncomputes by checking all qubits return to initial state.
        '''
        # create registers
        nData = (max(data)).bit_length()
        sel_bitsize = (len(data)-1).bit_length()
        sel_reg = cirq.NamedQubit.range(sel_bitsize,prefix='sel')
        new_sel = sel_reg[:-1]
        q_bit = sel_reg[-1]
        u_bit = cirq.NamedQubit.range(1,prefix='u')
        data_reg = cirq.NamedQubit.range(nData,prefix='data')
        # initialilze operator
        qrom_gate = QROMwithMeasurementUncompute(
            data=[np.array(data)],
            selection_bitsizes=(sel_bitsize,),
            target_bitsizes=(nData,)
            )
        operation = qrom_gate.on_registers(selection=sel_reg,target0_=data_reg)
        # circuit
        circuit = cirq.Circuit()
        ## prepare select in superposition
        # circuit.append(cirq.H.on_each(*sel_reg))
        prep = PrepareUniformSuperposition(len(data))
        circuit.append(prep.on_registers(target=sel_reg))
        ## qrom writing data
        circuit.append(operation)
        ## measurement uncompute
        circuit.append([
            qrom_gate.measurement_uncompute(selection=sel_reg,data=data_reg)
        ])
        ## unprepare select
        circuit.append(cirq.inverse(prep.on_registers(target=sel_reg)))

        # verify simulated results
        num_qubits = len(circuit.all_qubits())
        in_out_state = [0]*num_qubits
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=circuit.all_qubits(),inputs=in_out_state,outputs=in_out_state)

    @pytest.mark.parametrize("data",[[2,3,4,8]])
    def test_QROMwithMeasurementUncompute_decomposes(self,data):
        '''
        Tests gate decomposition existence.
        '''
        # create registers
        nData = (max(data)).bit_length()
        sel_bitsize = (len(data)-1).bit_length()
        sel_reg = cirq.NamedQubit.range(sel_bitsize,prefix='sel')
        new_sel = sel_reg[:-1]
        q_bit = sel_reg[-1]
        u_bit = cirq.NamedQubit.range(1,prefix='u')
        data_reg = cirq.NamedQubit.range(nData,prefix='data')
        # initialilze operator
        qrom_gate = QROMwithMeasurementUncompute(
            data=[np.array(data)],
            selection_bitsizes=(sel_bitsize,),
            target_bitsizes=(nData,)
            )
        operation = qrom_gate.on_registers(selection=sel_reg,target0_=data_reg)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose of circuit with uncompute
        circuit = cirq.Circuit()
        ## prepare select in superposition
        circuit.append(cirq.H.on_each(*sel_reg))
        ## qrom writing data
        circuit.append(operation)
        ## measurement uncompute
        circuit.append([
            qrom_gate.measurement_uncompute(selection=sel_reg,data=data_reg)
        ])
        decomposed = cirq.decompose(circuit)
        assert decomposed is not None