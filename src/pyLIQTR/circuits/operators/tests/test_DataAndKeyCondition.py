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

from pyLIQTR.circuits.operators.FixupTableQROM import DataAndKeyCondition

class TestDataAndKeyCondition:

    @pytest.mark.parametrize("data",[0,1,2,5,15])
    def test_DataAndKeyCondition_resolve_int(self,data):
        '''
        Tests condition resolves as expected for integer type data
        '''
        measurement_key = 'test'
        if data != 0:
            num_bits = (data).bit_length()
        else:
            num_bits = 1
        init_key_condition = DataAndKeyCondition(key=measurement_key,data=data,max_meas_bits=num_bits)

        measurement_vals = np.random.randint(low=0,high=2,size=(5,num_bits)) # test 5 random measurements
        for val in measurement_vals:
            measurement_int = int(''.join(str(b) for b in val), 2)
            num_ones = (measurement_int & data).bit_count()

            classical_data = cirq.ClassicalDataDictionaryStore(_records={measurement_key: [list(val)]})

            assert (num_ones%2 == 1) == init_key_condition.resolve(classical_data)

    @pytest.mark.parametrize("data",[[1],[0,0],[1,0],[0,1,1],[1,0,1,0],[1,1,1,1,1,1,1,1,1]])
    def test_DataAndKeyCondition_resolve_array(self,data):
        '''
        Tests condition resolves as expected for array type data
        '''
        measurement_key = 'test'
        num_bits = len(data)
        init_key_condition = DataAndKeyCondition(key=measurement_key,data=np.array(data),max_meas_bits=num_bits)

        measurement_vals = np.random.randint(low=0,high=2,size=(5,num_bits)) # test 5 random measurements
        for val in measurement_vals:
            num_ones = sum(val & data)

            classical_data = cirq.ClassicalDataDictionaryStore(_records={measurement_key: [list(val)]})

            assert (num_ones%2 == 1) == init_key_condition.resolve(classical_data)
        