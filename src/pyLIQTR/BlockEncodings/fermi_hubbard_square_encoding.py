"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import  numpy as np

import  cirq
import  qualtran                     as  qt
import  qualtran.cirq_interop.testing  as  qt_test

import  qualtran.bloqs.chemistry.hubbard_model.qubitization as  qt_hm
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from functools import cached_property
import attrs


from    pyLIQTR.BlockEncodings                 import   VALID_ENCODINGS
from    pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding_select_prepare
from    pyLIQTR.circuits.operators.prepare_FermiHubbard import PrepareHubbardPYL,PrepareHubbardPYL_invert_workaround






@attrs.frozen
class _SelectHubbardZeroControl(SelectOracle):
    x_dim: int
    y_dim: int

    @cached_property
    def uncontrolled(self):
        return qt_hm.SelectHubbard(self.x_dim, self.y_dim)

    @cached_property
    def selection_registers(self):
        return self.uncontrolled.selection_registers

    @cached_property
    def target_registers(self):
        return self.uncontrolled.target_registers

    @cached_property
    def control_registers(self):
        return (qt.Register("control", qt.QBit()),)

    def decompose_from_registers(self, *, context, **quregs):
        control = quregs.pop("control")
        yield self.uncontrolled.controlled(qt.CtrlSpec(cvs=0)).on_registers(
            ctrl=control, **quregs
        )


class fermi_hubbard_square_encoding(BlockEncoding_select_prepare):

    def __init__(self,ProblemInstance, **kwargs):

        super().__init__(ProblemInstance,**kwargs)

        if (self.PI._model != 'Fermi-Hubbard Model - SquareLattice(regular)'):
            raise NotImplementedError

        self.dims = ProblemInstance.shape

        self._encoding_type  =  VALID_ENCODINGS.FermiHubbardSquare

        if self._control_val == 0:
            self._select_gate = _SelectHubbardZeroControl(*self.dims)
        else:
            self._select_gate = qt_hm.SelectHubbard(
                x_dim=self.dims[0], y_dim=self.dims[1], control_val=self._control_val
            )
   
        self._prepare_gate   =  PrepareHubbardPYL( x_dim=self.dims[0], 
                                                        y_dim=self.dims[1], 
                                                        t=-ProblemInstance.J,
                                                        u=ProblemInstance.U )
        self._inverse_prepare_workaround = PrepareHubbardPYL_invert_workaround( x_dim=self.dims[0], 
                                                        y_dim=self.dims[1], 
                                                        t=-ProblemInstance.J,
                                                        u=ProblemInstance.U )
        

    @property
    def alpha(self):
        N  =  2*np.prod(self.PI.shape)
        J  =  np.abs(self.PI.J)
        U  =  np.abs(self.PI.U)
        alpha = 2*N*J + U*N/2
        return (alpha)
