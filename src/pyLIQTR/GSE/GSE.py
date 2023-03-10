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
from pyLIQTR.PhaseEstimation.pe     import PhaseEstimation as PE
from pyLIQTR.PhaseEstimation.pe_sim import PE_Simulator    as pe_sim

# import cirq
import numpy  as np
import pandas as pd

class GSE():
    """
    A class for doing Ground State Estimation. Functionally, serves as a wrapper for the PhaseEstimation class.

    Attributes:
        _precision_order: (int) The precision for the calculation

        _init_state: (list(int)) A list containing either 1 or 0 for initializing each qubit in the state vector

        _num_runs: (int) The number of times to run the GSE circuit to calculate the ground state

        E_max: (float) The upper bound energy for the state in question. Note, try to set E_max and E_min such that the GSE is halfway between them.

        E_min: (float) The lower bound energy for the state in question. Note, try to set E_max and E_min such that the GSE is halfway between them.

        omega: (float) = E_max - E_min. Don't set this.

        t: (float) = 2 * pi / omega. Don't set this.

        E_s: (float) = E_max. Don't set this.

        _operator_power: (int) For raising (U**2**j)**_operator_power. Should be 1 unless explicitly needed for something.

        _phase_offset: (float) E_s * 2 * pi / omega. 

        pe_simulator: The PE_Simulator object. 
        
        df_: A pandas DataFrame where data from running the GSE simulations are stored.

    Methods:
        __init__(precision_order, init_state, E_max=10, E_min=-10, kwargs)
        
        initialize_GSE_circuit()

        do_gse()

    """
    def __init__(self, precision_order, init_state, E_max=10, E_min=-10, include_classical_bits:bool=True, **kwargs):
        """
        A function to initialize an instance of the GSE class. Runs initialize_GSE_circuit()

        Parameters:
            precision_order: (int) The precision for the calculation

            init_state: (list(int)) A list containing either 1 or 0 for initializing each qubit in the state vector

            E_max: (float) The upper bound energy for the state in question. Note, try to set E_max and E_min such that the GSE is halfway between them.

            E_min: (float) The lower bound energy for the state in question. Note, try to set E_max and E_min such that the GSE is halfway between them.

            kwargs: This is a special dict that should contain information about how to generate the unitary. See examples for more info.
        
        Returns:
            None
        """
        self._precision_order = precision_order
        self._init_state      = init_state
        self._num_runs        = 10

        self.include_classical_bits = include_classical_bits

        self._E_max       = E_max
        self._E_min       = E_min
        self.t            = 1
        self.pe_simulator = None
        
        ### 
        ### For analysis. Some of these
        ### Are deprecated
        ### (Notably, shouldn't be looking at AVERAGE)
        ###
        self.df_          = None
        self.gse_med_     = None

        # TODO: This feels like a hacky way to do this. Do it better?
        if kwargs:
            self.kwargs = kwargs['kwargs']
            if 'ev_time' in self.kwargs:
                self.kwargs['ev_time'] = self.t
                tmp = self.kwargs['ev_time']
        else:
            self.kwargs = None

        self.initialize_GSE_circuit()

    @property
    def E_max(self):
        return self._E_max

    @E_max.setter
    def E_max(self, new_val):
        self._E_max = new_val
        self.initialize_GSE_circuit()
    
    @property
    def E_min(self):
        return self._E_min

    @E_min.setter
    def E_min(self, new_val):
        self._E_min = new_val
        self.initialize_GSE_circuit()

    @property
    def num_runs(self):
        return self._num_runs

    @num_runs.setter
    def num_runs(self, new_val):
        self._num_runs = new_val
        self.initialize_GSE_circuit()

    @property
    def precision_order(self):
        return self._precision_order

    @precision_order.setter
    def precision_order(self, new_val):
        self._precision_order = new_val
        self.initialize_GSE_circuit()

    @property 
    def init_state(self):
        return self._init_state

    @init_state.setter
    def init_state(self, new_val):
        self._init_state = new_val
        self.initialize_GSE_circuit()

    @property
    def operator_power(self):
        return self._operator_power
    
    @operator_power.setter
    def operator_power(self, new_val):
        self._operator_power = new_val

    @property
    def phase_offset(self):
        return self._phase_offset
    
    @phase_offset.setter
    def phase_offset(self, new_val):
        self._phase_offset = self.E_s*self.t
        self._phase_offset += new_val

    def initialize_GSE_circuit(self):
        """
        A function to generate the GSE circuit for Simulation and/or resource estimation.

        Paramters:
            None

        Returns:
            None
        """

        ### Moved these out of __init__ -> Does that work?
        self.omega        = self.E_max - self.E_min
        self.t            = 2.*np.pi/self.omega
        self.E_s          = self.E_max# self.E_min

        self._operator_power = 1
        self._phase_offset   = self.E_s*self.t

        # TODO: This feels like a hacky way to do this. Do it better?
        if self.kwargs:
            if 'ev_time' in self.kwargs:
                self.kwargs['ev_time'] = self.t

        # These were always in here.
        self.pe_inst      = None
        self.pe_circ      = None
        self.pe_simulator = None

        self.pe_inst = PE(
            precision_order=self.precision_order,
            init_state=self.init_state,
            phase_offset=self.phase_offset,
            include_classical_bits=self.include_classical_bits,
            kwargs=self.kwargs)
        self.pe_inst.operator_power = self.operator_power
        self.pe_inst.generate_circuit()
        
        self.pe_circ = self.pe_inst.pe_circuit
        self.pe_simulator = pe_sim(
            num_runs=self.num_runs,
            phase_estimation_instance=self.pe_inst)


    def calc_gse_from_bits(self, phase_estimate:float):
        """
        Calculates gse from bits. Formula from quipper

        Parameters:
            phase_estimate: (float) The phase estimate from PE_Simulator
        
        Returns:
            gse: (float) value in Hartrees
        """
        return self.E_max - self.omega * phase_estimate
    
    def do_GSE(self, verbose:bool=False):
        """
        A function to run the GSE simulation. Uses the PE_Simulator as the backbone.

        Parameters:
            verbose: (bool) If true, print information to screen.

        Returns:
            None
        """
        phase_med = self.pe_simulator.sim_phase_estimation()

        # Note: formula from quipper
        gse_ = []
        prec_bits = []
        for pe in self.pe_simulator.df_['Phase Estimate']:
            gse_.append(self.calc_gse_from_bits(pe))
            prec_bits.append(self.precision_order)
        
        local_df = self.pe_simulator.df_.copy(deep=True)
        local_df = local_df.assign(gse=gse_)
        local_df = local_df.assign(precision_bits=prec_bits)
        if isinstance(self.df_, pd.DataFrame):
            self.df_ = pd.concat([self.df_, local_df], ignore_index=True)
        else:
            self.df_ = local_df.copy(deep=True)
        
        # Bulk calculation:
        self.gse_med_ = np.median(self.df_['gse'])

        if verbose:
            print(f'> phase_med  = {phase_med}')
            print(f'> Median gse = {self.gse_med_}')

    def do_GSE_sweep(self,prec_bits:list):
        """
        A function for sweeping over a list of precision bits.

        Parameters:
            prec_bits: (list) A list of precision bit values.

        Returns:
            None
        """
        self.df_ = None
        print(f'> ----- Doing GSE Precision Bit Sweep -----')
        for pb in prec_bits:
            print(f'> For precision bit = {pb}')
            self.precision_order = pb
            self.do_GSE()

