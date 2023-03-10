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
import pandas as pd
import tqdm

from pyLIQTR.PhaseEstimation.pe import PhaseEstimation 

class PE_Simulator():
    """
    A class for performing Phase Estimate circuit simulations

    Attributes:
        num_runs: (int) The number of simulation runs to perform.

        pe_inst: Instance of the PhaseEstimation class

        df_: Pandas DataFrame to hold the data

        med_phase: (float) The median phase from the simulated runs

        sim_run: (bool) A flag indicating whether the simulations have been run yet.

    Methods:
        __init__(num_runs:int, phase_estimation_instance:PhaseEstimation)

        extrac_info_from_sim()

        sim_phase_estimation()

        sim_run_analysis() 

    """
    def __init__(self, num_runs:int, phase_estimation_instance:PhaseEstimation):
        """
        Initializes the Phase Estimation Simulator.
        Note: Must pass active PhaseEstimation instance to simulate

        Parameters:
            num_runs : The number of runs to simulate

            phase_estimation_instance: The active phase estimation instance

        Returns:
            None

        """
        self.num_runs  = num_runs
        self.pe_inst   = phase_estimation_instance
        self.df_       = None
        self.med_phase = 0
        self.sim_run   = False

    @staticmethod
    def extract_info_from_sim(sim_result):
        '''
        A static method for extracting infromation from a sim_result

        Parameters:
            sim_result: The sim_results

        Returns:
            measurement: The measurement

            precision_bit_state: The state of the precision bit

            psi_bit_state: The state of the psi bits
        '''

        measurement = []
        for key in sim_result.measurements:
            measurement.extend(sim_result.measurements[key])
            
        precision_bit_state = 0
        psi_bit_state       = 0
        for idx, substate in enumerate(sim_result._get_substates()):
            
            if idx < 2:
                targ_tensor = substate.target_tensor
                tt_size     = np.prod(targ_tensor.shape)
                targ_vector = targ_tensor.reshape(tt_size)

                if tt_size == 2:
                    precision_bit_state = targ_vector
                else:
                    psi_bit_state = targ_vector   

        return (measurement, (precision_bit_state, psi_bit_state))


    def sim_phase_estimation(self):
        """
        A function to simulate the Phase Estimation circuit
        
        Parameters:
            None

        Returns:
            med_phase : The median phase from the number of runs
        """
        data            = []
        bit_state_list  = []
        psi_state_list  = []
        for idx in tqdm.tqdm(range(self.num_runs)):
        
            self.pe_inst.generate_circuit()
            test_circuit = self.pe_inst.pe_circuit

            simulator   = cirq.Simulator(dtype=np.complex128)
            sim_results = simulator.simulate(test_circuit)

            (mzrmnt, (pb_state, psi_state)) = self.extract_info_from_sim(sim_result=sim_results)

            self.pe_inst.bit_list.extend(mzrmnt)
            data.append([
                self.pe_inst.estimate_phase(),
                np.array(mzrmnt),
                np.array(pb_state),
                np.array(psi_state)
            ])
            self.pe_inst.bit_list.clear()
            bit_state_list.clear()
            psi_state_list.clear()

        self.df_       = pd.DataFrame(data, columns=['Phase Estimate','Bits','Precision Bit States', 'Psi Register States'])
        self.med_phase = np.median(self.df_['Phase Estimate'])
        self.sim_run   = True

        return self.med_phase

    def sim_run_analysis(self, true_phase):
        
        if not self.sim_run:
            print(' Simulations haven''t been run yet.')
        else:        
            differences_ = []
            prec_        = 1 / (2**self.pe_inst.precision_order)
            meet_thresh_ = 0

            for ii_ in range(len(self.df_)):
                tmp_diff = np.abs(self.df_.iloc[ii_]['Phase Estimate'] - true_phase)
                differences_.append(tmp_diff)
                if tmp_diff <= prec_:
                    meet_thresh_ += 1

            print(f'> -------- Sim Run Analysis Report --------')
            print(f'> There were a total of {len(self.df_)} runs.')
            print(f'> The required minimum precision was {prec_} that {8/(np.pi**2)*100:0.2f}% of runs need to meet.')
            print(f'> {meet_thresh_/len(differences_)*100:0.2f}% of events meet precision threshold')
            if (meet_thresh_/len(differences_) >= 8/(np.pi **2)):
                print(f'>\tThis is in line with the 8/pi theory prediction :)')
            else:
                print(f'>\tThis is not in line with the 8/pi theory prediction :(')
                print(f'>\tGo back and think harder.')
            median_phase_est_ = np.median(self.df_['Phase Estimate'])
            if (np.abs(median_phase_est_ - true_phase) <= prec_):
                print(f'> The median measurement {median_phase_est_} meets the precision threshold :)')
            else:
                print(f'> The median measurement {median_phase_est_} does not meet the precision threshold :(')
            print(f'>\t |{median_phase_est_} - {true_phase}| = {np.abs(median_phase_est_-true_phase)}')
        
        return differences_, prec_
