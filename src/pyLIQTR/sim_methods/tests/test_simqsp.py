"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
import pytest
import numpy as np
import numpy.linalg as nla
from numpy import int8 as int8
import scipy.linalg as sla
import scipy.special as sfn
import mpmath as mpm
import matplotlib.pyplot as plt
from pyLIQTR.sim_methods.fitter import Fitter

import cirq 

from pyLIQTR.QSP import gen_qsp as qspFuncs
from pyLIQTR.QSP import QSP as pQSP
from pyLIQTR.QSP.Hamiltonian import Hamiltonian as pyH
from pyLIQTR.QSP.gen_qsp import QSP_Simulator
from pyLIQTR.QSP.qsp_helpers import get_state_vector, time_step_comparison_mpe
from pyLIQTR.sim_methods.angler import Angler
from pyLIQTR.sim_methods.simqsp import SimQSP
from pyLIQTR.sim_methods.expander import Expander

import pyLIQTR.sim_methods.quantum_ops as qo
import pyLIQTR.utils.plot_helpers as ph

# from matplotlib import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
mpm.mp.prec = 512
plt.rcParams['text.usetex'] = True
H = 0.5*np.kron(qo.pz,qo.pz) + 0.25*(np.kron(qo.px,qo.id) -  np.kron(qo.id,qo.px))

class TestQspSim:
    @pytest.fixture(scope="class")
    def generated_qsp(self, generated_angler):
        # Use the generated Angles to generate a QSP that we will use for future testing
        print("")

    @pytest.fixture(scope="class")
    def generated_angler(self):
        tau_0 = 0.1
        evals, evecs = nla.eig(H)
        e_min = np.min(evals)
        e_max = np.max(evals)
        ID_n = np.eye(4)
        H_rs = (H - e_min * ID_n) / (e_max - e_min)  # min/max rescaling
        tau = tau_0 * (e_max - e_min) / 2.0
        eps0 = 1e-6
        f_cos = lambda x, tau: np.cos(tau * x)
        p_cos = Expander()
        p_cos.ja_cos(tau, eps0)

        generated_angler = Angler( cheb_poly=p_cos, bfgs_conv=1e-14, max_iter=400 )
        yield generated_angler
        del generated_angler

    def test_qsp_sequence_valid(self):
        # Test the QSP sequence creation method to make sure it handles valid data
        print("")

    @pytest.mark.run_this_test
    def test_qsp_simulation_success(self):
        # Let's test the entire logic of the QSP simulator. NOTE: The first run seems to take a very long time, but
        # after that it runs in less than a minute.
        np.set_printoptions(precision=2)
        N = 3
        k = 2.0
        nu = 0.0
        alpha = 0.6
        dt = 0.1
        tmax = 5
        sclf = 1
        Np = int(np.ceil(tmax / dt))
        timestep_vec = np.arange(0, tmax + dt, sclf * dt)
        required_precision = 1e-2
        occ_state = np.zeros(N)
        occ_state[0] = 1
        ham_strings = qo.hamiltonian_wfn_vlasov_hermite_linear_sym_string(k, alpha, nu, N)
        qsp_H = pyH(ham_strings)
        tmp = [qspFuncs.get_phis(qsp_H, simtime=t, req_prec=required_precision) for t in timestep_vec]
        tolerances = [a[1] for a in tmp]
        angles = [a[0] for a in tmp]
        qsp_generator = pQSP.QSP(phis=angles[1], hamiltonian=qsp_H, target_size=qsp_H.problem_size)
        initial_state_circuit = cirq.Circuit()
        for idx, state in enumerate(occ_state[::-1]):
            if state == 1:
                initial_state_circuit.append(cirq.X.on(qsp_generator.target[idx]))
            else:
                initial_state_circuit.append(cirq.I.on(qsp_generator.target[idx]))
        qsp_sim = QSP_Simulator(
            timestep_vec=timestep_vec,
            angles=angles,
            init_state=initial_state_circuit,
            qsp_H=qsp_H)

        sim_results = qsp_sim.do_sim()
        print(sim_results)