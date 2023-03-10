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

import unittest
import cirq
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from time import perf_counter
from random import random
import math
import numpy as np
from numpy.random import default_rng


class TestCirqTransforms(unittest.TestCase):
    def setUp(self):
        self.sim = cirq.Simulator(dtype=np.complex128)

    # Some timing tests - easy way to check that changes aren't dramatically increasing
    # the runtime
    def test_profile_prec10(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        times = []
        for _ in range(50):
            original_circuit = cirq.Circuit()
            angle = random() * 2 * math.pi
            original_circuit.append(cirq.H(q0))
            original_circuit.append(cirq.rz(angle).on(q0))
            start = perf_counter()
            new_circuit = clifford_plus_t_direct_transform(original_circuit)
            end = perf_counter()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        self.assertLessEqual(
            avg_time,
            0.1,
            "Avg time over 50 decomps was greater than 0.1s (prec=10)",
        )

    def test_profile_prec15(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        times = []
        for _ in range(50):
            original_circuit = cirq.Circuit()
            angle = random() * 2 * math.pi
            original_circuit.append(cirq.H(q0))
            original_circuit.append(cirq.rz(angle).on(q0))
            start = perf_counter()
            new_circuit = clifford_plus_t_direct_transform(
                original_circuit, precision=15
            )
            end = perf_counter()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        self.assertLessEqual(
            avg_time,
            0.05,
            "Avg time over 50 decomps was greater than 0.05s (prec=15)",
        )

    def test_single_qubit_z_rotation(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.H(q0))
        original_circuit.append(cirq.rz(0.43298).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.H(q0))
        original_circuit.append(cirq.rz(-1.87069546).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_single_qubit_x_rotation(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.rx(0.1234).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.rx(-1.456897203).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_single_qubit_y_rotation(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.ry(0.1234).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.ry(-1.456897203).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_multi_axis_rotation_prec10(self):
        q0 = cirq.NamedQubit("q0")
        precision = 10
        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.ry(0.1234).on(q0))
        original_circuit.append(cirq.rx(-1.9687).on(q0))
        original_circuit.append(cirq.rz(0.834).on(q0))
        original_circuit.append(cirq.rx(-2.9687).on(q0))
        original_circuit.append(cirq.rz(1.7896).on(q0))
        original_circuit.append(cirq.ry(0.3421).on(q0))
        original_circuit.append(cirq.rx(-2.3241).on(q0))
        original_circuit.append(cirq.rz(-0.4312).on(q0))
        original_circuit.append(cirq.rx(1.3241).on(q0))
        original_circuit.append(cirq.rz(1.7896).on(q0))
        num_original_gates = 10
        new_circuit = clifford_plus_t_direct_transform(original_circuit, precision)
        max_error = num_original_gates * (2 * 10**-precision)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()

        self.assertLessEqual(abs((abs(vec1[0]) ** 2 - abs(vec2[0]) ** 2)), max_error)
        self.assertLessEqual(abs((abs(vec1[1]) ** 2 - abs(vec2[1]) ** 2)), max_error)

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_multi_axis_rotation_prec13(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        precision = 13
        original_circuit.append(cirq.ry(0.1234).on(q0))
        original_circuit.append(cirq.rx(-1.9687).on(q0))
        original_circuit.append(cirq.rz(0.834).on(q0))
        original_circuit.append(cirq.rx(-2.9687).on(q0))
        original_circuit.append(cirq.rz(1.7896).on(q0))
        original_circuit.append(cirq.ry(0.3421).on(q0))
        original_circuit.append(cirq.rx(-2.3241).on(q0))
        original_circuit.append(cirq.rz(-0.4312).on(q0))
        original_circuit.append(cirq.rx(1.3241).on(q0))
        original_circuit.append(cirq.rz(1.7896).on(q0))
        new_circuit = clifford_plus_t_direct_transform(original_circuit, precision)
        num_original_gates = 10
        max_error = num_original_gates * (2 * 10**-precision)

        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()
        self.assertLessEqual(abs((abs(vec1[0]) ** 2 - abs(vec2[0]) ** 2)), max_error)
        self.assertLessEqual(abs((abs(vec1[1]) ** 2 - abs(vec2[1]) ** 2)), max_error)

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_qsp_circuit(self):
        original_circuit = cirq.read_json("data/test_circuit.json")
        num_original_rotations = 0
        precision = 10
        for moment in original_circuit:
            for op in moment:
                if "Rx" in str(op) or "Ry" in str(op) or "Rz" in str(op):
                    num_original_rotations += 1
        max_error = num_original_rotations * (2 * 10**-precision)
        new_circuit = clifford_plus_t_direct_transform(original_circuit)
        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()
        for i in range(len(vec1)):
            self.assertLessEqual(
                abs((abs(vec1[i]) ** 2 - abs(vec2[i]) ** 2)), max_error
            )

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_XPow_gates(self):
        q0, q1, q2 = cirq.LineQubit.range(3)
        original_circuit = cirq.Circuit()
        original_circuit.append(cirq.X.on(q0) ** 0.4254)
        original_circuit.append(cirq.X.on(q1) ** -1.7438)
        original_circuit.append(cirq.X.on(q2) ** 2.782)
        original_circuit.append(cirq.rz(0.437268).on(q1))
        original_circuit.append(cirq.Y.on(q1) ** 2.782)
        original_circuit.append(cirq.Z.on(q1) ** 0.782)
        transformed_circuit = clifford_plus_t_direct_transform(original_circuit)
        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(transformed_circuit).state_vector()
        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_gse_circuit(self):
        original_circuit = cirq.read_json("data/gse_h2_decomp_circuit_example.json")
        new_circuit = clifford_plus_t_direct_transform(original_circuit)
        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()
        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_circuit_precision(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        precision = 5
        rng = np.random.default_rng(1)
        for _ in range(50):
            original_circuit.append(cirq.rx((2 * rng.random() - 1) * np.pi).on(q0))
            original_circuit.append(cirq.ry((2 * rng.random() - 1) * np.pi).on(q0))
            original_circuit.append(cirq.rz((2 * rng.random() - 1) * np.pi).on(q0))
        new_circuit = clifford_plus_t_direct_transform(
            original_circuit, circuit_precision=precision
        )
        max_error = 2 * 10**-precision
        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()
        self.assertLessEqual(abs((abs(vec1[0]) ** 2 - abs(vec2[0]) ** 2)), max_error)
        self.assertLessEqual(abs((abs(vec1[1]) ** 2 - abs(vec2[1]) ** 2)), max_error)

    def test_arb_precision(self):
        q0 = cirq.NamedQubit("q0")
        original_circuit = cirq.Circuit()
        precision = 5.648e-14
        original_circuit.append(cirq.rz(1.734).on(q0))
        new_circuit = clifford_plus_t_direct_transform(
            original_circuit, precision=precision
        )
        max_error = 2 * precision
        vec1 = self.sim.simulate(original_circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()
        self.assertLessEqual(abs((abs(vec1[0]) ** 2 - abs(vec2[0]) ** 2)), max_error)
        self.assertLessEqual(abs((abs(vec1[1]) ** 2 - abs(vec2[1]) ** 2)), max_error)


if __name__ == "__main__":
    unittest.main()
