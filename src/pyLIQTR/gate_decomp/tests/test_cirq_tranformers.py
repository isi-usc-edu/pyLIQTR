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

import cProfile
import math
import unittest
from random import random, seed
from time import perf_counter

import cirq
import numpy as np
import pkg_resources

from pyLIQTR.gate_decomp.cirq_transforms import (
    clifford_plus_t_direct_transform,
    clifford_plus_T_ops,
)
from pyLIQTR.gate_decomp.rotation_gates import rz_decomp

GSE_CIRCUIT_FILE = pkg_resources.resource_filename(
    "pyLIQTR", r"gate_decomp/tests/data/gse_h2_decomp_circuit_example.json"
)
CIRCUIT_FILE = pkg_resources.resource_filename(
    "pyLIQTR", r"gate_decomp/tests/data/test_circuit.json"
)


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
        original_circuit = cirq.read_json(CIRCUIT_FILE)
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
        original_circuit = cirq.read_json(GSE_CIRCUIT_FILE)
        new_circuit = clifford_plus_t_direct_transform(
            original_circuit, precision=1e-10
        )
        sim1 = cirq.Simulator(dtype=np.complex128, seed=1)
        sim2 = cirq.Simulator(dtype=np.complex128, seed=1)
        res1 = sim1.simulate(original_circuit)
        res2 = sim2.simulate(new_circuit)
        vec1 = res1.state_vector()
        vec2 = res2.state_vector()
        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_rz_decomp1(self):
        seed(0)
        q0 = cirq.NamedQubit("q0")
        circuit1 = cirq.Circuit()
        circuit1.append(cirq.rz(0.34).on(q0))
        circuit1 = clifford_plus_t_direct_transform(circuit1, precision=1e-10)

        seed(0)
        circuit2 = cirq.Circuit()
        circuit2.append(rz_decomp(rads=0.34).on(q0))
        circuit2 = cirq.expand_composite(
            circuit2, no_decomp=lambda g: g.gate in [cirq.H]
        )

        self.assertEqual(circuit1, circuit2)

    def test_rz_decomp2(self):
        seed(0)
        q0 = cirq.NamedQubit("q0")
        circuit = cirq.Circuit()
        circuit.append(cirq.rz(0.21 * np.pi).on(q0))
        circuit1 = clifford_plus_t_direct_transform(
            circuit, precision=1e-15, use_rotation_decomp_gates=True
        )
        circuit1 = cirq.expand_composite(
            circuit1, no_decomp=lambda g: g.gate in [cirq.H]
        )
        seed(0)
        circuit2 = clifford_plus_t_direct_transform(circuit, precision=1e-15)
        self.assertEqual(circuit1, circuit2)

    def test_rz_decomp_w_classical_controls(self):
        original_circuit = cirq.read_json(GSE_CIRCUIT_FILE)
        seed(0)
        new_circuit1 = clifford_plus_t_direct_transform(
            original_circuit, precision=1e-10, use_rotation_decomp_gates=True
        )
        new_circuit1 = cirq.expand_composite(
            new_circuit1, no_decomp=lambda g: g.gate in [cirq.H]
        )
        sim1 = cirq.Simulator(dtype=np.complex128, seed=1)
        sim2 = cirq.Simulator(dtype=np.complex128, seed=1)
        res1 = sim1.simulate(original_circuit)
        res2 = sim2.simulate(new_circuit1)
        vec1 = res1.state_vector()
        vec2 = res2.state_vector()

        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_rz_decomp_qsp_circuit(self):
        original_circuit = cirq.read_json(CIRCUIT_FILE)
        seed(0)
        new_circuit1 = clifford_plus_t_direct_transform(original_circuit)
        seed(0)
        new_circuit2 = clifford_plus_t_direct_transform(
            original_circuit, use_rotation_decomp_gates=True
        )
        new_circuit2 = cirq.expand_composite(
            new_circuit2, no_decomp=lambda g: g.gate in [cirq.H]
        )
        self.assertTrue(new_circuit1, new_circuit2)

    def test_rx_decomp_inverse_gate(self):
        q0 = cirq.NamedQubit("q0")
        circuit = cirq.Circuit()
        circuit.append(cirq.inverse(cirq.rx(0.947).on(q0)))
        new_circuit = clifford_plus_t_direct_transform(
            circuit, precision=1e-10, use_rotation_decomp_gates=True
        )
        new_circuit = cirq.expand_composite(
            new_circuit, no_decomp=lambda g: g.gate in [cirq.H]
        )
        vec1 = self.sim.simulate(circuit).state_vector()
        vec2 = self.sim.simulate(new_circuit).state_vector()
        self.assertTrue(cirq.allclose_up_to_global_phase(vec1, vec2))

    def test_rz_decomp_circuit2qasm(self):
        original_circuit = cirq.read_json(CIRCUIT_FILE)[0:2]
        seed(0)
        print("original")
        print(cirq.qasm(original_circuit))
        print()
        new_circuit2 = clifford_plus_t_direct_transform(
            original_circuit, use_rotation_decomp_gates=True
        )
        print("decomped")
        print(cirq.qasm(new_circuit2))
        print()
        print("expanded")
        new_circuit2 = cirq.expand_composite(
            new_circuit2, no_decomp=lambda g: g.gate in [cirq.H]
        )
        print(cirq.qasm(new_circuit2))

    def test_benchmark_gse_circuit(self):
        original_circuit = cirq.read_json(GSE_CIRCUIT_FILE)
        seed(0)
        time1 = perf_counter()
        new_circuit1 = clifford_plus_t_direct_transform(original_circuit)
        time2 = perf_counter()
        new_circuit2 = clifford_plus_t_direct_transform(
            original_circuit, use_rotation_decomp_gates=True
        )
        time3 = perf_counter()
        new_circuit2 = cirq.expand_composite(
            new_circuit2, no_decomp=lambda g: g.gate in [cirq.H]
        )
        time4 = perf_counter()
        print("Time original:{:.2f}s".format(time2 - time1))
        print("Time new: {:.2f}".format(time3 - time2))
        print("Time cirq expand composite: ", time4 - time3)

    def test_rz_decomp_to_qasm(self):
        seed(0)
        q0 = cirq.NamedQubit("q0")
        circuit = cirq.Circuit()
        circuit.append(cirq.rz(0.143).on(q0))
        new_circuit = clifford_plus_t_direct_transform(
            circuit, use_rotation_decomp_gates=True
        )
        qasm_string = cirq.qasm(new_circuit)
        # don't want the line in the header saying what version of cirq is used
        index = qasm_string.find("OPENQASM 2.0")
        correct_string = (
            "OPENQASM 2.0;\ninclude"
            ' "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\nrz_d(pi*0.0455183137)'
            " q[0];\n// (False, 66393976246411458480317831285, 98, [S, H, Z])"
        )
        self.assertEqual(qasm_string[index:], correct_string)

    def test_ry_decomp_to_qasm(self):
        seed(0)
        q0 = cirq.NamedQubit("q0")
        circuit = cirq.Circuit()
        circuit.append(cirq.ry(0.967).on(q0))
        new_circuit = clifford_plus_t_direct_transform(
            circuit, use_rotation_decomp_gates=True
        )
        qasm_string = cirq.qasm(new_circuit)
        index = qasm_string.find("OPENQASM 2.0")
        # whether you run this test individually or as part of the suite seems to affect
        # the random seeding for the decomposition - to deal with this just check if
        # the string is either of these.
        correct_string1 = (
            "OPENQASM 2.0;\ninclude"
            ' "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\n// Gate:'
            " Ry_d(0.3078056599397256π)\nsdg q[0];\nh q[0];\nrz_d(pi*0.3078056599)"
            " q[0];\n// (True, 648022807138153475916841871980, 101, [Z])\nh q[0];\ns"
            " q[0];"
        )
        correct_string2 = (
            "OPENQASM 2.0;\ninclude"
            ' "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\n// Gate:'
            " Ry_d(0.3078056599397256π)\nsdg q[0];\nh q[0];\nrz_d(pi*0.3078056599)"
            " q[0];\n// (True, 493997145995858371046147137373, 101, [S, Y])\nh q[0];\ns"
            " q[0];"
        )
        self.assertIn(qasm_string[index:], [correct_string1, correct_string2])

    def test_rx_decomp_to_qasm(self):
        q0 = cirq.NamedQubit("q0")
        circuit = cirq.Circuit()
        circuit.append(cirq.rx(0.246).on(q0))
        seed(0)
        new_circuit = clifford_plus_t_direct_transform(
            circuit, use_rotation_decomp_gates=True
        )
        qasm_string = cirq.qasm(new_circuit)
        index = qasm_string.find("OPENQASM 2.0")
        # whether you run this test individually or as part of the suite seems to affect
        # the random seeding for the decomposition - to deal with this just check if
        # the string is either of these.
        correct_string1 = (
            "OPENQASM 2.0;\ninclude"
            ' "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\n// Gate:'
            " Rx_d(0.0783042320012125π)\nh q[0];\nrz_d(pi*0.078304232) q[0];\n//"
            " (False, 156581822512854592575398412630, 98, [(S**-1), H, (S**-1)])\nh"
            " q[0];"
        )
        correct_string2 = (
            "OPENQASM 2.0;\ninclude"
            ' "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\n// Gate:'
            " Rx_d(0.0783042320012125π)\nh q[0];\nrz_d(pi*0.078304232) q[0];\n//"
            " (False, 244176282553465898102860881568, 98, [H, Z])\nh"
            " q[0];"
        )
        self.assertIn(qasm_string[index:], [correct_string1, correct_string2])

    def test_multiple_decomp_rotations_to_qasm(self):
        seed(0)
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()
        circuit.append(cirq.rz(0.143).on(q0))
        circuit.append(cirq.ry(0.967).on(q1))
        circuit.append(cirq.rx(0.246).on(q2))
        correct_string = (
            "OPENQASM 2.0;\ninclude"
            ' "qelib1.inc";\n\n\n// Qubits: [q(0), q(1), q(2)]\nqreg'
            " q[3];\n\n\nrz_d(pi*0.0455183137) q[0];\n// (False,"
            " 66393976246411458480317831285, 98, [S, H, Z])\n// Gate:"
            " Ry_d(0.3078056599397256π)\nsdg q[1];\nh q[1];\nrz_d(pi*0.3078056599)"
            " q[1];\n// (True, 493997145995858371046147137373, 101, [S, Y])\nh q[1];\ns"
            " q[1];\n// Gate: Rx_d(0.0783042320012125π)\nh q[2];\nrz_d(pi*0.078304232)"
            " q[2];\n// (False, 244176282553465898102860881568, 98, [H, Z])\nh q[2];"
        )
        new_circuit = clifford_plus_t_direct_transform(
            circuit, use_rotation_decomp_gates=True
        )
        qasm_string = cirq.qasm(new_circuit)
        index = qasm_string.find("OPENQASM 2.0")
        self.assertEqual(qasm_string[index:], correct_string)

    def test_random_decomp1(self):
        q0 = cirq.NamedQubit("q0")
        # mainly just want to check no errors get thrown here
        for _ in range(10):
            circuit = cirq.Circuit()
            circuit.append(
                rz_decomp(0.134, precision=1e-16, use_random_decomp=True).on(q0)
            )
            new_circuit = cirq.expand_composite(
                circuit, no_decomp=lambda g: g.gate in [cirq.H]
            )

    def test_random_decomp2(self):
        q0, q1 = cirq.LineQubit.range(2)
        for _ in range(10):
            circuit = cirq.Circuit()
            circuit.append(cirq.rz(0.312).on(q0))
            circuit.append(cirq.ry(0.123).on(q1))
            new_circuit = clifford_plus_t_direct_transform(
                circuit, use_random_decomp=True
            )

    def test_accuracy_issue(self):
        q0 = cirq.NamedQubit("q0")
        circuit = cirq.Circuit()
        circuit.append(cirq.ry(0.20483276469913342 * np.pi).on(q0))
        new_circuit = clifford_plus_t_direct_transform(circuit)


if __name__ == "__main__":
    unittest.main()
