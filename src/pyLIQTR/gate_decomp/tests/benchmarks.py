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

import cirq
from random import seed, randint
from time import perf_counter
import pkg_resources
import gmpy2
from gmpy2 import mpfr
import numpy as np
import cProfile, pstats, io
from pstats import SortKey

from pyLIQTR.gate_decomp.cirq_transforms import (
    clifford_plus_t_direct_transform,
)

from pyLIQTR.gate_decomp.gate_approximation import (
    get_ring_elts_direct,
    approximate_rz_direct,
    approximate_rz_fallback,
)

GSE_CIRCUIT_FILE = pkg_resources.resource_filename(
    "pyLIQTR", r"gate_decomp/tests/data/gse_h2_decomp_circuit_example.json"
)


def estimate_t_count_direct(num_samples):
    precisions = np.array([10, 20, 30, 40, 50])
    avg_t_counts = np.zeros(precisions.shape)
    num_samples = 50
    denominator = 7919  # use a prime denominator so we don't end up approximating a multiple of pi/4
    for i, p in enumerate(precisions):
        total_t_count = 0
        for _ in range(num_samples):
            numerator = randint(1, 2 * 7919 - 1)
            total_t_count += approximate_rz_direct(numerator, denominator, int(p))[1]
        avg_t_counts[i] = total_t_count / num_samples
    prec = np.array([1e-10, 1e-20, 1e-30, 1e-40, 1e-50])
    # perform a linear regression
    A = np.vstack([np.log2(1 / prec), np.ones(len(prec))]).T
    m, c = np.linalg.lstsq(A, avg_t_counts, rcond=None)[0]
    print("Direct T count = {:.2f} x log2(1/eps) + {:.2f}".format(m, c))


def estimate_t_count_fallback(num_samples):
    precisions = np.array([10, 20, 30, 40, 50])
    r = mpfr("0.9999")
    avg_t_counts = np.zeros(precisions.shape)
    num_samples = 10
    denominator = 7919  # use a prime denominator so we don't end up approximating a multiple of pi/4
    for i, p in enumerate(precisions):
        total_t_count = 0
        for _ in range(num_samples):
            numerator = randint(1, 2 * 7919 - 1)
            _, primary_t_count, _, fallback_t_count = approximate_rz_fallback(
                numerator, denominator, int(p), r
            )
            total_t_count += (
                float(r) * primary_t_count + (1 - float(r)) * fallback_t_count
            )
        avg_t_counts[i] = total_t_count / num_samples
    prec = np.array([1e-10, 1e-20, 1e-30, 1e-40, 1e-50])
    # perform a linear regression
    A = np.vstack([np.log2(1 / prec), np.ones(len(prec))]).T
    m, c = np.linalg.lstsq(A, avg_t_counts, rcond=None)[0]
    print("Fallback T count = {:.2f} x log2(1/eps) + {:.2f}".format(m, c))


def benchmark_random_angles(prec, num_samples):
    PI = gmpy2.const_pi()
    denominator = 7919
    start = perf_counter()
    for _ in range(num_samples):
        numerator = randint(1, 2 * 7919 - 1)
        u, t, k = get_ring_elts_direct(numerator * PI / denominator, prec)
    end = perf_counter()
    print(
        "Time to decompose {} random angles at precision = {}: {:.3f}s".format(
            num_samples, prec, end - start
        )
    )


def benchmark_gse_circuit():
    original_circuit = cirq.read_json(GSE_CIRCUIT_FILE)
    seed(0)
    time1 = perf_counter()
    new_circuit2 = clifford_plus_t_direct_transform(
        original_circuit, use_rotation_decomp_gates=True
    )
    time2 = perf_counter()
    print("Time to decompose GSE circuit: {:.3f}".format(time2 - time1))


def main():
    benchmark_random_angles(10, 100)
    benchmark_random_angles(30, 100)
    benchmark_random_angles(50, 100)
    estimate_t_count_direct(100)
    estimate_t_count_fallback(100)
    benchmark_gse_circuit()
    # pr = cProfile.Profile()
    # pr.enable()
    # benchmark_random_angles(10, 100)
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())



if __name__ == "__main__":
    main()
