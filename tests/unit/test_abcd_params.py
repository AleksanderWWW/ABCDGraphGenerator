import unittest

import numpy as np
import pytest

from abcd_graph_generator.abcd_params import ABCDParams


class TestABCDParams(unittest.TestCase):
    def test_validation_w_and_s(self):
        with pytest.raises(ValueError):
            ABCDParams(w=np.array([1, 2, 3]), s=np.array([1, 1, 2]), mu=0.5)

    def test_validation_mu(self):
        with pytest.raises(ValueError):
            ABCDParams(
                w=np.array([1, 2, 3]),
                s=np.array([1, 1, 1]),
                mu=5,
            )

    def test_validation_xi(self):
        with pytest.raises(ValueError):
            ABCDParams(
                w=np.array([1, 2, 3]),
                s=np.array([1, 1, 2]),
                xi=10,
            )

        with pytest.raises(ValueError):
            ABCDParams(
                w=np.array([1, 2, 3]), s=np.array([1, 1, 2]), xi=0.5, is_local=True
            )

    def test_validation_xi_and_mu(self):
        with pytest.raises(ValueError):
            ABCDParams(
                w=np.array([1, 2, 3]),
                s=np.array([1, 1, 1]),
            )

        with pytest.raises(ValueError):
            ABCDParams(
                w=np.array([1, 2, 3]),
                s=np.array([1, 1, 1]),
                mu=0.5,
                xi=0.3,
            )

    def test_minimal_validation_succeeds(self):
        ABCDParams(w=np.array([1, 2, 3]), s=np.array([1, 1, 1]), mu=0.5)

        ABCDParams(w=np.array([1, 2, 3]), s=np.array([1, 1, 1]), xi=0.7)

    def test_sort(self):
        p1 = ABCDParams(
            w=np.array([1, 2, 3]), s=np.array([0, 2, 1]), mu=0.5, has_outliers=False
        )

        assert p1.w.tolist() == [3, 2, 1]
        assert p1.s.tolist() == [2, 1, 0]

        p2 = ABCDParams(
            w=np.array([1, 2, 3, 0, 7]),
            s=np.array([0, 2, 1, 2]),
            mu=0.5,
            has_outliers=True,
        )
        assert p2.w.tolist() == [7, 3, 2, 1, 0]
        assert p2.s.tolist() == [2, 1]
