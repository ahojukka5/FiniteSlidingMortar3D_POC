import numpy as np
from contact3d.coupled_oracle import FrozenMatchingMortarInterface


def test_initial_normal_gap_offsets_matching_oracle_kinematics():
    interface = FrozenMatchingMortarInterface(
        np.array([0, 1, 2, 3]),
        np.array([4, 5, 6, 7]),
        np.array([0.0, 0.0, -1.0]),
        100.0,
        initial_normal_gap=-0.05,
    )
    _, gaps = interface._kinematics(np.zeros(24))
    np.testing.assert_allclose(gaps, -0.05)
    displacement = np.zeros((8, 3))
    displacement[:4, 2] = -0.06
    _, gaps = interface._kinematics(displacement.ravel())
    np.testing.assert_allclose(gaps, 0.01)
