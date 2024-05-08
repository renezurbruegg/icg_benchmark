import numpy as np


def sample_top_down_view() -> tuple[float, float, float]:
    """Sample a top-down view pose.

    Returns:
        tuple[float, float, float]: The sampled top-down view pose,
        as spherical coordinates (r, theta, phi).
    """
    r = np.random.uniform(0.6, 0.75)
    theta = np.random.uniform(-np.pi / 90, -0.01)
    phi = np.random.uniform(0.0, np.pi)
    return r, theta, phi
