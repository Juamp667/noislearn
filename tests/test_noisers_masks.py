import numpy as np
import pytest

from noisers.funcs import urlf


def test_urlf_can_return_noise_mask():
    y = np.array([0, 0, 1, 1, 2, 2])

    y_noisy, noise_mask = urlf(y, noise_level=0.5, random_state=7, return_mask=True)

    assert noise_mask.dtype == bool
    np.testing.assert_array_equal(noise_mask, y_noisy != y)


def test_urlf_estimator_stores_noise_mask():
    pytest.importorskip("sklearn")
    from noisers.classes import URLFNoise

    y = np.array([0, 0, 1, 1, 2, 2])
    noiser = URLFNoise(noise_level=0.5, random_state=7)

    _, y_noisy = noiser.fit_resample(np.zeros((len(y), 1)), y)

    np.testing.assert_array_equal(noiser.noise_mask_, y_noisy != y)
