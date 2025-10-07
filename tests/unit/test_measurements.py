import numpy as np
import pytest

from src.measurements import Measurements


@pytest.mark.parametrize(
    ("measurements", "expected_value"),
    [
        (np.array([1.0, 2.0, 3.0]), 2.0),
        (np.array([1]), 1),
        (np.array([100, -100, 0]), 0),
        (np.array([0, -0.5, 0.75, 0.2, -0.66]), -0.042),
    ],
)
def test_expected_value(measurements: np.ndarray, expected_value: float) -> None:
    m = Measurements(measurements=measurements, instrument_uncertainty=np.float64(0.1))
    assert m.expected_value == pytest.approx(expected_value)


@pytest.mark.parametrize(
    ("measurements", "standard_deviation"),
    [
        (np.array([2, 2]), 0),
        (np.array([100, -100, 0]), 100),
        (np.array([2, 2.5, -80, 1.8, 2.3, 2]), 33.5263),
    ],
)
def test_standard_deviation(measurements: np.ndarray, standard_deviation: float) -> None:
    m = Measurements(measurements=measurements, instrument_uncertainty=np.float64(0.1))
    assert m.standard_deviation == pytest.approx(standard_deviation, rel=1e-3)


@pytest.mark.parametrize(
    ("measurements", "standard_error_of_the_mean"),
    [
        (np.array([2, 2]), 0),
        (np.array([100, -100, 0]), 57.735),
        (np.array([2, 2.5, -80, 1.8, 2.3, 2]), 13.687),
    ],
)
def test_standard_error_of_the_mean(measurements: np.ndarray, standard_error_of_the_mean: float) -> None:
    m = Measurements(measurements=measurements, instrument_uncertainty=np.float64(0.1))
    assert m.standard_error_of_the_mean == pytest.approx(standard_error_of_the_mean, rel=1e-3)
