import math
from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass
class Measurements:
    """
    Measurements manager data class.

    Manages the measurements data and properties.
    """

    measurements: np.ndarray

    instrument_uncertainty: np.float64

    @property
    def xs(self) -> np.ndarray:
        """Measurements array."""
        return self.measurements

    @property
    def N(self) -> int:  # noqa: N802 invalid-function-name
        """Number of measurements."""
        return len(self.measurements)

    @property
    def expected_value(self) -> np.float64:
        """Expected value of the measurements."""
        return cast("np.float64", np.mean(self.measurements))

    @property
    def standard_deviation(self) -> np.float64:
        """Standard deviation of the measurements."""
        return cast("np.float64", np.std(self.measurements, ddof=1))

    @property
    def general_uncertainty(self) -> np.float64:
        """General uncertainty of the measurements."""
        return cast("np.float64", np.sqrt(self.instrument_uncertainty**2 + self.standard_deviation**2 / self.N))

    @property
    def standard_error_of_the_mean(self) -> np.float64:
        """Standard error of the mean of the measurements."""
        return self.standard_deviation / math.sqrt(self.N)
