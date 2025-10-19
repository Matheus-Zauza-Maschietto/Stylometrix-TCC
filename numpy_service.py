import numpy as np

class NumpyService:
    @staticmethod
    def to_float64_list(series) -> list:
        """Convert a pandas Series to a list of float64 values."""
        return np.array(series.values, dtype=np.float64).tolist()
