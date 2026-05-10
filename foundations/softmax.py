import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = np.max(z)
        numerator = np.exp(z - max_z)
        denominator = np.sum(numerator)
        z = np.divide(numerator, denominator)
        return np.round(z, 4)
