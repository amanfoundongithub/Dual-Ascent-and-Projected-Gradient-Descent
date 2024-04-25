import numpy as np
import numpy.typing as npt


# Trid function


def trid_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum((point - 1) ** 2) - np.sum(point[1:] * point[:-1])


def trid_function_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    der = np.zeros_like(point)
    for i in range(point.shape[0]):
        der[i] = 2 * (point[i] - 1)
        if i != 0:
            der[i] -= point[i - 1]
        if i != point.shape[0] - 1:
            der[i] -= point[i + 1]
    return der

# ------------------------------------------------------------------------------

def matyas_function(point: npt.NDArray[np.float64]) -> np.float64:
    return 0.26*(point[0]**2 + point[1]**2) - 0.48*point[0]*point[1] 

def matyas_function_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            0.52*point[0] -0.48*point[1],
            0.52*point[1] -0.48*point[0],
        ]
    ) 
