import numpy as np
import numpy.typing as npt

from functions import (
    trid_function,
    trid_function_derivative,
    matyas_function,
    matyas_function_derivative
)

PGD_test_cases = [
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-3., -3.5, 2., -4.]),
        "l_2",                                      # constraint type
        ([-1., -1., -1., -1.], 13.0)                # constraints
    ],
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-3., -3.5, 2., -4.]),
        "l_2",                                      
        ([-1., -1., -1., -1.], 10.0)                
    ],
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-3., 0.]),
        "linear",
        np.asarray([[-4., -4.], [2., 2.]])

    ],
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-3., 0.]),
        "linear",
        np.asarray([[-4., -4.], [1.5, 1.5]])
    ],
    [
        trid_function,
        trid_function_derivative,
        np.asarray([1.5, -1.5]),
        "l_2",
        (np.asarray([0., 0.]), 4.)            
    ],
    [
        matyas_function,
        matyas_function_derivative,
        np.asarray([3., 1.5]),
        "linear",
        np.asarray([[0., 0.], [5., 5.]])
    ],
    [
        matyas_function,
        matyas_function_derivative,
        np.asarray([7., -6.]),
        "linear",
        np.asarray([[-10., -10.], [10., 10.]])
    ],
    [
        matyas_function,
        matyas_function_derivative,
        np.asarray([-2., 1.5]),
        "l_2",
        (np.asarray([0., 0.]), 4.)
    ],
    [
        matyas_function,
        matyas_function_derivative,
        np.asarray([1.5, -0.5]),
        "l_2",
        (np.asarray([1., 0.]), 1)
    ],
]


def c1(point: npt.NDArray[np.float64]) -> np.float64:
    return -1 - point[0]


def c2(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 1


def c3(point: npt.NDArray[np.float64]) -> np.float64:
    return -1 - point[1]


def c4(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 1

# ------------------------------------------------------------------------------

def c5(point: npt.NDArray[np.float64]) -> np.float64:
    return 0 - point[0]


def c6(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 3


def c7(point: npt.NDArray[np.float64], ) -> np.float64:
    return 0 - point[1]


def c8(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 3

# ------------------------------------------------------------------------------

def c9(point: npt.NDArray[np.float64]) -> np.float64:
    return 3 - point[0]


def c10(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 4


def c11(point: npt.NDArray[np.float64], ) -> np.float64:
    return 3 - point[1]


def c12(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 4

# ------------------------------------------------------------------------------

def c13(point: npt.NDArray[np.float64]) -> np.float64:
    return 0 - point[0]


def c14(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 1


def c15(point: npt.NDArray[np.float64], ) -> np.float64:
    return 0 - point[1]


def c16(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 1

# ------------------------------------------------------------------------------

def c17(point: npt.NDArray[np.float64]) -> np.float64:
    return 1 - point[0]


def c18(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 2


def c19(point: npt.NDArray[np.float64], ) -> np.float64:
    return 1 - point[1]


def c20(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 2

# ------------------------------------------------------------------------------

def c21(point: npt.NDArray[np.float64]) -> np.float64:
    return -1 - point[0]


def c22(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] + 0.5


def c23(point: npt.NDArray[np.float64], ) -> np.float64:
    return -0.5 - point[1]


def c24(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 0.5

# ------------------------------------------------------------------------------

def d_c1(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([-1, 0])


def d_c2(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([1, 0])


def d_c3(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([0, -1])


def d_c4(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([0, 1])

# ------------------------------------------------------------------------------

def c25(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0]**2 - 2*point[1]

def d_c25(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([2*point[0], -2])

def c26(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0]**2 - point[1]**2 + 1

def d_c26(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([2*point[0], -2*point[1]])

def c27(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0]**2 - point[1]**2

def d_c27(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([2*point[0], -2*point[1]])

# def c28(point: npt.NDArray[np.float64]) -> np.float64:
#     return point[0]**3 - point[1]**2 + 1

# def d_c28(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     return np.asarray([3*point[0]**2, -2*point[1]])

# def c29(point: npt.NDArray[np.float64]) -> np.float64:
#     return point[0] + point[1]

# def d_c29(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     return np.asarray([1., 1.])

DA_test_cases = [
    [
        trid_function,
        trid_function_derivative,
        [c25],
        [d_c25],
        np.asarray([-0.7, 0.0])
    ],
    [
        trid_function,
        trid_function_derivative,
        [c26],
        [d_c26],
        np.asarray([-0.7, 0.0])
    ],
    [
        trid_function,
        trid_function_derivative,
        [c1, c2, c3, c4],
        [d_c1, d_c2, d_c3, d_c4],
        np.asarray([-0.7, 0.0])
    ],
    [
        trid_function,
        trid_function_derivative,
        [c5, c6, c7, c8],
        [d_c1, d_c2, d_c3, d_c4],
        np.asarray([0., 1.])
    ],
    [
        trid_function,
        trid_function_derivative,
        [c9, c10, c11, c12],
        [d_c1, d_c2, d_c3, d_c4],
        np.asarray([3., 3.5])
    ],
    [
        matyas_function,
        matyas_function_derivative,
        [c13, c14, c15, c16],
        [d_c1, d_c2, d_c3, d_c4],
        np.asarray([1.5, 1.5])
    ],
    [
        matyas_function,
        matyas_function_derivative,
        [c17, c18, c19, c20],
        [d_c1, d_c2, d_c3, d_c4],
        np.asarray([1., 2.5])
    ],
    [
        matyas_function,
        matyas_function_derivative,
        [c21, c22, c23, c24],
        [d_c1, d_c2, d_c3, d_c4],
        np.asarray([-0.5, 0.5])
    ]
]