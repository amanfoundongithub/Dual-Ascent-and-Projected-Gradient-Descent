from typing import Callable, Literal
import sys

import numpy.typing as npt
from numpy.typing import NDArray
import numpy as np

from algos import projected_gd, dual_ascent


# -------------------------------------------------------------------------------
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
# Matyas function
def matyas_function(point: npt.NDArray[np.float64]) -> np.float64:
    return 0.26 * (point[0] ** 2 + point[1] ** 2) - 0.48 * point[0] * point[1]


def matyas_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            0.52 * point[0] - 0.48 * point[1],
            0.52 * point[1] - 0.48 * point[0],
        ]
    )


# ------------------------------------------------------------------------------
# PGD test cases

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
        np.asarray([[2., 2.], [5., 5.]])
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
        (np.asarray([1.5, 1.5]), 1)
    ],
]


# ------------------------------------------------------------------------------
# Dual ascent constraints
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


def c7(
    point: npt.NDArray[np.float64],
) -> np.float64:
    return 0 - point[1]


def c8(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 3


# ------------------------------------------------------------------------------


def c9(point: npt.NDArray[np.float64]) -> np.float64:
    return 3 - point[0]


def c10(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 4


def c11(
    point: npt.NDArray[np.float64],
) -> np.float64:
    return 3 - point[1]


def c12(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 4


# ------------------------------------------------------------------------------


def c13(point: npt.NDArray[np.float64]) -> np.float64:
    return 0 - point[0]


def c14(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 1


def c15(
    point: npt.NDArray[np.float64],
) -> np.float64:
    return 0 - point[1]


def c16(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 1


# ------------------------------------------------------------------------------


def c17(point: npt.NDArray[np.float64]) -> np.float64:
    return 1 - point[0]


def c18(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] - 2


def c19(
    point: npt.NDArray[np.float64],
) -> np.float64:
    return 1 - point[1]


def c20(point: npt.NDArray[np.float64]) -> np.float64:
    return point[1] - 2


# ------------------------------------------------------------------------------


def c21(point: npt.NDArray[np.float64]) -> np.float64:
    return -1 - point[0]


def c22(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] + 0.5


def c23(
    point: npt.NDArray[np.float64],
) -> np.float64:
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
    return point[0] ** 2 - 2 * point[1]


def d_c25(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([2 * point[0], -2])


def c26(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] ** 2 - point[1] ** 2 + 1


def d_c26(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([2 * point[0], -2 * point[1]])


def c27(point: npt.NDArray[np.float64]) -> np.float64:
    return point[0] ** 2 - point[1] ** 2


def d_c27(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray([2 * point[0], -2 * point[1]])


# ------------------------------------------------------------------------------
# Dual testcases
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


def check_constraints_pgd(
    point: npt.NDArray[np.float64],
    f: Callable[[npt.NDArray[np.float64]], float | np.float64],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    constraint: NDArray[np.float64] | tuple[NDArray[np.float64], float],
):
    tol = 1e-2
    print(f"Point: {np.round(point, 3)}")
    print(f"Value at point: {f(point)}")
    print(f"Gradient norm at point: {np.linalg.norm(d_f(point))}")
    if isinstance(constraint, tuple):
        distance = np.linalg.norm(constraint[0] - point)
        print("L2 constraint")
        print(f"Center: {np.round(constraint[0], 3)}")
        print(f"Max distance allowed: {constraint[1]}")
        print(f"Distance is {round(distance, 3)}")
        print(f"Is the constraint violated: {distance > constraint[1]+tol}")
        print("_______________________________________________________________")
    else:
        lb_violated: npt.NDArray[np.bool_] = constraint[0] > point + tol
        ub_violated: npt.NDArray[np.bool_] = constraint[1] < point - tol
        print(f"Box constraint")
        for i in range(point.shape[0]):
            print(
                f"Lower bound and Upper bound for dimension {i}: {np.round(constraint[0][i], 3)} and {np.round(constraint[1][i], 3)}"
            )
            print(f"Does the point violate lower bound: {lb_violated[i]}")
            print(f"Does the point violate upper bound: {ub_violated[i]}")
        print(
            f"Does the point satisfy all the constraints: {np.all(np.logical_or(np.logical_not(lb_violated), np.logical_not(ub_violated)))}"
        )
        print("_______________________________________________________________")


def check_kkt_conditions(
    point: npt.NDArray[np.float64],
    f: Callable[[npt.NDArray[np.float64]], float | np.float64],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    constraints: list[Callable[[npt.NDArray[np.float64]], float | np.float64]],
    d_constraints: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    lambdas: list[float] | npt.NDArray[np.float64],
):
    tol = 1e-2
    print("KKT conditions")
    print(f"Point: {np.round(point, 3)}")
    print(f"Value at point: {f(point)}")
    print("Dual feasibility: Lambdas should be >= 0")
    print("Values of lambdas are:")
    satisfy_dual = True
    for i in range(len(constraints)):
        print(round(lambdas[i], 3), end=" ")
        if lambdas[i] < -tol:
            satisfy_dual = False

    print(f"\nDo the lambdas satisfy Dual feasibility: {satisfy_dual}")
    print("Primal feasibility: Constraints should be <= 0")
    print("Values of constraints are:")
    satisfy_primal = True
    for i in range(len(constraints)):
        print(np.round(constraints[i](point), 3), end=" ")
        if constraints[i](point) > tol:
            satisfy_primal = False

    print(f"\nDoes the point satisfy primal feasibility: {satisfy_primal}")
    print("Complimentary slackness: Lambda_i * constraint_i should be = 0")
    print("Values of lambda_i * constraint_i are:")
    satisfy_slackness = True
    for i in range(len(constraints)):
        print(np.round(constraints[i](point) * lambdas[i], 3), end=" ")
        if np.abs(constraints[i](point) * lambdas[i]) > tol:
            satisfy_slackness = False
    print(f"\nIs complementary slackness satisfied: {satisfy_slackness}")

    print(f"Stationary condition: derivative of Lagrangian w.r.t. x should be 0")
    sum = d_f(point)
    for i in range(len(d_constraints)):
        sum += lambdas[i] * d_constraints[i](point)
    satisfy_stationary = np.all(np.abs(sum) < tol)
    print(f"Value of derivative of Lagrangian w.r.t. x is: {np.round(sum, 3)}")
    print(f"Is stationary condition satisfied?: {satisfy_stationary}")
    print("____________________________________________________________________")


def main():
    with open("output.txt", "w") as sys.stdout:
        for test_case_num, test_case in enumerate(PGD_test_cases):
            print(f"PGD test case {test_case_num}")
            try:
                ans = projected_gd(
                    test_case[0], test_case[1], test_case[2], test_case[3], test_case[4]
                )
                if not isinstance(ans, np.ndarray):
                    print("Wrong return type")
                else:
                    check_constraints_pgd(ans, test_case[0], test_case[1], test_case[4])
            except Exception as e:
                print(f"PGD test case {test_case_num}: {e}")
            print(flush=True)
        for test_case_num, test_case in enumerate(DA_test_cases):
            print(f"Dual ascent test case {test_case_num}")
            try:
                ans = dual_ascent(
                    test_case[0], test_case[1], test_case[2], test_case[3], test_case[4]
                )
                if not isinstance(ans, tuple):
                    print("Wrong return type")
                else:
                    check_kkt_conditions(
                        ans[0], test_case[0], test_case[1], test_case[2], test_case[3], ans[1]
                    )
            except Exception as e:
                print(f"Dual ascent test case {test_case_num}: {e}")
            print(flush=True)


if __name__ == "__main__":
    main()
