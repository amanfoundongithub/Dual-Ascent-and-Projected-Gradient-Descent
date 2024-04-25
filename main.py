import os

from prettytable import PrettyTable
import numpy as np
import numpy.typing as npt

from algos import projected_gd, dual_ascent
from test_cases import PGD_test_cases, DA_test_cases


def main():
    PGD_table = PrettyTable()
    PGD_table.field_names = [
        "Test case",
        "Projected Gradient Descent",
    ]
    dividers = [4]
    for test_case_num, test_case in enumerate(PGD_test_cases):
        print(test_case_num)
        row = [
            test_case_num,
        ]
        try:
            ans = projected_gd(
                test_case[0], test_case[1], test_case[2], test_case[3], test_case[4]
            )
            if type(ans) != np.ndarray:
                print(
                    f"Wrong type of value returned projected gradient descent"
                )
                print(
                    f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                )
                row += [None]
            else:
                row += [np.round(ans, 3)]
        except Exception as e:
            print(f"Error in projected gradient descent")
            print(
                f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
            )
            print(e)
            row += [None]
        PGD_table.add_row(row, divider=test_case_num in dividers)

    DA_table = PrettyTable()
    DA_table.field_names = [
        "Test case",
        "Dual ascent",
    ]
    dividers = [4]

    for test_case_num, test_case in enumerate(DA_test_cases):
        print(test_case_num)
        row = [
            test_case_num,
        ]
        try:
            ans = dual_ascent(
                test_case[0], test_case[1], test_case[2], test_case[3], test_case[4]
            )
            if type(ans) != np.ndarray:
                print(
                    f"Wrong type of value returned in dual ascent"
                )
                print(
                    f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
                )
                row += [None]
            else:
                row += [np.round(ans, 3)]
        except Exception as e:
            print(f"Error in dual ascent")
            print(
                f"Test function was {test_case[0].__name__} with {test_case[2]} as starting point"
            )
            print(e)
            row += [None]
        DA_table.add_row(row, divider=test_case_num in dividers)

    print(PGD_table)
    print(DA_table)


if __name__ == "__main__":
    main()