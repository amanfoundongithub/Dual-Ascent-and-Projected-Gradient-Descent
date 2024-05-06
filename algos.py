from typing import Callable, Literal

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np



# --------------------- PROJECTION FORMULAS ----------------------------
def linear_projection(constraints, point):

    lower_bound = constraints[0]
    upper_bound = constraints[1] 

    point    = np.maximum(lower_bound, point) 
    point    = np.minimum(point, upper_bound)
    
    return point 

def ball_projection(constraints, point):

    center = constraints[0]
    radius = constraints[1]

    point  = point - center

    factor = radius/max(radius, np.linalg.norm(point))  

    point  = factor * point + center
    
    return point


def Pc(constraints, point, constraint_type):
    if constraint_type == 'linear':
        return linear_projection(constraints, point)
    elif constraint_type == 'l_2':
        return ball_projection(constraints, point) 

# ---------------- PROJECTED GRADIENT ----------------
def G(x, d_f, constraint_type, constraints, m):
    return m * (x - Pc(constraints, 
                       point = (x - (1/m) * d_f(x)),
                       constraint_type = constraint_type))

# ---------------- STEP SIZE FORMULA -----------------
def get_projected_step_size(initial_point: npt.NDArray[np.float64],
    f: Callable[[npt.NDArray[np.float64]], np.float64 ],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    constraint_type,
    constraints):
    
    x = initial_point

    beta = 0.05

    tk = 0.75

    while True: 
        f_diff = f(x) - f(Pc(constraints, x - tk * d_f(x), constraint_type))

        proj_gradient = tk * np.linalg.norm(G(x, d_f, constraint_type, constraints,
                                              m = 1/tk))**2
    
    
        if abs(f_diff - proj_gradient) < 1e-3:
            return tk 
        else:
            tk = tk * beta


# ---------------- PROJECTED GRADIENT DESCENT ----------
def projected_gd(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    constraint_type: Literal["linear", "l_2"],
    constraints: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64]
) -> npt.NDArray[np.float64]:
    tolerance = 1e-10
    
    x = point 

    k = 0
    while k < 1e3:
        # Descent direction
        dk = - d_f(x) 

        # Step size
        tk = get_projected_step_size(initial_point = x, 
                               f = f, d_f = d_f,
                               constraint_type = constraint_type,
                               constraints = constraints) 

        x_new = Pc(constraints, x + tk * dk, constraint_type)

        if np.linalg.norm(x_new - x) <= tolerance:
            return x_new
        else: 
            k += 1 
            x = x_new
        
     
    return x

def dual_ascent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    c: list[Callable[[npt.NDArray[np.float64]], np.float64 | float]],
    d_c: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    initial_point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    
    pass

