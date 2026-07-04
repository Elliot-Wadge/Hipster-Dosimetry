from numba import njit
import numpy as np
from .calibration import objective_derivative, objective_second_derivative


@njit
def newton_raphson(x0, ODs, a, b, c, tol, bound, max_iter=100):
    '''newton raphson plus bisection method to better gaurantee convergence'''
    score = objective_derivative(x0, ODs, a, b, c)
    x = x0
    bracket = bound
    iterations = 0 
    if objective_derivative(bound[0], ODs, a, b, c) * objective_derivative(bound[1], ODs, a, b, c) < 0:
        bracketed = True
    else:
        bracketed = False

    while abs(score) > tol and iterations < max_iter:

        dfx = objective_second_derivative(x, ODs, a, b, c)
        # if the derivative is too small don't use newton method,
        if dfx < 1e-15:
            # if we have a sign change already the bisect
            if bracketed:
                new_x = 0.5 * (bracket[0] + bracket[1])
            # if no sign change yet then start with a random
            else:
                new_x = bracket[0] + np.random.random() * (bracket[1] - bracket[0])
        else:
            new_x = x - score / dfx
        
        new_score = objective_derivative(new_x, ODs, a, b, c)
        # if the new_x is outside of the bounds and we have a sign change then use bisection
        if (new_x < bracket[0] or new_x > bracket[1]) and bracketed:
            new_x =  (bracket[0] + bracket[1])/2
        # if the new_x is outside of the bound and we don't have a sign change then select random x in range
        elif not bracketed and (new_x < bracket[0] or new_x > bracket[1]):
            new_x = bracket[0] + np.random.random()*(bracket[1] - bracket[0])
        # in case new_x has changed update the score
        new_score = objective_derivative(new_x, ODs, a, b, c)

        # if we have found a sign change, update our bound and set flag
        if not bracketed and new_score * score < 0:
            if x < new_x:
                bracket[0] = x
                bracket[1] = new_x
            else:
                bracket[0] = new_x
                bracket[1] = x
            bracketed = True
        # update the bounds based on new_x
        if bracketed:
            if objective_derivative(bracket[0], ODs, a, b, c) * new_score < 0:
                bracket[1] = new_x
            else:
                bracket[0] = new_x

        x = new_x
        score = new_score
        iterations += 1

    return x