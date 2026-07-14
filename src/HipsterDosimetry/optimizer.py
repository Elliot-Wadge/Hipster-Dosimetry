from numba import njit
import numpy as np
from .calibration import objective_derivative, objective_second_derivative, objective


@njit(cache=True)
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

    converged = True
    while abs(score) > tol:
        if iterations >= max_iter:
            converged = False
            break
        dfx = objective_second_derivative(x, ODs, a, b, c)
        # if the derivative is too small don't use newton method,
        if dfx < 1e-15:
            # if we have a sign change already the bisect
            if bracketed:
                new_x = 0.5 * (bracket[0] + bracket[1])
            # if no sign change yet then start with a random new x in the bound
            else:
                new_x = bracket[0] + np.random.random() * (bracket[1] - bracket[0])
        else:
            new_x = x - score / dfx
        
        # if the new_x is outside of the bounds and we have a sign change then use bisection
        if (new_x < bracket[0] or new_x > bracket[1]) and bracketed:
            new_x =  (bracket[0] + bracket[1])/2
        # if the new_x is outside of the bound and we don't have a sign change then select random x in range
        elif not bracketed and (new_x < bracket[0] or new_x > bracket[1]):
            new_x = bracket[0] + np.random.random()*(bracket[1] - bracket[0])
        # update the score
        new_score = objective_derivative(new_x, ODs, a, b, c)


        # update the bounds based on new_x
        if bracketed:
            if objective_derivative(bracket[0], ODs, a, b, c) * new_score < 0:
                bracket[1] = new_x
            else:
                bracket[0] = new_x
        # if we have found a sign change for the first time, update our bound and set flag
        elif not bracketed and new_score * score < 0:
            if x < new_x:
                bracket[0] = x
                bracket[1] = new_x
            else:
                bracket[0] = new_x
                bracket[1] = x
            bracketed = True
        

        x = new_x
        score = new_score
        iterations += 1

    if not converged:
        score0 = objective(ODs*bound[0], a, b, c)
        score1 = objective(ODs*bound[1], a, b, c)
        if score0 < score1:
            x = bound[0]
        else:
            x = bound[1]
    return x