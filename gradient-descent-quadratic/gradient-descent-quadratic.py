def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for i in range(steps):
        
        x_pred = 2*a*x0 + b
        x_true = -b/(2*a)
            
        if (x_pred != 0):
            x1 = x0 - lr*x_pred
        else: 
            break

        x0 = x1

    return x1