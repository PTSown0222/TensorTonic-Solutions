def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x0 = 0
    for _ in range(steps):
        grad_fx = 2 * a * x0 + b
        x0 = x0 - lr * grad_fx
    return x0