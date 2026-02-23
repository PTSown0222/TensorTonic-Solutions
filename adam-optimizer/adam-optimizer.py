import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    bias_correction1 = 1 - beta1 ** t
    bias_correction2 = 1 - beta2 ** t
    
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    
    param_new = param - (lr * m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m, v