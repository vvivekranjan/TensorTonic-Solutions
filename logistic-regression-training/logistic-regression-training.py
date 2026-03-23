import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _sigmoid_derivation(z):
    return _sigmoid(z) * (1 - _sigmoid(z))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    
    for i in range(steps):

        z = (X @ w) + b
        p = _sigmoid(z)

        loss = -1 * ((y*np.log(p) + (1 - y)*np.log(1 - p)).sum()).mean()

        dw = (X.T @ (p - y))/m

        
        w -= lr * dw
        b -= lr * (p - y).mean()

    return (w, b)
        