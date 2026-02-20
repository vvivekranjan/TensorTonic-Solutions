import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    num = np.asarray(x)
    relu = np.maximum(0, num)
    return relu
    pass