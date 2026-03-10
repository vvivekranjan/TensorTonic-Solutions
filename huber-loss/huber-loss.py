import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """

    e = [np.abs(y_true[i] - y_pred[i]) for i in range(len(y_pred))]
    for i in range(len(e)):
        if e[i] <= delta:
            e[i] = np.square(e[i])/2
        else:
            e[i] = delta * (e[i] - delta/2)
    return np.mean(e)