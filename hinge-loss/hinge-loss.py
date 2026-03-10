import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    loss = []
    for i in range(len(y_score)):
        loss.append(np.maximum(0, margin-(y_true[i]*y_score[i])))
    return np.mean(loss) if reduction == "mean" else np.sum(loss)