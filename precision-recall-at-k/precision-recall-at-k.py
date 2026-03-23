def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top = recommended[:k]
    data = set(top) & set(relevant)

    res = []
    res.append(len(data)/k)
    res.append(len(data)/len(relevant))

    return res

    