def get_accuracy(preds, gts):
    assert len(preds) == len(gts)
    num_total = len(preds)
    num_correct = sum([1 if y_pred == y else 0 for y_pred, y in zip(preds, gts)])

    acc = num_correct / num_total
    err = 1 - acc

    return acc, err
