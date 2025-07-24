import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


def accuracy(output, target, return_length=False):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)


def classwise_accuracy(output, target, num_classes):
    target = target.detach().cpu().numpy()
    output = output.argmax(dim=1).detach().cpu().numpy()
    cf = confusion_matrix(target, output, labels=list(range(num_classes))).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    out_cls_acc = "Class Accuracy: %s" % (
        (
            np.array2string(
                cls_acc,
                separator=",",
                formatter={"float_kind": lambda x: "%.3f" % x},
            )
        ),
    )
    return cls_acc


def macro_f1(output, target, num_classes):
    output = output.argmax(dim=1).detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return f1_score(output, target, average="macro", labels=list(range(num_classes)))


def top_k_acc(output, target, k=5, return_length=False):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)
