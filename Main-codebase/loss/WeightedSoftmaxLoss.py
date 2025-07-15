"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch
import torch.nn as nn


def create_loss():
    print("Loading Weighted Softmax Loss.")
    prob = [0.8997, 0.1003]
    prob = torch.FloatTensor(prob)
    # normalization
    max_prob = prob.max().item()
    prob = prob / max_prob
    # class reweight
    weight = -prob.log() + 1

    return nn.CrossEntropyLoss(weight=weight)
