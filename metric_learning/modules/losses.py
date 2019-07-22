import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_instances,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss
