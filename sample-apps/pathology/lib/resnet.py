from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck


class CRF(nn.Module):
    def __init__(self, num_nodes, iteration=10):
        """Initialize the CRF module
        Args:
            num_nodes: int, number of nodes/patches within the fully CRF
            iteration: int, number of mean field iterations, e.g. 10
        """
        super(CRF, self).__init__()
        self.num_nodes = num_nodes
        self.iteration = iteration
        self.W = nn.Parameter(torch.zeros(1, num_nodes, num_nodes))

    def forward(self, feats, logits):
        """Performing the CRF. Algorithm details is explained below:
        Within the paper, I formulate the CRF distribution using negative
        energy and cost, e.g. cosine distance, to derive pairwise potentials
        following the convention in energy based models. But for implementation
        simplicity, I use reward, e.g. cosine similarity to derive pairwise
        potentials. So now, pairwise potentials would encourage high reward for
        assigning (y_i, y_j) with the same label if (x_i, x_j) are similar, as
        measured by cosine similarity, pairwise_sim. For
        pairwise_potential_E = torch.sum(
            probs * pairwise_potential - (1 - probs) * pairwise_potential,
            dim=2, keepdim=True
        )
        This is taking the expectation of pairwise potentials using the current
        marginal distribution of each patch being tumor, i.e. probs. There are
        four cases to consider when taking the expectation between (i, j):
        1. i=T,j=T; 2. i=N,j=T; 3. i=T,j=N; 4. i=N,j=N
        probs is the marginal distribution of each i being tumor, therefore
        logits > 0 means tumor and logits < 0 means normal. Given this, the
        full expectation equation should be:
        [probs * +pairwise_potential] + [(1 - probs) * +pairwise_potential] +
                    case 1                            case 2
        [probs * -pairwise_potential] + [(1 - probs) * -pairwise_potential]
                    case 3                            case 4
        positive sign rewards logits to be more tumor and negative sign rewards
        logits to be more normal. But because of label compatibility, i.e. the
        indicator function within equation 3 in the paper, case 2 and case 3
        are dropped, which ends up being:
        probs * pairwise_potential - (1 - probs) * pairwise_potential
        In high level speaking, if (i, j) embedding are different, then
        pairwise_potential, as computed as cosine similarity, would approach 0,
        which then as no affect anyway. if (i, j) embedding are similar, then
        pairwise_potential would be a positive reward. In this case,
        if probs -> 1, then pairwise_potential promotes tumor probability;
        if probs -> 0, then -pairwise_potential promotes normal probability.
        Args:
            feats: 3D tensor with the shape of
            [batch_size, num_nodes, embedding_size], where num_nodes is the
            number of patches within a grid, e.g. 9 for a 3x3 grid;
            embedding_size is the size of extracted feature representation for
            each patch from ResNet, e.g. 512
            logits: 3D tensor with shape of [batch_size, num_nodes, 1], the
            logit of each patch within the grid being tumor before CRF
        Returns:
            logits: 3D tensor with shape of [batch_size, num_nodes, 1], the
            logit of each patch within the grid being tumor after CRF
        """
        feats_norm = torch.norm(feats, p=2, dim=2, keepdim=True)
        pairwise_norm = torch.bmm(feats_norm, torch.transpose(feats_norm, 1, 2))
        pairwise_dot = torch.bmm(feats, torch.transpose(feats, 1, 2))

        # cosine similarity between feats
        pairwise_sim = pairwise_dot / pairwise_norm

        # symmetric constraint for CRF weights
        W_sym = (self.W + torch.transpose(self.W, 1, 2)) / 2
        pairwise_potential = pairwise_sim * W_sym
        unary_potential = logits.clone()

        for i in range(self.iteration):
            # current Q after normalizing the logits
            probs = torch.transpose(logits.sigmoid(), 1, 2)
            # taking expectation of pairwise_potential using current Q
            pairwise_potential_E = torch.sum(
                probs * pairwise_potential - (1 - probs) * pairwise_potential, dim=2, keepdim=True
            )
            logits = unary_potential + pairwise_potential_E

        return logits

    def __repr__(self):
        return "CRF(num_nodes={}, iteration={})".format(self.num_nodes, self.iteration)


class ResNetWithCF(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_nodes=1,
    ):
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        self.crf = CRF(num_nodes) if num_nodes else None

    def forward(self, x):
        batch_size, grid_size, _, crop_size = x.shape[0:4]
        # flatten grid_size dimension and combine it into batch dimension
        x = x.view(-1, 3, crop_size, crop_size)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        f = x.view(x.size(0), -1)  # feats/features, i.e. patch embeddings from ResNet
        logits = self.fc(f)

        # restore grid_size dimension for CRF
        f = f.view((batch_size, grid_size, -1))
        logits = logits.view((batch_size, grid_size, -1))

        if self.crf:
            logits = self.crf(f, logits)
        return torch.squeeze(logits, dim=2)  # exclude batch dim


def resnet18_cf(**kwargs):
    return ResNetWithCF(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_cf(**kwargs):
    return ResNetWithCF(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_cf(**kwargs):
    return ResNetWithCF(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_cf(**kwargs):
    return ResNetWithCF(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_cf(**kwargs):
    return ResNetWithCF(Bottleneck, [3, 8, 36, 3], **kwargs)
