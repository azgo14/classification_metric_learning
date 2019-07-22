import pretrainedmodels
import torch.nn as nn


class EmbeddedFeatureWrapper(nn.Module):
    """
    Wraps a base model with embedding layer modifications.
    """
    def __init__(self,
                 feature,
                 input_dim,
                 output_dim):
        super(EmbeddedFeatureWrapper, self).__init__()

        self.feature = feature
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.remap = None
        if input_dim != output_dim:
            self.remap = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, images):
        x = self.feature(images)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.standardize(x)

        if self.remap:
            x = self.remap(x)

        x = nn.functional.normalize(x, dim=1)

        return x

    def __str__(self):
        return "{}_{}".format(self.feature.name, str(self.embed))


def resnet50(output_dim):
    """
    resnet50 variant with `output_dim` embedding output size.
    """
    basemodel = pretrainedmodels.__dict__["resnet50"](num_classes=1000)

    model = nn.Sequential(
        basemodel.conv1,
        basemodel.bn1,
        basemodel.relu,
        basemodel.maxpool,

        basemodel.layer1,
        basemodel.layer2,
        basemodel.layer3,
        basemodel.layer4
    )
    model.name = "resnet50"
    featurizer = EmbeddedFeatureWrapper(feature=model, input_dim=2048, output_dim=output_dim)
    featurizer.input_space = basemodel.input_space
    featurizer.input_range = basemodel.input_range
    featurizer.input_size = basemodel.input_size
    featurizer.std = basemodel.std
    featurizer.mean = basemodel.mean

    return featurizer
