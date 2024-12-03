import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipetteDetector(nn.Module):
    def __init__(self):
        super(PipetteDetector, self).__init__()
        resnet101 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = torch.nn.Sequential(*(list(resnet101.children())[:-2]))

        # all parameters trainiable!
                    
        # global average pooling 2d
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.xyzw_out = nn.LazyLinear(4)

    def forward(self, input):
        resnet_output = self.resnet(input)

        pooled = self.pooling(resnet_output)

        flat = pooled.reshape(pooled.size(0), -1)

        xyzw = self.xyzw_out(flat)
        return xyzw



# decide how to transform data
class Normalizer:
    """Used to normalize and denormalize position+snr data.
    
    Input data should be a 1D numpy array with 4 elements: (z, row, col, snr)
    (z, row, col_) values are scaled and shifted to be in the range [-1, 1]
    snr is log-transformed
    """
    def __init__(self, pos_min, pos_max):
        range = np.array([pos_min+[-1], pos_max+[1]])
        diff = range[1] - range[0]
        self.scale = 2 / diff
        self.offset = range[0] + diff / 2

    def normalize(self, x):
        xn = (x - self.offset) * self.scale
        xn[:, 3] = np.log10(xn[:, 3] + 1)
        assert np.all(np.isfinite(xn))
        return xn

    def denormalize(self, x):
        x = (x / self.scale) + self.offset
        x[:, 3] = 10**(x[:, 3]) - 1
        return x

pos_min = [-100, 0, 0]
pos_max = [100, 500, 500]
pos_normalizer = Normalizer(pos_min, pos_max)

def make_image_tensor(data):
    """Convert ubyte image data to a 0.0-1.0 float32 torch tensor on the GPU."""
    normalized = (data / 255).transpose((0, 3, 1, 2)).astype(np.float32)
    return torch.tensor(normalized).to(device)

def make_position_tensor(data):
    """Convert position data to a normalized (-0.5 to 0.5) float32 torch tensor on the GPU."""
    return torch.tensor(pos_normalizer.normalize(data).astype(np.float32)).to(device)
