import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import numpy as np
from torchvision.ops import misc as misc_nn_ops

# Regular resnet50, pretrained on ImageNet, without the classifier and the average pooling layer
resnet50_1 = torch.nn.Sequential(*(list(torchvision.models.resnet50(pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d).children())[:-2]))
resnet50_1.eval()
# Resnet50, extract from the Faster R-CNN, also pre-trained on ImageNet
resnet50_2 = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True).backbone.body
resnet50_2.eval()
# Loading a random image, converted to torch.Tensor, rescalled to [0, 1] (not that it matters)
image = torch.ones((1, 3, 224, 224))
# Obtaining the model outputs
with torch.no_grad():
    # Output from the regular resnet50
    output_1 = resnet50_1(image)
    # Output from the resnet50 extracted from the Faster R-CNN
    output_2 = resnet50_2(image)["3"]
    # Their outputs aren't the same, which I would assume they should be
    np.testing.assert_almost_equal(output_1.numpy(), output_2.numpy())