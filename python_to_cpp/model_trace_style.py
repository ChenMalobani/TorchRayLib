import torch
import torchvision
import re
import torch
import os
import random

# https://github.com/gnsmrky/pytorch-fast-neural-style-for-web
from .transformer_net import *
style_model = TransformerNet()
# m="candy.pth"
state_dict = torch.load("./resources/udnie.pth")
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
style_model.load_state_dict(state_dict)
style_model.eval()

# model = torchvision.models.resnet50(pretrained=True)
# model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(style_model, example)
traced_script_module.save("./resources/udnie_cpp.pt")
