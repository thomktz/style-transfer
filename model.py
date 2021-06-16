# %%

import torchvision.models as models
import torch.nn as nn
# %%

class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()
        self.model = models.resnet152(pretrained = True)
        
    def forward(self, x):
        features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        features.append(x)
        x = self.model.layer2(x)
        features.append(x)
        x = self.model.layer3(x)
        features.append(x)
        x = self.model.layer4(x)
        features.append(x)
        return features
    
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.select = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features
# %%
