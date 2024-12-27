import torch 
from torchvision.models import resnet18

class Model: 
    def __init__(self, n_class): 
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=n_class)

    def forward(self, x): 
        return self.model(x)


