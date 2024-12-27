import torch 
from torchvision.models import resnet18

class Model(torch.nn.Module):
    def __init__(self, n_class): 
        torch.nn.Module.__init__(self)
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=n_class)


    def forward(self, x): 
        return self.model(x)


# if __name__ == "__main__":
#     model = Model(n_class = 2)
#     input = torch.rand(1,3,512,512)
#     print(model(input))
