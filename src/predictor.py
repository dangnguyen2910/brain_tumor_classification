import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score 

class Predictor: 
    def __init__(self, model):
        self.__model = model.eval()

    def predict(self, img): 
        output = self.__model(img.unsqueeze(0))
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        output = torch.argmax(output)
        return output

    def eval(self, dataset): 
        outputs = []
        labels = []
        for i in range(len(dataset)): 
            img = dataset[i][0]
            label = dataset[i][1]
            output = self.predict(img)
            outputs.append(output)
            labels.append(label)

        precision = precision_score(y_true = labels, y_pred = outputs)
        recall = recall_score(y_true = labels, y_pred = outputs)
        f1 = f1_score(y_true = labels, y_pred = outputs)
        return precision, recall, f1
            


        
        


