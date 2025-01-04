from dataset.ct_dataset import CTDataset
from predictor import Predictor
from torch.utils.data import random_split
from torchvision.transforms import v2
from model import Model
from trainer import Trainer
import torch
import os

class Main: 
    def __init__(self): 
        self.mri_path = '../data/Brain Tumor CT scan Images/'
        ct_path = '../data/Brain Tumor CT scan Images/'
        self.pretrained_model_path = '../pretrained_model'

    
    def run(self): 
        choose = input("mri/ct? ")
        match choose: 
            case "mri":
                self.__mri()
            case "ct":
                self.__ct()
            case _:     
                print("Huh")


    def __mri(self):
        dataset = MRIDataset()
        train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])

        model = Model(n_class = 5)

        want_train = input("Do you want to train (y/n)")

        if want_train == 'y': 
            Trainer(model, train_dataset).train()
            predictor = Predictor(model, test_dataset)
            return

        model_name_list = os.listdir(pretrained_model_path)
        for i, model_path in enumerate(model_name_list):
            print(model_path)

        choosed_model = int(input("Choose model: "))
        model.load_state_dict(torch.load(model, weights_only = True))
        predictor = Predictor(model, test_dataset)
        return 


    def __ct(self):
        transform = v2.Compose([
            v2.Resize((256, 256))
        ])
        dataset = CTDataset(transform = transform)
        train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])

        model = Model(n_class = 2)

        want_train = input("Do you want to train (y/n): ")

        if want_train == 'y': 
            Trainer(model, train_dataset).train()
            predictor = Predictor(model)
            precision, recall, f1 = predictor.eval(test_dataset)
            print(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}")
            return

        model_name_list = os.listdir(self.pretrained_model_path)
        for i, model_path in enumerate(model_name_list):
            print(f'{i}.{model_path}')

        choosed_model = int(input("Choose model: "))
        choosed_model = os.path.join(self.pretrained_model_path, model_name_list[choosed_model])
        model.load_state_dict(torch.load(choosed_model, weights_only = True))
        predictor = Predictor(model)
        precision, recall, f1 = predictor.eval(test_dataset)
        print(f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}")
        return 


if __name__ == '__main__':
    # torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    Main().run()
