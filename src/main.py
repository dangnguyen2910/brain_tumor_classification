from dataset.mri_dataset import mri_dataset
import os

class Main: 
    def __init__(self): 
        mri_path = '../data/Brain Tumor CT scan Images/'
        ct_path = '../data/Brain Tumor CT scan Images/'
        pretrained_model_path = '../pretrained_model'

    
    def run(self): 
        choose = input("mri/ct? ")
        match choose: 
            case "mri":
                __mri()
            case "ct":
                __ct()
            case _:     
                print("Huh")


    def __mri(self):
        dataset = MRIDataset()
        model = Model(n_class = 5)

        want_train = input("Do you want to train (y/n)")

        if want_train == 'y': 
            model = Trainer(model, train_dataset).train()
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
        dataset = CTDataset()
        model = Model(n_class = 2)

        want_train = input("Do you want to train (y/n)")

        if want_train == 'y': 
            model = Trainer(model, train_dataset).train()
            predictor = Predictor(model, test_dataset)
            return

        model_name_list = os.listdir(pretrained_model_path)
        for i, model_path in enumerate(model_name_list):
            print(model_path)

        choosed_model = int(input("Choose model: "))
        model.load_state_dict(torch.load(model, weights_only = True))
        predictor = Predictor(model, test_dataset)
        return 


if __name__ == '__main__':
    Main().run()
