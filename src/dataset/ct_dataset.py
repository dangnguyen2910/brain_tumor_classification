from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from torchvision.transforms import v2
import os
import cv2

class CTDataset(Dataset): 
    def __init__(self, transform = None): 
        self.__datapath = '../data/Brain Tumor CT scan Images/'
        self.__dataset = self.__make_imagepath_label_list()
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__transform = transform


    def __getitem__(self, index): 
        img = cv2.imread(self.__dataset[0][index])
        # img = read_image(self.__dataset[0][index])
        img = torch.tensor(img).permute(2,0,1).float()
        label = self.__dataset[1][index]
        if self.__transform: 
            img = self.__transform(img).float()

        return img, label


    def __make_imagepath_label_list(self):
        image_path_list = []
        label_list = []

        healthy_folder_path = os.path.join(self.__datapath, "Healthy")
        tumor_folder_path = os.path.join(self.__datapath, "Tumor")

        image_path_list, label_list = self.__create_image_label_path_list(image_path_list, label_list, 
                                                                         healthy_folder_path, torch.tensor(0))
        image_path_list, label_list = self.__create_image_label_path_list(image_path_list, label_list, 
                                                                         tumor_folder_path, torch.tensor(1))
        return image_path_list, label_list


    def __create_image_label_path_list(self, image_path_list, label_list, folder, label_name):
        for image_name in os.listdir(folder):
            image_path_list.append(os.path.join(folder, image_name))
            label_list.append(label_name)

        return image_path_list, label_list


    def __len__(self): 
        return len(self.__dataset[0]) 
    

# if __name__ == "__main__":
#     dataset = CTDataset()
#     for i in range(len(dataset)):
#         if (dataset[i][0].size(0) == 1):
#             print(dataset[i][0].size())
