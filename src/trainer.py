import torch 
from torch.utils.data import random_split, DataLoader
import os

class Trainer: 
    def __init__(self, model, dataset): 
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__model = model.cuda()
        self.__train_dataset, self.__val_dataset = random_split(dataset, [0.9, 0.1])
        self.__train_dataloader = DataLoader(self.__train_dataset, batch_size=4, shuffle=True)
        self.__val_dataloader = DataLoader(self.__val_dataset, batch_size=4, shuffle=True)
        self.__loss_fn = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=0.0001)
        self.__epochs = 70
        self.train_loss_list = []
        self.val_loss_list = []


    def train(self):
        model_name = input("Enter model name(data_model_version.pth): ")
        best_vloss = 100
        train_losses = []
        val_losses = []

        for epoch in range(self.__epochs):
            print('-' * 50)
            print(f'Epoch [{epoch+1}/{self.__epochs}]: ')

            start = torch.cuda.Event(enable_timing = True)
            end = torch.cuda.Event(enable_timing = True)

            start.record()

            train_loss = self.__train_one_epoch()
            val_loss = self.__eval_one_epoch()

            end.record()
            torch.cuda.synchronize()

            print(f'  Train loss      : {train_loss:.3f}')
            print(f'  Validation loss : {val_loss:.3f}')
            print(f'  Train time/epoch: {start.elapsed_time(end) / 1000:.3f}')

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_vloss: 
                best_vloss = val_loss
                torch.save(self.__model.state_dict(), os.path.join('../pretrained_model/', model_name))

        return train_losses, val_losses


    
    def __train_one_epoch(self): 
        self.__model.train()
        train_running_loss = 1000

        for i, data in enumerate(self.__train_dataloader):
            input = data[0].cuda()
            label = data[1].cuda()

            self.__optimizer.zero_grad()
            output = self.__model(input)
            loss = self.__loss_fn(output, label)
            loss.backward()
            self.__optimizer.step()

            train_running_loss += loss.item()
            if i % 100 == 99: 
                print(f'  [{i+1}/{len(self.__train_dataloader)}]')

        return train_running_loss/len(self.__train_dataloader)


    def __eval_one_epoch(self):
        self.__model.eval()
        val_running_loss = 0 

        for i, data in enumerate(self.__val_dataloader):
            input = data[0].cuda()
            label = data[1].cuda()

            output = self.__model(input)
            loss = self.__loss_fn(output, label)

            val_running_loss += loss.item()

        return val_running_loss/len(self.__val_dataloader)


if __name__ == "__main__":
    Trainer().train()
