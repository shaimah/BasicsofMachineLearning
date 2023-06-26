from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
import torch.optim as optim


import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 12
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.L1Loss()
    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    print(losses)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    losses = []
    for epoch_i in range(no_epochs):
        model.train()
        running_loss = []
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            inputs, labels = sample['input'],sample['label'].unsqueeze(1)

        # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            

        # print statistics
            
            #if epoch_i % 10 == 0:    # print every 2000 mini-batches
                #print(f'Epoch_i {epoch_i}: Loss {loss.item()}')
            optimizer.zero_grad()
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)   
    print(f'after training : {min_loss}')
    torch.save(model.state_dict(), "saved/saved_model.pkl", _use_new_zipfile_serialization=False)        



if __name__ == '__main__':
    no_epochs = 20
    train_model(no_epochs)
