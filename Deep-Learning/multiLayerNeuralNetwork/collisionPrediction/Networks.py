import torch
import torch.nn as nn
import numpy as np


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()

        self.input_to_hidden = nn.Linear(6, 900)
        self.hidden_to_hidden1 = nn.Linear(900, 1000)

        self.hidden1_to_hidden2 = nn.Linear(1000, 1100)
        self.hidden2_to_hidden3 = nn.Linear(1100, 1200)


        self.nonlinear_activation = nn.ReLU() 


        self.hidden_to_output = nn.Linear(1200, 1)
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture

        

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.hidden_to_hidden1(hidden)
        hidden = self.nonlinear_activation(hidden)

        hidden = self.hidden1_to_hidden2(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.hidden2_to_hidden3(hidden)
        hidden = self.nonlinear_activation(hidden)

        output = self.hidden_to_output(hidden)

        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        losses = []
        for idx,sample in enumerate(test_loader):
            model_output =  model(sample['input'])
            
            target_output = sample['label'].unsqueeze(1)

            loss = loss_function(model_output, target_output).item()
        	
            losses.append(loss)
        loss = sum(losses)/len(losses)
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
	
# output and the desired output.

        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
    
