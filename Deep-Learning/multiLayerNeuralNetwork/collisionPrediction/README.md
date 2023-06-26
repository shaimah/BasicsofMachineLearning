# Overview
### The purpose of this project is to help a small robot navigate a simulated environment without any collision by training a neural network. 

### The simulation design lies outside the scope of this project. The sole purpose of the simulation is to: 1. collect the training data to be used as input for the neural network 2. observe the performance of the robot after training the network. 

### We shall leverage the power of PyTorch to perform forward and backward propagation and effectively train our network. However, this shields the mathematics behind the training process and takes away from your learning value. If you want to really understand how a network is trained, please refer to the [digit classification project](https://github.com/shaimah/Basics-of-Machine-Learning/tree/main/Deep-Learning/multiLayerNeuralNetwork/digitClassification), where everything is implemented from scratch.

## Project Instructions
- *Environment setup:* It is recommended to use Anaconda prompt to complete this project to avoid any dependency issues. The following libraries should be installed (the exact version is not a must but this is what worked for me)

    *Python  3.7*

    *cython 0.29.32*

    *matplotlib 3.5.3*

    *scikit-learn 0.21.1*

    *scipy 1.7.3*

    *pymunk 5.7*

    *pillow 9.2.0*

    *pygame 2.1.2*

    *numpy 1.21.5*

    *noise 1.2.2*

- *Collecting train data:* run the `collect_data.py` to start the simulated environment and collect the training input data for every action taken by the robot. Every sample will contain 5 distance readings and the action taking - this is considered as x or the input data. Whether a collison or no collision occurs is the label or y, which will have a binary representation. However, this step is not mandatory to successfully complete the project. The file is hard-coded to produce 100 sample, which is extremely insufficient for training. You can tweak the code to collect as many samples as you please or simply use the training_data `.csv file` in the `saved\` folder to save time.  

- Now that you have collected your training data, you can package it into an iterable PyTorch DataLoader for ease of use. Make sure to create both a
training and testing DataLoader. For this step, you need to run `Data_Loaders.py`. 

- Use the [torch.nn](https://pytorch.org/docs/stable/nn.html) to design your own custom network. Regardless, you must start with 6 neurons and end with 1 neuron. The number of hidden layers, activation function and optimizer(for backpropagation is upto you). Run or edit `Networks.py` followed by `train_model.py` to start the training.

- Evaluate your model by running `goal_seeking.py`. You can visually observe how the robot behaves when it encounters an obstacle. There is also a counter that keeps track of the number of collisions that occur until the robot reaches 10 goals.