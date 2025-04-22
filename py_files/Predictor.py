import trace
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback

class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_widths: list[int], num_classes: int):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        previous_width = hidden_widths[0]
        # create list of layers
        self.layers = [nn.Linear(input_size, previous_width)]
        for width in hidden_widths[1:-1]: 
            self.layers.append(nn.Linear(previous_width, width))
            previous_width = width
        self.layers.append(nn.Linear(previous_width, num_classes))
        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.ReLU()
        self.layer_activations = [] # the activations from each layer following relu

    def forward(self, x):
        self.layer_activations.clear()
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            self.layer_activations.append(x.tolist())
        x = self.layers[-1](x)
        self.layer_activations.append(x.tolist())
        # no softmax at the end
        return x
    
    def get_activations(self) -> list:
        try:
            reconstructed_list = []
            pointer = 0
            widths = [len(layer) for layer in self.layer_activations] # for list reconstruction
            print(f'\nWidths: {widths}\n')
            flattened_data = np.array([x for layer in self.layer_activations for x in layer])
            norm_data = (flattened_data-np.min(flattened_data))/(np.max(flattened_data)-np.min(flattened_data))
            norm_data_list = norm_data.tolist()
            
            print("Reconstructed List:")
            for width in widths:
                slice = norm_data_list[pointer:pointer+width]
                print(slice)
                reconstructed_list.append(slice)
                pointer += width
            
            return reconstructed_list
            
        except Exception as e:
            print(e)
            traceback.print_exc()

    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.layer_activations = [] # the activations from each layer following relu

    def forward(self, x):
        self.layer_activations.clear()
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        self.layer_activations.append(x.tolist())
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        self.layer_activations.append(x.tolist())
        return x
    
    def get_activations(self) -> list:
        data = np.array(self.layer_activations)
        normalized_data = (data-np.min(data))/(np.max(data)-np.min(data)) # ensure values are from 0-1
        return normalized_data.toList()
