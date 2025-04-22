""" 
Author: Liam Laidlaw
Filename: train_nets.py
Purpose: Trains and saves various types of torch based neural networks through an interactive CLI. 
"""
#Feed Forward Neural Network Trained and Tested on MNIST dataset
import os
from matplotlib.pylab import f
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pyinputplus as pyip
from Predictor import NeuralNet, CNN
from Model_Loader import Loader
from tqdm import tqdm

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using GPU") if torch.cuda.is_available() else print('Using CPU')

def train_model(model: NeuralNet, num_epochs: int, train_loader, criterion, optimizer, is_cnn: bool):
    keep_training = True
    accuracies = []
    losses = []
    data_length = len(train_loader.dataset)
    epochs = num_epochs

    while keep_training:
        bar = tqdm( # progress bar for training
            range(epochs), 
            desc="Training", 
            unit="epoch", 
            leave=False, 
            total=epochs,
            dynamic_ncols=True,
            colour="green",
            bar_format="{l_bar}{bar}{r_bar}{postfix}"
            )
        

        #training loop
        for epoch in range(epochs):
            correct_predictions = 0
            loss_accumulator = 0
            for i, (images, labels) in enumerate(train_loader):
                # (100, 1, 28, 28)
                # (100, 784)
                if not is_cnn:
                    images = images.reshape(-1, 28*28)
                images = images.to(device)
                labels = labels.to(device)

                #forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # accumulate acuracy and loss
                loss_accumulator += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                
                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            bar.update(1)
            epoch_accuracy = correct_predictions / data_length
            epoch_loss = loss_accumulator / data_length
            accuracies.append(epoch_accuracy) 
            losses.append(epoch_loss)
            
        print(f"\nFinal Loss: {epoch_loss}, Final Accuracy: {epoch_accuracy:.5f}")
        bar.close()

        keep_training = pyip.inputYesNo("Continue Training? (y/n):") == 'yes'
        if keep_training:
            epochs = pyip.inputInt("How many additional epochs should we train for? (1-1000): ", min=1, max=1000)

    return accuracies, losses

    
def test_model(model, test_loader, is_cnn: bool):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            if not is_cnn:
                images = images.reshape(-1, 28*28)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        return acc


def print_and_save(model, dimensions: list[int] = None, is_cnn: bool = False):
    if not is_cnn:
        dims = "-".join([str(x) for x in dimensions])
        filename = f"{dims}.pt"
    else:
        filename = "cnn.pt"
    save_path = os.path.join("models")
    print(f"Save Path: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    
    save = pyip.inputYesNo(prompt="Save Model? Note: This saves the full model object. (y/n): ", yesVal='y', noVal='n') == 'y'
    if save:
        torch.save(model, os.path.join(save_path, filename))
        print(f"Model saved to: {save_path}")


def load_save_all_state_dicts():
    load_dir = os.path.join("models")
    save_dir = os.path.join("model_dicts")
    os.makedirs(save_dir, exist_ok=True)
    model = None

    for filename in os.listdir(load_dir):
        # get file paths
        model_load_path = os.path.join(load_dir, filename)
        filename_no_ext = os.path.splitext(filename)[0]
        model_save_path = os.path.join(save_dir, f"{filename_no_ext}-dict.pt")

        # load model and save its dicts (uses cpu becuase it is unknown)
        model: nn.Module = torch.load(model_load_path, map_location=device, weights_only=False)
        model.eval()
        torch.save(model.state_dict(), model_save_path)


def load_data(batch_size: int):
    # Load MNIST
    try:
        # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        # test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        train_dataset = torchvision.datasets.EMNIST(root='./data', split='digits', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.EMNIST(root='./data', split='digits', train=False, transform=transforms.ToTensor(), download=True)
    except Exception as e:
        raise RuntimeError(f"Error loading MNIST dataset: {e}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def plot_graphs(accuracies: list[float], losses: list[float], num_epochs: int):

    fig, axis_1 = plt.subplots()
    axis_1.set_xlabel('Epochs')
    axis_1.set_ylabel('Accuracy')
    x_axis = np.arange(1, num_epochs+1)
    axis_1.plot(x_axis, accuracies, label='Accuracy', color='tab:blue')
    axis_1.tick_params(axis='y', labelcolor='tab:blue')

    axis_2 = axis_1.twinx()
    axis_2.set_ylabel('Loss')
    axis_2.plot(x_axis, losses, label='Loss', color='tab:orange')
    axis_2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Training Accuracy and Loss')
    plt.legend()
    plt.show()


def get_network_dimensions() -> list[int]:
    """Gets the dimensions of the network from the user.
    The user is prompted to enter the number of neurons in each layer separated by commas.
    Uses pyinputplus to validate the input.

    Returns:
        list[int]: List of integers representing the number of neurons in each layer.
    """
    dimensions = []
    num_layers = pyip.inputInt(prompt="How many layers should the network have? (1-10): ", min=1, max=10)

    for i in range(num_layers):
        layer_width = pyip.inputInt(prompt=f"\tHow many neurons in layer {i+1}? (1-1000): ", min=1, max=1000)
        dimensions.append(layer_width)

    return dimensions



def main():


    print("\n----------Welcome to the MNIST Neural Network Trainer----------\n")
    print("- This program trains a feed forward neural network on the MNIST dataset.")
    print("- Use the following prompts to train and save your models.\n")

    print(Loader("model_dicts", device=device, from_dicts=True).models)


    # hyperparameters
    input_size = 784  # 28x28
    num_classes = 10
    batch_size = 1000
    learning_rate = 0.001

    train_loader, test_loader = load_data(batch_size)

    while pyip.inputYesNo(prompt="Train a new model? (y/n): ", yesVal='y', noVal='n') == 'y':
        # get the type of NN to train
        train_cnn = pyip.inputMenu(["linear", "cnn"], "Which type of network do you want to train? (Enter the number of your choice):\n", numbered=True) == "cnn"
        if train_cnn:
            print(f"Initializing Model...")
            model = CNN().to(device)
            print("\nModel Initialized.")
        else: 
            hidden_widths = get_network_dimensions()
            print(f"\nTraining Parameters:\nInput Size: {input_size}\nNetwork Dimensions: {hidden_widths}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nEpochs: {num_epochs}\n")
            print(f"Initializing Model...")
            model = NeuralNet(input_size, hidden_widths, num_classes).to(device)
            print("\nModel Initialized.")

        num_epochs = pyip.inputInt(prompt="How many epochs should we train for? (1-1000): ", min=1, max=1000)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        print("Training Model...")
        accuracies, losses = train_model(model, num_epochs, train_loader, criterion, optimizer, train_cnn)
        
        # display training accuracy and loss as a graph
        # print("\nModel Trained.\n\nDisplaying Training Accuracy and Loss Graph...")
        # plot_graphs(accuracies, losses, num_epochs)
        print("\nTesting Model...")
        test_acc = test_model(model, test_loader, train_cnn)
        print(f'Accuracy of the network on the 10000 test images: {test_acc} %')
        if not train_cnn:
            print_and_save(model, dimensions=hidden_widths, is_cnn=train_cnn)
        else:
            print_and_save(model, is_cnn=train_cnn)

    save_full_models = pyip.inputYesNo("Save previous trained models? (y/n): ", yesVal='y', noVal='n') == 'y'
    if save_full_models:
        load_save_all_state_dicts()


if __name__ == "__main__":
    main()