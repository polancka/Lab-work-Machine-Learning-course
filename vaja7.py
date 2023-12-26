import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
import math
import torch.nn.functional as f
import torch.optim as optim 

#Disclaimer: Some parts of the code are commented-out (one time plot making etc.)

#1.DATASET
# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Normalize the training data - fix the normalisation - Q normalize just training set and save train_mean and train_sd
train_samples = train_dataset.data.float() / 255.0
train_labels = train_dataset.targets

classes = train_dataset.classes
#visualize random members from each class - visualized in picture vaja_random_members
# fig, axs = plt.subplots(1, len(classes), figsize=(15, 2))

# for i in range(len(classes)):
#     class_indices = torch.where(train_labels == i)[0]
#     random_index = class_indices[np.random.randint(len(class_indices))]
#     image = train_samples[random_index].numpy()
#     axs[i].imshow(image, cmap='gray')
#     axs[i].axis('off')
#     axs[i].set_title(classes[i])

# plt.show()

# Split the training data into training and validation subsets
validation_size = 0.16 # Q = should this be 0.2 or 0.16? 0.2 of the whole dataset but 0.16 of training for the same size
num_validation = int(validation_size * len(train_samples))
indices = torch.randperm(len(train_samples))

train_indices = indices[num_validation:]
validation_indices = indices[:num_validation]

train_x = train_samples[train_indices]
train_y= train_labels[train_indices]

validation_x = train_samples[validation_indices]
validation_y = train_labels[validation_indices]

# Convert samples and labels to PyTorch tensors
train_x_tensor = train_x.unsqueeze(1)  # Add channel dimension
train_y = torch.nn.functional.one_hot(train_y, num_classes=len(classes))

validation_x_tensor = validation_x.unsqueeze(1)  # Add channel dimension
validation_y = torch.nn.functional.one_hot(validation_y, num_classes=len(classes))

# Dataset class from the lecture
class Dataset: 
    def __init__(self, x,y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.length = int(np.ceil(x.shape[0]/batch_size))
        self.indices = np.arange(x.shape[0]) # 1,2,3,4,5,6...,n-1 we will shuffle this not the whole dataset to retain the original ordering
        
    def __getitem__(self, i):
        #find lowest and highest value
        i0 = i*self.batch_size 
        i1 = min((i+1)*self.batch_size, self.x.shape[0])
        index = self.indices[i0:i1]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length

    def shuffle(self):
        self.indices = np.random.permutation(self.indices)

# Create training and validation datasets using the Dataset class
batch_size = 16 #Q = Should the batch size here be bigger?
train_dataset = Dataset(train_x_tensor, train_y, batch_size)
validation_dataset = Dataset(validation_x_tensor, validation_y, batch_size)

# Visualize 16 samples from a batch belonging to the training dataset - vaja7_batch_samples.png
# batch_x, batch_y = train_dataset[0]

# fig, axs = plt.subplots(2, 8, figsize=(15, 5))

# for i in range(batch_size):
#     axs[i // 8, i % 8].imshow(batch_x[i, 0].numpy(), cmap='gray')
#     axs[i // 8, i % 8].axis('off')
#     axs[i // 8, i % 8].set_title(classes[torch.argmax(batch_y[i])])

# plt.show()

#2.MODEL
#Three basic operations: 
 
class Conv2d(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride): 
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.stride = stride

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_size, _ = self.weight.shape

        # Calculate output dimensions
        out_height = (in_height - kernel_size) // self.stride + 1
        out_width = (in_width - kernel_size) // self.stride + 1

        # Initialize output tensor
        output = torch.zeros(batch_size, out_channels, out_height, out_width)

        # Perform convolution
        for b in range(batch_size):
            for c_out in range(out_channels):
                for i in range(0, in_height - kernel_size + 1, self.stride):
                    for j in range(0, in_width - kernel_size + 1, self.stride):
                        # Extract the patch from the input
                        patch = x[b, :, i:i + kernel_size, j:j + kernel_size]

                        # Perform element-wise multiplication with the filter weights and sum
                        output[b, c_out, i // self.stride, j // self.stride] = (patch * self.weight[c_out]).sum() + self.bias[c_out]

        return output



class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape

        # Calculate output dimensions
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        # Initialize output tensor
        output = torch.zeros(batch_size, in_channels, out_height, out_width)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(0, in_height - self.kernel_size + 1, self.stride):
                    for j in range(0, in_width - self.kernel_size + 1, self.stride):
                        # Extract the patch from the input
                        patch = x[b, c, i:i + self.kernel_size, j:j + self.kernel_size]

                        # Perform max pooling
                        output[b, c, i // self.stride, j // self.stride] = patch.max()

        return output
        


class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, x):
        return torch.maximum(torch.tensor(0.0), x) # element-wise compare each value in the input tensor x with zero and returns the maximum of the two. If a value in x is negative, it will be replaced by zero.

# Create instances of the custom neural network operations 
my_convolution = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
my_pool = MaxPool2d(kernel_size=3, stride=2)
my_relu = ReLU()

#create instances of torch nn operations  Q - add the parameters
# nn_convolution = nn.functional.conv2d()
# nn_pool = nn.functional.max_pool2d()
# nn_relu = nn.functional.relu()

# Apply operations to a batch of samples
x, y = train_dataset[0]

# conv_result = my_convolution(x)
# pool_result = my_pool(x)
# relu_result = my_relu(x)
# print(conv_result[:4])
# print(pool_result[:4])
# print(relu_result[:4])


#ORDER OF OPERATIONS
# Create a PyTorch tensor with random elements in the range [-1, 1] of dimensions [1, 100, 100]
input_tensor = torch.rand(1, 100, 100) * 2 - 1
#print(input_tensor[:3])

# # Function one: MaxPool2d(kernel_size=3, stride=2) followed by ReLU
# output_one = my_relu(my_pool(input_tensor.unsqueeze(0)))
# #print(output_one[:3])

# # Function two: ReLU followed by MaxPool2d(kernel_size=3, stride=2)
# output_two = my_pool(my_relu(input_tensor.unsqueeze(0)))
# #print(output_two[:3])

# Compare the outputs
#print("Are the outputs equal?", torch.equal(output_one, output_two)) TRUE

#The outputs are equal. In the case of these two functions the order of executing functions does not matter. 
#The reason for this is in the nature od the given transformations. The ReLU function will set all negative values to zero,
#while max2dpooling will get the maximal value of a certain area. If the maximal value is positive, it will remain the same
# and if it is negative it will be set to zero. Equally, the other weay around, the negative values will be set to zero, and the maximum
#will be picked between zero and other values. In both cases we get the same result. 

# # Function one: MaxPool2d(kernel_size=3, stride=2) followed by ReLU
# output_three = my_convolution(my_relu(input_tensor.unsqueeze(0)))
# print(output_three[:3])

# # Function two: ReLU followed by MaxPool2d(kernel_size=3, stride=2)
# output_four = my_relu(my_convolution(input_tensor.unsqueeze(0)))
# print(output_four[:3])

# Compare the outputs
#print("Are the outputs equal?", torch.equal(output_three, output_four)) FALSE

#In the case of convolution function and ReLu function the results are not the same and the order of functions matters. 
#Convolution is sensitive to ReLUs non-linearity and produces different result with different orders. 

## DEFINING A NEURAL NETWORK

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.block1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU()
        )
        self.block2 = nn.Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU()
        )
        self.final_conv = Conv2d(64, 10, kernel_size=4, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)


# Create an instance of the custom network
custom_net = CustomNet()

# Print the architecture of the custom network
print(custom_net)
    
# PyTorch neural network using Sequential
class PyTorchNet(nn.Module):
    def __init__(self):
        super(PyTorchNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 10, kernel_size=4, stride=1)
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)

# Create an instance of the PyTorch network
pytorch_net = PyTorchNet()

# Print the architecture of the PyTorch network
print(pytorch_net)

#FITTING

def fit(model, number_of_epochs, train_dataset, validation_dataset, batch_size=16, learning_rate=0.01):
    # Define SGD optimizer and CrossEntropy loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize variables to store best model and losses
    best_model = None
    best_validation_loss = float('inf')
    training_losses = []
    validation_losses = []

    # DataLoader for training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

    for epoch in range(number_of_epochs):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0

        # Training loop
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(batch_x)  # Forward pass
            loss = criterion(output, torch.argmax(batch_y, dim=1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_train_loss += loss.item()

        # Calculate average training loss for the epoch
        average_train_loss = running_train_loss / len(train_loader)
        training_losses.append(average_train_loss)

        model.eval()  # Set the model to evaluation mode
        running_validation_loss = 0.0

        # Validation loop
        with torch.no_grad():
            for batch_x_val, batch_y_val in validation_loader:
                output_val = model(batch_x_val)  # Forward pass
                loss_val = criterion(output_val, torch.argmax(batch_y_val, dim=1))  # Compute loss
                running_validation_loss += loss_val.item()

        # Calculate average validation loss for the epoch
        average_validation_loss = running_validation_loss / len(validation_loader)
        validation_losses.append(average_validation_loss)

        # Check if the current model has the best validation performance
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_model = model.state_dict().copy()

        print(f"Epoch {epoch + 1}/{number_of_epochs}, "
              f"Train Loss: {average_train_loss:.4f}, "
              f"Validation Loss: {average_validation_loss:.4f}")

    return best_model, training_losses, validation_losses

best_custom_model, custom_train_losses, custom_val_losses = fit(custom_net, 10, train_dataset, validation_dataset)
best_pytorch_model, pytorch_train_losses, pytorch_val_losses = fit(pytorch_net, 10, train_dataset, validation_dataset)

# # Visualize training and validation losses
# epochs = np.arange(1, 11)
# plt.plot(epochs, custom_train_losses, label='Custom Model (Train)')
# plt.plot(epochs, custom_val_losses, label='Custom Model (Validation)')
# plt.plot(epochs, pytorch_train_losses, label='PyTorch Model (Train)')
# plt.plot(epochs, pytorch_val_losses, label='PyTorch Model (Validation)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


##BONUS TASK



