import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.metrics import f1_score
import numpy as np

"""
Initialises the net and specifies the amount of layers and neurons
The forward function which represents the feedforward action in the MLP is also defined in the Net class
"""


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fcl1 = nn.Linear(input_size, 300)
        self.fcl2 = nn.Linear(300, output_size)
        self.fcl3 = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.m = nn.Sigmoid()

    def forward(self, x):
        x1 = self.dropout(F.relu(self.fcl1(x)))
        x2 = self.dropout(F.relu(self.fcl2(x1)))
        return self.m(self.fcl3(x2))


# The eval_performance function takes testing data and evaluates the network based on it, returning the F1-Score
def eval_performance(model_name, test_features, test_labels, unique_labels):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    # Import trained MLP
    checkpoint = torch.load(model_name, map_location=device)
    net = Net(test_features.shape[1], test_labels.shape[1])
    net.load_state_dict(checkpoint['model_state_dict'])
    # Set to evaluation
    net.eval()

    # Transform the data into the correct format for the MLP
    test_tensor_X = torch.tensor(test_features, dtype=torch.float)
    test_tensor_y = torch.tensor(test_labels, dtype=torch.float)
    validate = data_utils.TensorDataset(test_tensor_X, test_tensor_y)
    """
    The train loader holds all the data and serves as an iterator with batch_size defining how large batches
    are evaluated at the same time
    """
    train_loader_test = data_utils.DataLoader(validate, batch_size=100, shuffle=False)
    # This variable holds the predictions from the MLP
    predicted_labels = np.array([])
    # Iterated over all the validation data
    for sample in train_loader_test:
        feat, label = sample
        pred = net(Variable(feat.to(device), requires_grad=False))
        pred = pred.cpu().detach().numpy()
        # The threshold for which sigmoid value should accept the predicted label as correct is set here
        pred = (pred > 0.1).astype(int)
        predicted_labels = np.append(predicted_labels, pred)
    predicted_labels = predicted_labels.reshape(-1, unique_labels)
    # Calculates F1-Score
    fsample = f1_score(test_labels, predicted_labels, average='samples')
    print("F1-Score sample", fsample)
    return fsample


# The MLP is created in this function ans stored for future iterations to use
def mlp_process(iteration, train_features, train_labels, input_size, output_size):
    # Determines the batch size and amount of epochs to train
    batch_size = 540
    # Determines amount of epochs to train for
    epochs = 100
    print(torch.__version__)
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True
    net = Net(input_size, output_size)
    net.cuda()

    # Transform the data into the correct format for the MLP
    train_tensor_X = torch.tensor(train_features, dtype=torch.float)
    train_tensor_y = torch.tensor(train_labels, dtype=torch.float)
    train = data_utils.TensorDataset(train_tensor_X, train_tensor_y)
    train_loader = data_utils.DataLoader(train, batch_size, shuffle=True, num_workers=30, pin_memory=True)

    # Set the network to training mode
    net.train()

    # Define error/loss algorithm
    criterion = torch.nn.MSELoss()
    # Define optimizer function /learning algorithm
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    error = []
    # Iterate through the epochs, training the MLP
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        # Clear error value each epoch
        error.clear()
        for sample in train_loader:
            features, labels = sample
            local_batch, local_labels = features.to(device), labels.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Pass batch to network
            outputs = net.forward(Variable(local_batch, requires_grad=True))
            # calculate loss
            loss = criterion(outputs, local_labels)
            error.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()

        # Error value were used during testing to determine loss after each epoch and total loss for a part of the data
        # print(sum(error)/len(error))
    # Save model for next iteration to use
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, "mynet{}".format(iteration + 1) + ".pt")


# Contrinues the training process by loading a previously stored model and carries on training
def mlp_train_more(iteration, train_features, train_labels):
    # Determines the batch size and amount of epochs to train
    batch_size = 540
    # Determines amount of epochs to train for
    epochs = 100

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    # Import trained net
    checkpoint = torch.load("mynet{}".format(iteration) + ".pt", map_location=device)
    net = Net(train_features.shape[1], train_labels.shape[1])
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)
    net.cuda()

    # Set to training mode
    net.train()

    # Transform the data into the correct format for the MLP
    train_tensor_X = torch.tensor(train_features, dtype=torch.float)
    train_tensor_y = torch.tensor(train_labels, dtype=torch.float)
    train = data_utils.TensorDataset(train_tensor_X, train_tensor_y)
    train_loader = data_utils.DataLoader(train, batch_size, shuffle=True, num_workers=30, pin_memory=True)

    # Define error/loss algorithm
    criterion = torch.nn.MSELoss()
    # Define optimizer function /learning algorithm
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # Load optimizer values from saved model
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load loss from saved model
    loss = checkpoint['loss']
    error = []
    # Iterate through the epochs, training the MLP
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))

        error.clear()
        for sample in train_loader:
            features, labels = sample
            local_batch, local_labels = features.to(device), labels.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Pass batch to network
            outputs = net.forward(Variable(local_batch, requires_grad=True))
            # calculate loss
            loss = criterion(outputs, local_labels)
            error.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()

    # Error value were used during testing to determine loss after each epoch and total loss for a part of the data
    # print(sum(error)/len(error))

    # Save model for next iteration to use
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, "mynet{}".format(iteration + 1) + ".pt")
