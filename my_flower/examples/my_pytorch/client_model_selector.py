import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import csv
import pandas as pd
import numpy as np
import sys
import os 
# Import additional libraries
from typing import List, Tuple

#from flwr.common import FitRes,  Parameters
from flwr.common.typing import EvaluateRes
from sklearn.metrics import precision_recall_fscore_support

args = sys.argv

if len(args) < 3:
    print("Usage: python3 client_model_selector.py <dataset_name> <dir name> <node id>")
    sys.exit()

dataset_name = args[1]
dirname = 'result/'+ dataset_name + '/' + args[2]
node_id = args[3]

if not os.path.exists(dirname):
    os.makedirs(dirname)

time_list = [0]
loss_list = []
accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []
make_csv =["Round" , "Time [s]","Accuracy", "Loss", "Precision", "Recall", "F-score"]

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print(DEVICE)

class MNIST_Net(nn.Module):
    print("Model (simple CNN for MNIST)")

    def __init__(self) -> None:
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # MNISTは1チャンネル（グレースケール）
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CIFAR10_Net(nn.Module):
    print("""Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')""")
    print("Model (simple CNN for CIFAR-10)")

    def __init__(self) -> None:
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    print("""Train the model on the training set.""")
    time_now = time.perf_counter()
    print("now",time_now)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, file=sys.stdout):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

def test(net, testloader):
    print("""Validate the model on the test set.""")
    time_now = time.perf_counter()
    print("nowtest",time_now)
    
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for images, labels in tqdm(testloader, file=sys.stdout):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
            # calculate true positives, false positives, true negatives, false negatives
            predicted = torch.max(outputs.data, 1)[1]
            y_pred_list.append(predicted.cpu().numpy())
            y_true_list.append(labels.cpu().numpy())
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
    loss_get = loss / len(testloader.dataset)
    accuracy_get = correct / total
    time_now = time.perf_counter()
    time_list.append(time_now)
    print("Loss ",loss / len(testloader.dataset)," accuracy ", correct / total, " total " ,total)
    print("now",time_now)
    
    return loss/ len(testloader.dataset), correct /total, y_true_list, y_pred_list
    
def load_data():
    """Load MNIST (training and test set)."""
    print("Load MNIST (training and test set).")
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])  # MNISTのための正規化
    trainset = MNIST("./data", train=True, download=True, transform=trf)
    testset = MNIST("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def load_data(dataset_name):
    if dataset_name == "MNIST":
        trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        trainset = MNIST("./data", train=True, download=True, transform=trf)
        testset = MNIST("./data", train=False, download=True, transform=trf)
    elif dataset_name == "CIFAR-10":
        trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = CIFAR10("./data", train=True, download=True, transform=trf)
        testset = CIFAR10("./data", train=False, download=True, transform=trf)
    else:
        raise ValueError("Dataset not recognized. Use 'MNIST' or 'CIFAR-10'.")
    
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
#net = Net().to(DEVICE)
# Load data
trainloader, testloader = load_data(dataset_name)

# Load model
if dataset_name == "MNIST":
    net = MNIST_Net().to(DEVICE)
elif dataset_name == "CIFAR-10":
    net = CIFAR10_Net().to(DEVICE)  
else:
    raise ValueError("Dataset not recognized. Use 'MNIST' or 'CIFAR-10'.")



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config) -> EvaluateRes:
        self.set_parameters(parameters)
        loss, accuracy, y_true, y_pred = test(net, testloader)
        
        # sklearnのprecision_recall_fscore_supportを使って、各指標を計算します。
        precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(f_score)

        custom_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
        }
        return loss, len(testloader.dataset), custom_metrics


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)

filename = dirname + "/result_client-" + node_id  + ".csv"
with open(filename, 'w') as f:
    writer = csv.writer(f,lineterminator = '\n')
    writer.writerow(make_csv )

for i in range(0,len(accuracy_list),1):
    make_csv = []
    make_csv.append(i+1)
    make_csv.append(time_list[i+1])
    make_csv.append(accuracy_list[i])
    make_csv.append(loss_list[i])
    make_csv.append(precision_list[i])
    make_csv.append(recall_list[i])
    make_csv.append(fscore_list[i])
    with open(filename, 'a') as f:
        writer = csv.writer(f,lineterminator = '\n')
        writer.writerow(make_csv)
