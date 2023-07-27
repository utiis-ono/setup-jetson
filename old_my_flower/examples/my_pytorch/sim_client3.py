import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


# 先頭にカスタム例外を追加
#class EvaluationError(Exception):
#    pass


args = sys.argv

if len(args) != 5:
    print("Usage: python3 sim_client2.py <dir name> <node id> <round time[s]> <prog [%]>")
    sys.exit()

dirname = 'result/' + args[1] 
if not os.path.exists(dirname):
    os.makedirs(dirname)


round_time = int(args[3]) #[s]実測値
time_list = [0]
loss_list = []
accuracy_list = []
prog_list = []
make_csv =["Round" , "Sim Time [s]" , "Loss" ,"Accuracy" ,"prog [%]"]
#make_csv =[[]]
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print(DEVICE)



class Net(nn.Module):
    print("""Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')""")

    def __init__(self) -> None:
        super(Net, self).__init__()
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
    print("""------------------------------------""")
    
    time_now = time.perf_counter()

    print("nowtrain", time_now)
    print("""------------------------------------""")

    time_now = time.perf_counter()
    time_list.append(time_now)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, file=sys.stderr):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    print("""Validate the model on the test set.""")
    print("now",time.perf_counter())
    time_now = time.perf_counter()
    time_list.append(time_now)
    print("nowtest",time_now)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    prog = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, file=sys.stderr):
            prog = prog + 1
            if int(args[4]) <= (prog/len(testloader)*100):
                break
            else:
                outputs = net(images.to(DEVICE))
                labels = labels.to(DEVICE)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    loss_get = loss / len(testloader.dataset)
    accuracy_get = correct / total
    time_list.append(time.perf_counter())

    print("Loss ",loss / len(testloader.dataset)," accuracy ", correct / total, " total " ,total)
    print("now",time_list[-1])
    
    loss_list.append(loss_get)
    accuracy_list.append(accuracy_get)
    prog_list.append(prog/len(testloader)*100)
    return loss/ len(testloader.dataset), correct /total

    
    
def load_data():
    """Load CIFAR-10 (training and test set)."""
    print("""Load CIFAR-10 (training and test set).""")
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=128, shuffle=True), DataLoader(testset)
    #return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

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

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(),
)

filename = dirname + "/result_client-" + args[2] + ".csv"
with open(filename, 'w') as f:
    writer = csv.writer(f,lineterminator = '\n')
    writer.writerow(make_csv )

for i in range(0,len(accuracy_list),1):
    make_csv = []
    make_csv.append(i+1)
    make_csv.append(time_list[i+1])
    make_csv.append(loss_list[i])
    make_csv.append(accuracy_list[i])
    make_csv.append(prog_list[i])
    with open(filename, 'a') as f:
        writer = csv.writer(f,lineterminator = '\n')
        writer.writerow(make_csv)

#del make_csv[-1]
#df = pd.DataFrame(make_csv, columns = ['Round' , 'Time [s]' , 'Loss' , 'Accuracy'])

#df.to_csv(filename)

#with open(filename, 'w') as f:
#    writer = csv.writer(f,lineterminator = '\n')
#    writer.writerow(make_csv)



