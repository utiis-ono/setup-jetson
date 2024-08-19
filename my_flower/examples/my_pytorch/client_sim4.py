import argparse
import csv
import flwr as fl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import ssl
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from PIL import Image
from collections import OrderedDict
from flwr.common.typing import EvaluateRes
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
from torchvision import datasets, transforms
#from torchvision.datasets import cifar100
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from typing import List, Tuple

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Command line arguments
parser = argparse.ArgumentParser(description='CIFAR-100 selective training')
parser.add_argument('--dir_name', default='test', type=str, help='Decide dir name :default to "test"')
parser.add_argument('--node_id', default=1, type=int, help='Select node_id :default to 1')
parser.add_argument('--timeout', default=60, type=int, help='Decide timeout[s] :default to 60')
parser.add_argument('--roundtime', default=35, type=int, help='Decide roundtime[s] :default to 35')
parser.add_argument('--method', default="std", type=str, help='Select method type std or my :default to "std"')
parser.add_argument('--select_train', default=[0,1,2,3,4], nargs='+', type=int, help='Select classes to include from 0 to 4 ex) "--select_train 0 1 2" :default to 0 1 2 3 4')
parser.add_argument('--select_test', default=[0,1,2,3,4], nargs='+', type=int, help='Select classes to include from 0 to 4 ex) "--select_test 2 3 4" :default to 0 1 2 3 4')
args = parser.parse_args()

time_out = int(args.timeout) #サーバのタイムアウト[s]
round_time = int(args.roundtime) #[s]実測値

#if time_out < round_time:
#    print("time outをround timeよりも短く設定することはできません")
#    sys.exit()

train_list = '-'.join(map(str, args.select_train))
test_list = '-'.join(map(str, args.select_test))
dirname = f'result/CIFAR-100/{args.dir_name}/train{train_list}/test{test_list}'  
if not os.path.exists(dirname):
    os.makedirs(dirname)

dirpath = f"nodes/node_{args.node_id}.csv"
df = pd.read_csv(dirpath)
max_time = df['time'].max()
print(max_time)

BATCH_SIZE = 128  # 適切なバッチサイズを設定
time_list = [0]
loss_list = []
accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []
prog_list = []
make_csv =["Round" , "Sim Time [s]","Accuracy", "Loss", "Precision", "Recall", "F-score", "prog [%]"]
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print(DEVICE)

# Define dictionary for classes
class_dict = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]   # vehicles 2
    }

class MyCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Creating the mapping
map_dict = {}
for superclass, classes in class_dict.items():
    for original_class in classes:
        map_dict[original_class] = superclass
remap = lambda x: map_dict[x]

def load_data():
    """Load CIFAR-10 (training and test set)."""
    print("""Load CIFAR-10 (training and test set).""")
    trainset = MyCIFAR100(root='./data', train=True, download=True, transform=transform, target_transform=remap)
    testset = MyCIFAR100(root='./data', train=False, download=True, transform=transform, target_transform=remap)
    # Include selected class if necessary
    # For trainset
    if any(0 <= s <= 4 for s in args.select_train):
        include_labels = [class_dict[i][s] for s in args.select_train for i in class_dict]
        train_indices = [i for i in range(len(trainset)) if trainset.targets[i] in include_labels]
        trainset = Subset(trainset, train_indices)
    # For testset
    if any(0 <= s <= 4 for s in args.select_test):
        include_labels = [class_dict[i][s] for s in args.select_test for i in class_dict]
        test_indices = [i for i in range(len(testset)) if testset.targets[i] in include_labels]
        testset = Subset(testset, test_indices)
    print('trainset',trainset)
    print('testset',testset)
    print('train dataset',args.select_train)
    print('test dataset',args.select_test)
    return DataLoader(trainset, batch_size=128, shuffle=True), DataLoader(testset)

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

class Net(nn.Module):
    print("""Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')""")

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    #def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs):
    print("""Train the model on the training set.""")
    print("""------------------------------------""")
    
    #time_now = time.perf_counter()
    time_now = time_list[-1] #シミュレーションはじめ
    reference_index = time_now 
    in_or_out_value = df.loc[reference_index, 'in_or_out']

    print("nowtrain", time_now)
    print(reference_index)
    print(f"in_or_out: {in_or_out_value}")
    print("""------------------------------------""")

    while in_or_out_value == 0:
        print("out of spot")
        #time.sleep(1)
        #time_now = time.perf_counter()
        time_now = time_now + 1
        reference_index = time_now
        print("nowtrain", time_now)
        print(reference_index)
        in_or_out_value = df.loc[reference_index, 'in_or_out']
        print(f"in_or_out: {in_or_out_value}")
        print("""------------------------------------""")
        
    #time_list.append(time_now)
    print("in spot")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, file=sys.stderr):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    print("""Validate the model on the test set.""")
    #print("now",time.perf_counter())
    time_now = time_list[-1]
    time_list.append(time_now)
    print("nowtest",time_now)
    
    #time_now = time.perf_counter()
    reference_index = time_now
    in_or_out_value = df.loc[reference_index, 'in_or_out']
    
    while in_or_out_value == 0:
        print("out of spot")
        time_now = time_now + 1
        #time.sleep(1)
        #time_now = time.perf_counter()
        reference_index = time_now
        print("nowtest", time_now)
        print(reference_index)
        in_or_out_value = df.loc[reference_index, 'in_or_out']
        print(f"in_or_out: {in_or_out_value}")
        print("""------------------------------------""")
        
    print("in spot")
    
    check_move_spot = -1 #スポットから出そうになるかどうかを判定するための変数
    pre_prog = 0
    
    for i in range(round_time):
        #print('debug',time_now, 'before potision',check_move_spot,'now potision',in_or_out_value)
        pre_prog = pre_prog + 1
        time_now = time_now + 1
        reference_index = time_now
        in_or_out_value = df.loc[reference_index, 'in_or_out']
        lane = df.loc[reference_index,'y']
        if pre_prog > time_out:
            print("time_out")
            print("""------------------------------------""")
            pre_prog = (pre_prog / round_time) * 100
            break

        elif in_or_out_value == 0 and check_move_spot == 1:
            pre_prog = (pre_prog / round_time) * 100
            print(check_move_spot)
            print(reference_index)
            print(f"Lane: {lane}")
            print(f"in_or_out: {in_or_out_value}")
            print(f"send in-progress model pre {pre_prog}%")
            print("""------------------------------------""")
            break
        else:
            check_move_spot = in_or_out_value
        
    else:
        print("train compleate")
        pre_prog = 100
    
    
    prog = 0
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for images, labels in tqdm(testloader, file=sys.stderr):
            prog = prog + 1
            if pre_prog <= (prog/len(testloader)*100):
                print(check_move_spot)
                print(reference_index)
                print(f"Lane: {lane}")
                print(f"in_or_out: {in_or_out_value}")
                print("""------------------------------------""")
                break
            else:
                check_move_spot = in_or_out_value

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
    time_list[-1] = time_now
    #time_list.append(time.perf_counter())

    print("Loss ",loss / len(testloader.dataset)," accuracy ", correct / total, " total " ,total)
    print("now",time_list[-1])
    
    if pre_prog == 100:
        print(f"send {prog/len(testloader)*100}% model")
        prog_list.append(prog/len(testloader)*100)
        return loss/ len(testloader.dataset), correct /total, y_true_list, y_pred_list
    else:
        if args.method == 'std':##領域外に行ってしまったのでモデルを送信できない
            prog_list.append(0)
            print("Timeout occurred. Not sending the results to the server.")
            return float('inf'),0, [], []
        else:
            prog_list.append(prog/len(testloader)*100)
            print(f"send {prog/len(testloader)*100}% model")
            return loss/ len(testloader.dataset), correct /total, y_true_list, y_pred_list
    

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-100)
net = Net().to(DEVICE)
trainloader, testloader = load_data()
#trainloader, testloader, x_train, y_train, x_test, y_test = load_data()

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

    # Update the return statement in MyClient's evaluate function
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
    server_address="0.0.0.0:8080",
    client=FlowerClient(),
)

filename = f'{dirname}/result_client-{args.node_id}.csv'
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
    make_csv.append(prog_list[i])
    with open(filename, 'a') as f:
        writer = csv.writer(f,lineterminator = '\n')
        writer.writerow(make_csv)

