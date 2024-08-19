import argparse
import csv
import flwr as fl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import Metrics, Parameters
from flwr.server.strategy import FedAvg
from collections import OrderedDict
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

parser = argparse.ArgumentParser(description='CIFAR-100 selective training')
parser.add_argument('--dir_name', default='test', type=str, help='Decide dir name :default to "test"')
parser.add_argument('--rounds', default=3, type=int, help='Decide number of rounds :default to 3')
parser.add_argument('--timeout', default=60, type=int, help='Decide timeout[s] :default to 60')
parser.add_argument('--roundtime', default=35, type=int, help='Decide roundtime[s] :default to 35')
parser.add_argument('--pretrained_weights', default="None", type=str, help='Path to .pt file with pretrained weights :default to "None"')
args = parser.parse_args()


dirname = f'result/CIFAR-100/{args.dir_name}'  
timeout = int(args.timeout) #[s]Time outの時間
round_time = int(args.roundtime) #[s]実測値
round_list = []
time_list = []
simtime_list = [0]
receive_list = []
timeout_list = []
p_rate_list = []
# Add global lists to store results
losses_distributed = []
metrics_distributed = []
precision_distributed = []
recall_distributed = []
fscore_distributed = []
#make_csv = ["Round", "Real Time [s]", "Sim Time[s]","Accuracy","received","faliure","p_rate [%]"]

# CustomStrategyの定義
class CustomStrategy(FedAvg):
    def __init__(self, model: torch.nn.Module, *args, **kwargs):
    #def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.evaluate_metrics_aggregation_fn = weighted_average

    def aggregate_evaluate(self, rnd, results, failures):
        successful_results = []
        num_timeouts = 0
        for r in results:
            client, evaluate_res = r
            if evaluate_res.loss == float('inf'):
                num_timeouts += 1
            else:
                successful_results.append(r)

        #if num_timeouts > 0:
        print(f"evaluate_round {rnd} received {len(successful_results)} results and {num_timeouts} timeouts")    
        round_list.append(rnd)
        receive_list.append(len(successful_results))
        timeout_list.append(num_timeouts)


        # Call the evaluate_metrics_aggregation_fn with rnd and successful_results
        return self.evaluate_metrics_aggregation_fn(rnd, successful_results)
    
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        #print(aggregated_weights)
        if not os.path.exists(f'{dirname}/weight'):
            os.makedirs(f'{dirname}/weight')

        torch.save(model.state_dict(), f"{dirname}/weight/model_weights_round_{rnd}.pt")

        return aggregated_weights


# Define metric aggregation function
def weighted_average(rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]) -> Tuple[float, Metrics]:
    global losses_distributed
    global metrics_distributed
    global precision_distributed
    global recall_distributed
    global fscore_distributed

    # Multiply accuracy of each client by number of examples used
    accuracies = [(evaluate_res.metrics["accuracy"] * evaluate_res.num_examples) / evaluate_res.num_examples for _, evaluate_res in results]
    precisions = [evaluate_res.metrics["precision"] for _, evaluate_res in results]
    recalls = [evaluate_res.metrics["recall"] for _, evaluate_res in results]
    f_scores = [evaluate_res.metrics["f_score"] for _, evaluate_res in results]

    examples = [evaluate_res.num_examples for _, evaluate_res in results]
    time_now = time.perf_counter()

    # Save losses and accuracies
    losses = [evaluate_res.loss for _, evaluate_res in results]
    

    losses_distributed.append(losses)
    metrics_distributed.append(accuracies)
    precision_distributed.append(precisions)
    recall_distributed.append(recalls)
    fscore_distributed.append(f_scores)

    #print("metrics:", metrics)
    print("time:", time_now)
    print("losses_distributed:", losses_distributed)
    print("metrics_distributed:", metrics_distributed)
    print("precision_distributed:", precision_distributed)
    print("recall_distributed:", recall_distributed)
    print("fscore_distributed:", fscore_distributed)

    # Aggregate and return custom metric (weighted average)
    time_list.append(time_now)
    if sum(examples) == 0:
        return float('inf'), {"accuracy": 0}
    else:
        return sum(losses) / len(losses), {"accuracy": sum(accuracies) / sum(examples)}

model = MyModel()

if args.pretrained_weights != "None":
    checkpoint = torch.load(args.pretrained_weights)
    model.load_state_dict(checkpoint)
# Define strategy
strategy = CustomStrategy(
        model,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
        )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    #server_address="192.168.0.238:8080",
    config=fl.server.ServerConfig(num_rounds=int(args.rounds), round_timeout=timeout*100),#timeoutはsim用のパラメータなので実機でやる時は*100を無くしたほうがいいかも
    strategy=strategy,
)


for i in range(len(round_list)):
    if receive_list[i] == 0:
        p_rate_list.append(0)
    else:
        p_rate_list.append(receive_list[i]/(timeout_list[i]+receive_list[i])*100)

    if timeout_list[i] == 0:
        simtime_list.append(round_time+simtime_list[i])
    else:
        simtime_list.append(timeout+simtime_list[i])

simtime_list.pop(0)

# 無効な算術演算と除算のワーニングを無視し、'nan'を返すように設定する
print('metrics_distributed',len(metrics_distributed))
mean_accuracy = [np.mean(x) if len(x) > 0 else np.nan for x in metrics_distributed]
print('loss_distributed',len(losses_distributed))
mean_loss = [np.mean(x) if len(x) > 0 else np.nan for x in losses_distributed]
print('precision_distributed',len(precision_distributed))
mean_precision = [np.mean(x) if len(x) > 0 else np.nan for x in precision_distributed]
print('recall_distributed',len(recall_distributed))
mean_recall = [np.mean(x) if len(x) > 0 else np.nan for x in recall_distributed]
print('fscore_distributed',len(fscore_distributed))
mean_fscore = [np.mean(x) if len(x) > 0 else np.nan for x in fscore_distributed]

print("finish mean")


file_path = dirname + "/result_server.csv"
df = pd.DataFrame({'Round': round_list,
                   'Real Time[s]': time_list,
                   'Sim Time[s]': simtime_list,
                   'Accuracy': mean_accuracy,
                   'Loss': mean_loss,
                   'Precision': mean_precision,
                   'Recall': mean_recall,
                   'F-score': mean_fscore,
                   'received': receive_list,
                   'failure': timeout_list,
                   'p_rate[%]': p_rate_list})

df.to_csv(file_path, index=False)
