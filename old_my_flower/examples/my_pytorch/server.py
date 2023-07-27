from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.strategy import FedAvg
import sys

import time
import csv
import subprocess
import os
import pandas as pd
import numpy as np
import torch

from my_model import MyModel

args = sys.argv

if len(args) < 5:
    print("Usage: python3 server.py <dir name> <rounds> <time_out[s]> <round time[s]>")
    sys.exit()

dirname = 'result/' + args[1]
timeout = int(args[3]) #[s]Time outの時間
round_time = int(args[4]) #[s]実測値
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
    config=fl.server.ServerConfig(num_rounds=int(args[2]), round_timeout=timeout*100),#timeoutはsim用のパラメータなので実機でやる時は*100を無くしたほうがいいかも
    strategy=strategy,
)

# Preprocessing data
#file_path = dirname + "/log/result_serverlog.csv"

#if os.path.exists(file_path):
#    command = "python3 data_preprocessing.py " + dirname
#    subprocess.call(command.split())
#else:
#    print("If you want the result, please try sim_client.py")

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
