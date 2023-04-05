from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import sys

import time
import csv
import subprocess
import os
import pandas as pd
import numpy as np

args = sys.argv

if len(args) < 3:
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
#make_csv = ["Round", "Real Time [s]", "Sim Time[s]","Accuracy","received","faliure","p_rate [%]"]

# CustomStrategyの定義
class CustomStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

# Define metric aggregation function
# Add global lists to store results
losses_distributed = []
metrics_distributed = []

# Define metric aggregation function
def weighted_average(rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]) -> Tuple[float, Metrics]:
    global losses_distributed
    global metrics_distributed

    # Multiply accuracy of each client by number of examples used
    accuracies = [(evaluate_res.metrics["accuracy"] * evaluate_res.num_examples) / evaluate_res.num_examples for _, evaluate_res in results]
    #accuracies = [evaluate_res.num_examples * evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
    examples = [evaluate_res.num_examples for _, evaluate_res in results]
    time_now = time.perf_counter()

    # Save losses and accuracies
    losses = [evaluate_res.loss for _, evaluate_res in results]
    losses_distributed.append(losses)
    metrics_distributed.append(accuracies)

    #print("metrics:", metrics)
    print("time:", time_now)
    print("losses_distributed:", losses_distributed)
    print("metrics_distributed:", metrics_distributed)

    # Aggregate and return custom metric (weighted average)
    time_list.append(time_now)
    if sum(examples) == 0:
        return float('inf'), {"accuracy": 0}
    else:
        return sum(losses) / len(losses), {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = CustomStrategy(
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
    config=fl.server.ServerConfig(num_rounds=int(args[2]), round_timeout=timeout),
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
        p_rate_list.append((receive_list[i]+timeout_list[i])/receive_list[i]*100)

    if timeout_list[i] == 0:
        simtime_list.append(round_time+simtime_list[i])
    else:
        simtime_list.append(timeout+simtime_list[i])

simtime_list.pop(0)

# 無効な算術演算と除算のワーニングを無視し、'nan'を返すように設定する
print('print',len(metrics_distributed))
print('print',metrics_distributed)

mean_accuracy = [np.mean(x) if len(x) > 0 else np.nan for x in metrics_distributed]
mean_loss = [np.mean(x) if len(x) > 0 else np.nan for x in losses_distributed]

#if len(metrics_distributed) > 0:
#    mean_accuracy = np.mean(metrics_distributed, axis=1)
#else:
#    mean_accuracy = np.nan

#if len(losses_distributed) > 0:
#    mean_loss = np.mean(losses_distributed, axis=1)
#else:
#    mean_loss = np.nan 

file_path = dirname + "/result_server.csv"
df = pd.DataFrame({'Round' :round_list,
                  'Real Time[s]': time_list,
                  'Sim Time[s]': simtime_list,
                  'Accuracy': mean_accuracy,
                  'Loss': mean_loss,
                  'receved': receive_list,
                  'failue': timeout_list,
                  'p_rate[%]': p_rate_list})

df.to_csv(file_path,index = False)
