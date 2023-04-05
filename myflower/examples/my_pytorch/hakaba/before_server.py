from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics
import sys


import time
import csv
import subprocess
import os
import pandas as pd

args = sys.argv

if len(args) < 3:
    print("Usage: python3 server.py <dir name> <rounds> <time_out[s]>")
    sys.exit()

dirname = 'result/' + args[1]
time_list = []
make_csv = ["Round", "Time [s]"]





# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    time_now = time.perf_counter()
    #print("accuracy:",accuracies)
    print("metrics:",metrics)
    print("time:", time_now)
    # Aggregate and return custom metric (weighted average)
    time_list.append(time_now)

    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,  # Train on 25 clients (each round)
        fraction_evaluate=1,  # Evaluate on 50 clients (each round)
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
        )


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=int(args[2]),round_timeout=int(args[3])),
    strategy=strategy,
    )


#preprocessing data
file_path = dirname + "/log/result_serverlog.csv"

if os.path.exists(file_path):
    command = "python3 data_preprocessing.py " + dirname
    subprocess.call(command.split())
else:
    print("If you want the result please try sim_client.py")



file_path = dirname + "/result_server.csv"
if os.path.exists(file_path):
    #time_list = [x - time_list[0] for x in time_list]
    #read result_server.csv
    save_data = pd.read_csv(file_path)
    save_data.insert(1,"Real Time [s]",time_list)
    save_data.to_csv(file_path,index=False)
else:
    print("If you want the result please try sim_client.py")



#filename = dirname + "/result_servertime.csv"
#with open(filename, 'w') as f:
#    writer = csv.writer(f,lineterminator = '\n')
#    writer.writerow(make_csv)

#for i in range(0,len(time_list),1):
#    make_csv = []
#    make_csv.append(i+1)
#    make_csv = []
#    with open(filename, 'a') as f:
#        writer = csv.writer(f,lineterminator = '\n')
#        writer.writerow(make_csv)
