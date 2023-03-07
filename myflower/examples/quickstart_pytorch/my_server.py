from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import sys

args = sys.argv

if len(args) < 2:
    print("rounds,time_out[s]")
    sys.exit()

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
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
    server_address="10.108.251.152:8080",
    config=fl.server.ServerConfig(num_rounds=int(args[1]),round_timeout=int(args[2])),
    strategy=strategy,
    )
