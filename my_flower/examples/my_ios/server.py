import os
import csv
import time
import flwr
import argparse
import numpy as np
import torch


class SaveModelStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, directory="test", **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.start_time = time.time()
        self.standart_time = self.start_time
        self.times = []
        self.total_times = []

    def aggregate_fit(self, rnd: int, results, failures):
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Measure time
            elapsed_time = time.time() - self.start_time
            self.times.append(elapsed_time)
            self.total_times.append(time.time() - self.standart_time)
            self.start_time = time.time()

            # Save weights
            print(f"Saving round {rnd} weights...")
            #dir_path = os.path.join("result", self.directory, f"round-{rnd}-weights.pt")
            dir_path = os.path.join("result", self.directory, f"round-{rnd}-weights.npz")
            os.makedirs(os.path.dirname(dir_path), exist_ok=True)
            np.savez(dir_path, *weights)
        return weights

    def save_times(self):
        with open(f"result/{self.directory}/times.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "per_time", "total_time"])
            for i, (t, t_t) in enumerate(zip(self.times, self.total_times), 1):
                writer.writerow([i, t, t_t])


def main(directory, num_clients=1, num_rounds=1) -> None:
    strategy = SaveModelStrategy(
        directory=directory,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # Start Flower server
    hist = flwr.server.start_server(
        #server_address="10.17.79.112:8080",
        #server_address="172.20.40.195:8080",
        server_address="172.20.10.14",
        #server_address="169.254.220.233",
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Save times
    strategy.save_times()

    return hist


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--clients", type=int, help="minimum number of clients", default=1)
    argParser.add_argument("-r", "--rounds", type=int, help="number of rounds", default=1)
    argParser.add_argument("-d", "--directory", type=str, help="dir name", default="test")
    args = argParser.parse_args()
    hist = main(args.directory, args.clients, args.rounds)
