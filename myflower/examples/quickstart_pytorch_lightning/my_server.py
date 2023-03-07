import flwr as fl
import sys

args = sys.argv

if len(args) < 2:
    print("rounds, time_out[s]")
    sys.exit()

def main() -> None:
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        #config=fl.server.ServerConfig(num_rounds=10),
        config=fl.server.ServerConfig(num_rounds=int(args[1]),round_timeout=int(args[2])),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
