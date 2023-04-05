from flwr.server.strategy import FedAvg
from sim_client2 import EvaluationError  # sim_client2.py から EvaluationError をインポート

class MyStrategy(FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )

    def weighted_average(self, metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    def evaluate(self, *args, **kwargs):
        try:
            return super().evaluate(*args, **kwargs)
        except EvaluationError as e:
            print("evaluate_round received 0 results and 1 failure:", e)
            return None, {"accuracy": 0.0}

    # 他のカスタムメソッドやオーバーライドが必要なメソッドをここに追加

