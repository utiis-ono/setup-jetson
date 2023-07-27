
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from my_model import MyModel

def main():
    model = MyModel()

    # サーバAとサーバBのディレクトリとラウンド数を指定
    dirname_server_a = 'result/test1/weight'
    dirname_server_b = 'result/test2/weight'
    round_number = 3

    state_dict_server_a = torch.load(f"{dirname_server_a}/model_weights_round_{round_number}.pt")
    state_dict_server_b = torch.load(f"{dirname_server_b}/model_weights_round_{round_number}.pt")

    averaged_weights = {k: (state_dict_server_a[k] + state_dict_server_b[k]) / 2 for k in state_dict_server_a.keys()}

    # 統合した重みをモデルに適用
    model.load_state_dict(averaged_weights)

    # データセットの前処理
    transform = transforms.Compose(
        [transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # モデルを評価モードに設定
    model.eval()

    # 評価
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(testloader)

    # 評価指標の計算
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")


if __name__ == '__main__':
    main()
