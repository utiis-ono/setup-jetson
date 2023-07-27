import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.datasets import MNIST
import numpy as np
from my_model import MyModel

def load_npz_as_state_dict(file_path):
    loaded = np.load(file_path, allow_pickle=True)
    return {key: torch.from_numpy(loaded[key]) for key in loaded.keys()}

def main():
    model = MyModel()

    # iosで計算したモデルのディレクトリを指定
    dirname_server = 'result/dd'
    round_number = 1
    
    npz = np.load(f"{dirname_server}/round-{round_number}-weights.npz",allow_pickle=True)
    print(npz.files)
    print(npz['arr_0'].shape)
    print(npz['arr_0'].dtype)
    #print(npz['arr_0'])
    #print(npz['arr_1'])

    # モデルに適用
    new_state_dict = {
        "conv1.weight": torch.from_numpy(npz['arr_0']),  # Assuming 'arr_0' corresponds to 'conv1.weight'
        "conv1.bias": torch.from_numpy(npz['arr_1']),  # Assuming 'arr_1' corresponds to 'conv1.bias'
        # Add other parameters as necessary
    }
    model.load_state_dict(new_state_dict)

    # データセットの前処理
    transform = transforms.Compose(
        [transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
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
