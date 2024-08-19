import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib
from tqdm import tqdm
from collections import OrderedDict
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from tqdm import tqdm

class_dict = {
    # Your class dictionary
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your custom dataset
class MyCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

parser = argparse.ArgumentParser(description='CIFAR-100 selective training')
parser.add_argument('--rounds', default=10, type=int, help='Number of tuning rounds')
parser.add_argument('--dir_name', default='test', type=str, help='Decide dir name')
#parser.add_argument('--select_train', default=0, type=int, help='Select class to exclude from 0 to 4 (5 to exclude none)')
#parser.add_argument('--select_test', default=0, type=int, help='Select class to exclude from 0 to 4 (5 to exclude none)')
parser.add_argument('--select_train', default=[0,1,2,3,4], nargs='+', type=int, help='Select classes to include from 0 to 4')
parser.add_argument('--select_test', default=[0,1,2,3,4], nargs='+', type=int, help='Select classes to include from 0 to 4')
parser.add_argument('--model_paths', nargs='+', type=str, help='Paths to .pt files of trained models')
args = parser.parse_args()

# モデルパスが指定されていない場合はエラーメッセージを表示し、プログラムを終了します
if args.model_paths is None:
    print("Error: No model paths were provided. Please provide the paths to the .pt files of the trained models.")
    exit()

#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])
#データ拡張
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

map_dict = {original_class: superclass for superclass, classes in class_dict.items() for original_class in classes}
remap = lambda x: map_dict[x]

trainset = MyCIFAR100(root='./data', train=True, download=True, transform=transform, target_transform=remap)
testset = MyCIFAR100(root='./data', train=False, download=True, transform=transform, target_transform=remap)

# Include selected class if necessary
# For trainset
if any(0 <= s <= 4 for s in args.select_train):
    include_labels = [class_dict[i][s] for s in args.select_train for i in class_dict]
    train_indices = [i for i in range(len(trainset)) if trainset.targets[i] in include_labels]
    trainset = Subset(trainset, train_indices)

# For testset
if any(0 <= s <= 4 for s in args.select_test):
    include_labels = [class_dict[i][s] for s in args.select_test for i in class_dict]
    test_indices = [i for i in range(len(testset)) if testset.targets[i] in include_labels]
    testset = Subset(testset, test_indices)

#if 0 <= args.select_train <= 4:
#    exclude_labels = [class_dict[i][args.select_train] for i in class_dict]
#    train_indices = [i for i in range(len(trainset)) if trainset.targets[i] not in exclude_labels]
#    trainset = Subset(trainset, train_indices)
#
#if 0 <= args.select_test <= 4:
#    exclude_labels = [class_dict[i][args.select_test] for i in class_dict]
#    test_indices = [i for i in range(len(testset)) if testset.targets[i] not in exclude_labels]
#    testset = Subset(testset, test_indices)

    print('trainset',trainset)
    print('testset',testset)
    print('train dataset',args.select_train)
    print('test dataset',args.select_test)

# Define data loader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# 例として単純なCNNを使用します
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)  # 変更: 20クラスに対応

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = SimpleCNN()
net.to(device)

# Load each model and take the average of parameters
model_state_list = [torch.load(model_path, map_location=device) for model_path in args.model_paths]
avg_model_state = OrderedDict()

for key in model_state_list[0].keys():
    avg_model_state[key] = torch.stack([model_state[key] for model_state in model_state_list]).mean(dim=0)

net.load_state_dict(avg_model_state)

#new_dir = f'result/superclass/marge_train{args.select_train}_test{args.select_test}'
train_list = '-'.join(map(str, args.select_train))
test_list = '-'.join(map(str, args.select_test))
new_dir = f'result/marge/{args.dir_name}'
os.makedirs(new_dir, exist_ok=True)
os.makedirs(f'{new_dir}/models', exist_ok=True)
os.makedirs(f'{new_dir}/confusion_matrix', exist_ok=True)
os.makedirs(f'{new_dir}/confusion_matrix/train/', exist_ok=True)
os.makedirs(f'{new_dir}/confusion_matrix/test/', exist_ok=True)
print("Please enter your comment. Enter 'END' on a new line to finish.")
lines = []
while True:
    line = input()
    if line == "END":
        break
    lines.append(line)
comment = "\n".join(lines)
file_path = os.path.join(new_dir, "_README.md")
with open(file_path, "w") as file:
    file.write(comment)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Metrics dataframes
metrics_train = {"round": [], "accuracy": [], "loss": [], "recall": [], "precision": [], "f1_score": []}
metrics_test  = {"round": [], "accuracy": [], "loss": [], "recall": [], "precision": [], "f1_score": []}

# Add training loop here
for epoch in range(args.rounds):  # loop over the dataset multiple times
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Add labels to y_true and y_pred for training metrics calculation
        _, predicted = torch.max(outputs.data, 1)
        y_true_train.extend(labels.tolist())
        y_pred_train.extend(predicted.tolist())
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Save model state at the end of each round
    torch.save(net.state_dict(), f"{new_dir}/models/model_{epoch+1}.pt")

    # Calculate metrics for the round
    accuracy_train = accuracy_score(y_true_train, y_pred_train)
    recall_train = recall_score(y_true_train, y_pred_train, average='macro')
    precision_train = precision_score(y_true_train, y_pred_train, average='macro', zero_division=0)
    f1_train = f1_score(y_true_train, y_pred_train, average='macro', zero_division=0)

    # Store metrics for this round
    metrics_train["round"].append(epoch+1)
    metrics_train["accuracy"].append(accuracy_train)
    metrics_train["loss"].append(running_loss / len(trainloader))
    metrics_train["recall"].append(recall_train)
    metrics_train["precision"].append(precision_train)
    metrics_train["f1_score"].append(f1_train)

    # Create confusion matrix and save as .png
    super_class_name = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 
    'fruit and vegetables', 'household electrical\ndevices', 
    'household furniture', 'insects', 'large carnivores', 
    'large man-made\noutdoor things', 'large natural\noutdoor scenes',
    'large omnivores\nand herbivores', 'medium-sized\nmammals', 
    'non-insect\ninvertebrates', 'people', 'reptiles', 
    'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
    ]
    cm = confusion_matrix(y_true_train, y_pred_train)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
    fig, ax = plt.subplots(figsize=(10,10))
    heatmap = sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.2f', xticklabels=super_class_name, yticklabels=super_class_name, vmin=0, vmax=1, annot_kws={"size": 9})
    heatmap.set_xticklabels(heatmap.get_xticklabels())
    heatmap.set_yticklabels(heatmap.get_yticklabels())
    
    # Change color of x-axis labels
    for  i, label in enumerate(heatmap.get_xticklabels()):
        label.set_color('black' if i % 2 == 0 else 'brown')

    for  i, label in enumerate(heatmap.get_yticklabels()):
        label.set_color('black' if i % 2 == 0 else 'brown')

    ax.set_xlabel('Predicted labels',size = 18)
    ax.set_ylabel('True labels', size = 18)
    ax.set_title('Round '+ str(epoch+1))
    ax.tick_params(axis='both', which='major', labelsize=10)  # Change font size here
    plt.savefig(f'{new_dir}/confusion_matrix/train/cm_train_round_{str(epoch+1)}.pdf', bbox_inches='tight')
    plt.close()

    # Test the network
    y_true_test = []
    y_pred_test = []

    test_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
    
            y_true_test.extend(labels)
            y_pred_test.extend(predicted.tolist())
    
    # 評価指標を計算
    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    loss_test = test_loss / len(testloader.dataset)
    recall_test = recall_score(y_true_test, y_pred_test, average='macro')
    precision_test = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    f1_test = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    
    metrics_test["round"].append(epoch+1)
    metrics_test["accuracy"].append(accuracy_test)
    metrics_test["loss"].append(loss_test)
    metrics_test["recall"].append(recall_test)
    metrics_test["precision"].append(precision_test)
    metrics_test["f1_score"].append(f1_test)

    cm = confusion_matrix(y_true_test, y_pred_test)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
    fig, ax = plt.subplots(figsize=(10,10))
    heatmap = sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.2f', xticklabels=super_class_name, yticklabels=super_class_name, vmin=0, vmax=1, annot_kws={"size": 9})
    heatmap.set_xticklabels(heatmap.get_xticklabels())
    heatmap.set_yticklabels(heatmap.get_yticklabels())

    # Change color of x-axis labels
    for  i, label in enumerate(heatmap.get_xticklabels()):
        label.set_color('black' if i % 2 == 0 else 'brown')

    for  i, label in enumerate(heatmap.get_yticklabels()):
        label.set_color('black' if i % 2 == 0 else 'brown')

    ax.set_xlabel('Predicted labels',size = 18)
    ax.set_ylabel('True labels', size = 18)
    ax.set_title('Marge')
    ax.tick_params(axis='both', which='major', labelsize=10)  # Change font size here

    plt.savefig(f'{new_dir}/confusion_matrix/test/cm_test_round_{str(epoch+1)}.pdf', bbox_inches='tight')
    plt.close()

    print('Round %d finished' % (epoch+1))

# Convert metrics data to dataframe and save as .csv
df_metrics_train = pd.DataFrame(metrics_train)
df_metrics_train.to_csv(f'{new_dir}/metrics_train.csv', index=False)

df_metrics_test  = pd.DataFrame(metrics_test)
df_metrics_test.to_csv(f'{new_dir}/metrics_test.csv', index=False)

# Calculate metrics
cm = confusion_matrix(y_true_test, y_pred_test)
df_cm = pd.DataFrame(cm)
df_cm.to_csv(f'{new_dir}/confusion_matrix/confusion_matrix_test.csv')

report = classification_report(y_true_test, y_pred_test, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f'{new_dir}/classification_report.csv')

print('Finished Training')
