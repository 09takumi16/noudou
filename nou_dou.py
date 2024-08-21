import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import numpy as np

# 1. データの準備
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('path_to_train_data', transform=transform)
validation_data = datasets.ImageFolder('path_to_validation_data', transform=transform)
test_data = datasets.ImageFolder('path_to_test_data', transform=transform)

# 2. 初期トレーニングデータの準備
n_initial = 50  # 初期サンプル数
initial_idx = np.random.choice(range(len(train_data)), size=n_initial, replace=False)
initial_data = Subset(train_data, initial_idx)

# 初期データのDataLoader
initial_loader = DataLoader(initial_data, batch_size=32, shuffle=True)

# 3. AlexNetモデルの初期化
model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(train_data.classes))

# モデルのトレーニング関数
def train_model(model, data_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 初期トレーニング
train_model(model, initial_loader)

# 4. ActiveLearnerの設定
learner = ActiveLearner(
    estimator=model,
    query_strategy=uncertainty_sampling
)

# 5. 能動学習ループ
n_queries = 10  # 能動学習のクエリ回数

for i in range(n_queries):
    # 残りの未ラベルデータプール
    remaining_indices = list(set(range(len(train_data))) - set(initial_idx))
    remaining_data = Subset(train_data, remaining_indices)
    
    # データローダーの作成
    remaining_loader = DataLoader(remaining_data, batch_size=32, shuffle=False)
    
    # 最も不確実なサンプルを選択
    for images, _ in remaining_loader:
        outputs = learner.estimator(images)
        uncertainty = -torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1).values
        query_idx = uncertainty.argmax().item()
        query_instance = remaining_data[query_idx]
        break
    
    # ラベル付けされたサンプルを追加して再学習
    initial_idx.append(remaining_indices[query_idx])
    train_model(model, DataLoader(Subset(train_data, initial_idx), batch_size=32, shuffle=True))
    
    print(f'Query {i + 1}/{n_queries} completed.')

# 6. テストデータでの評価
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Final model accuracy: {accuracy:.4f}')
