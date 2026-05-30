```
%pip install scikit-learn matplotlib seaborn tqdm pandas
%pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from tqdm import tqdm
```

```
class DVMColorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels =[]
        
        valid_colors = ['Black', 'White', 'Red', 'Blue', 'Silver', 'Grey', 'Yellow', 'Green', 'Brown']
        self.classes = sorted(valid_colors)
        self.color_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        print("Ищу картинки, жди...")
        all_files = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            parts = filename.split('$$')
            
            found_color = [p for p in parts if p in valid_colors]
            
            if len(found_color) > 0:
                color = found_color[0]
                self.image_paths.append(file_path)
                self.labels.append(self.color_to_idx[color])
                
        print(f"Найдено подходящих машин: {len(self.image_paths)}")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
```

```
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DATA_DIR = './dataset'
full_dataset = DVMColorDataset(root_dir=DATA_DIR)
NUM_CLASSES = len(full_dataset.classes)
print(f"Оставили {NUM_CLASSES} классов: {full_dataset.classes}")

indices = list(range(len(full_dataset)))
targets = full_dataset.labels

train_idx, temp_idx, _, temp_targets = train_test_split(indices, targets, test_size=0.3, stratify=targets, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_targets, random_state=42)

class MySubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self): return len(self.subset)

train_dataset = MySubset(Subset(full_dataset, train_idx), transform=transform_train)
val_dataset = MySubset(Subset(full_dataset, val_idx), transform=transform_test)
test_dataset = MySubset(Subset(full_dataset, test_idx), transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Будем учить на: {device}")
```

```
class MyCustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

```
def train_model(model, name, epochs=4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': [], 'train_f1':[], 'val_f1':[]}
    
    for epoch in range(epochs):
        model.train()
        train_loss, all_preds, all_labels = 0.0, [],[]
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_f1'].append(f1_score(all_labels, all_preds, average='macro'))
        
        model.eval()
        val_loss, all_preds_val, all_labels_val = 0.0, [],[]
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())
                
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(f1_score(all_labels_val, all_preds_val, average='macro'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_title(f'{name} - функция потерь (Loss)')
    ax1.legend()
    
    ax2.plot(history['train_f1'], label='Train F1', marker='o')
    ax2.plot(history['val_f1'], label='Val F1', marker='o')
    ax2.set_title(f'{name} - метрика F1_Macro')
    ax2.legend()
    plt.show()
    
    return model

def evaluate_on_test(model):
    model.eval()
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nФинальный экзамен (тестовая выборка):")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.xlabel('Предсказал')
    plt.ylabel('Реальный цвет')
    plt.title('Матрица ошибок')
    plt.show()
```

```
print("=== 1. Обучаем кастомную сверточную сеть (с нуля) ===")
model_custom = MyCustomCNN(NUM_CLASSES)
model_custom = train_model(model_custom, "My custom CNN", epochs=4)
evaluate_on_test(model_custom)

print("\n=== 2. Обучаем предобученную MobileNetV2 (Transfer Learning) ===")
model_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model_mobilenet.classifier[1] = nn.Linear(model_mobilenet.classifier[1].in_features, NUM_CLASSES)
model_mobilenet = train_model(model_mobilenet, "MobileNetV2 Transfer", epochs=4)
evaluate_on_test(model_mobilenet)

print("\n=== 3. Обучаем предобученную ShuffleNet (Transfer Learning) ===")
model_shufflenet = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
model_shufflenet.fc = nn.Linear(model_shufflenet.fc.in_features, NUM_CLASSES)
model_shufflenet = train_model(model_shufflenet, "ShuffleNet Transfer", epochs=4)
evaluate_on_test(model_shufflenet)
```

```
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def manual_inference(model):
    model.eval()
    idx = random.randint(0, len(test_dataset)-1)
    img_tensor, label_idx = test_dataset[idx]
    
    img_to_show = img_tensor.permute(1, 2, 0).numpy()
    img_to_show = std * img_to_show + mean 
    img_to_show = np.clip(img_to_show, 0, 1)
    
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        _, pred = torch.max(output, 1)
        
    true_color = full_dataset.classes[label_idx]
    pred_color = full_dataset.classes[pred.item()]
    
    plt.imshow(img_to_show)
    plt.title(f"Реальный цвет: {true_color} | Вариант нейросети: {pred_color}")
    plt.axis('off')
    plt.show()

print("Демонстрация инференса:")
manual_inference(model_custom)
```