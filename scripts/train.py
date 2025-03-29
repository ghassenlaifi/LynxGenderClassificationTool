import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== DATASET CLASS ======================
class LynxDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.dataset_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['male', 'female']
        
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, class_name)
            if not os.path.exists(class_dir):
                raise RuntimeError(f"Directory not found: {class_dir}")
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        logging.info(f"Loaded {split} dataset: {len(self.image_paths)} images "
                     f"({self.labels.count(0)} male, {self.labels.count(1)} female)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ====================== MODEL CLASS ======================
class LynxClassifier(nn.Module):
    def __init__(self):
        super(LynxClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Use ResNet18 for efficiency
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)  # Two classes: Male and Female
        )

    def forward(self, x):
        return self.model(x)

# ====================== TRAINING FUNCTIONS ======================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# ====================== CHECKPOINT FUNCTION ======================
def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'model_class': 'LynxClassifier'
    }, filename)

# ====================== MAIN FUNCTION ======================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_root = 'data/dataset'
    train_dataset = LynxDataset(dataset_root, 'train', transform_train)
    val_dataset = LynxDataset(dataset_root, 'val', transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    
    model = LynxClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
    
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    num_epochs = 10
    best_val_acc = 0.0
    patience = 5  # Early stopping patience
    early_stop_counter = 0

    for epoch in range(num_epochs):
        logging.info(f'\nEpoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logging.info(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, 'checkpoints/best_model.pth')
            logging.info("Checkpoint saved.")
            early_stop_counter = 0  # Reset counter if model improves
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            logging.info("Early stopping triggered.")
            break

if __name__ == '__main__':
    main()
