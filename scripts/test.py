import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import logging
from train import LynxClassifier, LynxDataset  # Assurez-vous d'importer les bonnes classes depuis votre script d'entraînement

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== CHARGEMENT DU MODÈLE ======================
def load_model(checkpoint_path, device):
    model = LynxClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f'Modèle chargé depuis {checkpoint_path}')
    return model

# ====================== ÉVALUATION DU MODÈLE ======================
def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    logging.info(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# ====================== MAIN FUNCTION ======================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Utilisation du périphérique : {device}')

    # Transformation pour l'ensemble de test
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_root = 'data/dataset'
    test_dataset = LynxDataset(dataset_root, 'test', transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Charger le modèle entraîné
    checkpoint_path = 'checkpoints/best_model.pth'
    model = load_model(checkpoint_path, device)

    # Tester le modèle
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()
