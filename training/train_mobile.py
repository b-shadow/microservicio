#!/usr/bin/env python3
"""
📊 Script de entrenamiento rápido con MobileNetV2
Más rápido que ResNet50, ideal para validar rápidamente

Uso:
    python train_mobile.py
    python train_mobile.py --epochs 20 --batch-size 32 --lr 0.001
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

CLASSES = {
    'COLISION_VISIBLE': 0,
    'PINCHAZO_LLANTA': 1,
    'HUMO_O_SOBRECALENTAMIENTO': 2,
    'VEHICULO_INMOVILIZADO': 3,
    'SIN_HALLAZGOS_CLAROS': 4,
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}


class IncidentDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx], str(self.image_paths[idx])
        except Exception as e:
            fallback = torch.zeros(3, 224, 224)
            return fallback, self.labels[idx], str(self.image_paths[idx])


def load_dataset():
    """Cargar imágenes"""
    print("📂 Cargando dataset...")
    
    image_paths = []
    labels = []
    
    for class_name, class_idx in CLASSES.items():
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️  Carpeta no encontrada: {class_dir}")
            continue
        
        image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  ✅ {class_name}: {len(image_files)} imágenes")
        
        for img_file in image_files:
            image_paths.append(img_file)
            labels.append(class_idx)
    
    print(f"  📊 Total: {len(image_paths)} imágenes")
    return image_paths, labels


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  GPU: {'SÍ' if torch.cuda.is_available() else 'NO'} | Dispositivo: {device}")
    
    print("\n" + "="*50)
    print("🚀 ENTRENAMIENTO RÁPIDO - MobileNetV2")
    print("="*50)
    
    # Dataset
    image_paths, labels = load_dataset()
    
    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    dataset = IncidentDataset(image_paths, labels, transform=train_transform)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    print(f"  📚 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Modelo
    print("\n🧠 Cargando MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    
    # Congelar capas iniciales
    for param in model.features[:-2].parameters():
        param.requires_grad = False
    
    # Clasificador
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, len(CLASSES))
    )
    
    model = model.to(device)
    
    # Entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\n⏱️  Iniciando entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss, train_acc, correct, total = 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", leave=False)
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Val
        model.eval()
        val_loss, val_acc, correct, total = 0, 0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]", leave=False)
            for images, labels, _ in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f"  Loss - Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print(f"  Acc  - Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        
        # Guardar mejor
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODELS_DIR / "best_mobile.pt")
            print(f"  💾 Mejor modelo guardado")
    
    elapsed = time.time() - start_time
    
    # Test
    print(f"\n{'='*50}")
    print("🧪 EVALUACIÓN EN TEST")
    print(f"{'='*50}")
    
    model.load_state_dict(torch.load(MODELS_DIR / "best_mobile.pt", map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nAccuracy: {test_acc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print("\nReporte:")
    print(classification_report(all_labels, all_preds, target_names=[CLASS_NAMES[i] for i in range(5)]))
    
    # Guardar historial
    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    history['elapsed_seconds'] = elapsed
    
    with open(MODELS_DIR / "history_mobile.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print("✅ COMPLETADO")
    print(f"{'='*50}")
    print(f"Tiempo total: {elapsed/60:.1f}m")
    print(f"Modelo: {MODELS_DIR / 'best_mobile.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
