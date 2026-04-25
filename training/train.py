#!/usr/bin/env python3
"""
📊 Script de entrenamiento para modelo de clasificación de incidentes vehiculares
Modelo: ResNet50 con transfer learning
Datos: 5 clases de incidentes

Uso:
    python train.py
    
    Con parámetros:
    python train.py --epochs 30 --batch-size 16 --lr 0.001
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== CONFIGURACIÓN ====================
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = BASE_DIR / "dataset" / "splits"

MODELS_DIR.mkdir(exist_ok=True)
SPLITS_DIR.mkdir(exist_ok=True)

CLASSES = {
    'COLISION_VISIBLE': 0,
    'PINCHAZO_LLANTA': 1,
    'HUMO_O_SOBRECALENTAMIENTO': 2,
    'VEHICULO_INMOVILIZADO': 3,
    'SIN_HALLAZGOS_CLAROS': 4,
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# ==================== DATASET ====================
class IncidentDataset(torch.utils.data.Dataset):
    """Dataset personalizado para imágenes de incidentes"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Abrir imagen en RGB
            image = Image.open(img_path).convert('RGB')
            
            # Aplicar transformaciones
            if self.transform:
                image = self.transform(image)
            
            return image, label, str(img_path)
        except Exception as e:
            print(f"⚠️  Error cargando {img_path}: {e}")
            # Retornar imagen gris como fallback
            fallback = torch.zeros(3, 224, 224)
            return fallback, label, str(img_path)


def load_dataset():
    """Cargar imágenes del dataset"""
    print("📂 Cargando dataset...")
    
    image_paths = []
    labels = []
    
    for class_name, class_idx in CLASSES.items():
        class_dir = DATASET_DIR / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Carpeta no encontrada: {class_dir}")
            continue
        
        # Obtener todas las imágenes
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        print(f"  ✅ {class_name}: {len(image_files)} imágenes")
        
        for img_file in image_files:
            image_paths.append(str(img_file))
            labels.append(class_idx)
    
    total = len(image_paths)
    print(f"\n📊 Total: {total} imágenes")
    
    if total == 0:
        print("❌ No se encontraron imágenes en el dataset")
        sys.exit(1)
    
    return image_paths, labels


def create_data_loaders(image_paths, labels, batch_size=16, num_workers=0):
    """Crear DataLoaders para train/val/test"""
    print(f"\n🔀 Dividiendo dataset (80/10/10)...")
    
    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Crear dataset
    dataset = IncidentDataset(image_paths, labels, transform=train_transform)
    
    # Dividir
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"  📚 Train: {len(train_dataset)}")
    print(f"  📚 Val: {len(val_dataset)}")
    print(f"  📚 Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# ==================== MODELO ====================
def create_model(num_classes=5):
    """Crear modelo ResNet50 con transfer learning"""
    print("\n🧠 Creando modelo ResNet50...")
    
    # Cargar modelo preentrenado
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Congelar capas iniciales (excepto últimas)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # Cambiar clasificador final
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    
    return model


# ==================== ENTRENAMIENTO ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrenar una época"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for images, labels, paths in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Actualizar barra
        avg_loss = total_loss / (pbar.n + 1)
        acc = 100 * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validar modelo"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Val", leave=False)
        for images, labels, paths in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), correct / total


def test(model, test_loader, device):
    """Evaluar en test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Test", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n📊 Resultados en Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Reporte detallado
    print("\n📋 Reporte de clasificación:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[CLASS_NAMES[i] for i in range(5)],
        digits=4
    ))
    
    return accuracy, f1


# ==================== MAIN ====================
def main(args):
    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    print(f"📦 GPU disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*50)
    print("🚀 ENTRENAMIENTO - Modelo de Clasificación")
    print("="*50)
    
    # Cargar datos
    image_paths, labels = load_dataset()
    train_loader, val_loader, test_loader = create_data_loaders(
        image_paths, labels, batch_size=args.batch_size
    )
    
    # Crear modelo
    model = create_model(num_classes=len(CLASSES))
    model = model.to(device)
    
    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Entrenamiento
    print(f"\n🔧 Configuración:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Dispositivo: {device}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\n⏱️  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*50)
    
    for epoch in range(args.epochs):
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        
        # Entrenar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validar
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Scheduler
        scheduler.step()
        
        # Mostrar resultados
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = MODELS_DIR / "best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Mejor modelo guardado ({val_acc:.4f})")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tiempo total: {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")
    
    # Evaluar en test set
    print(f"\n{'='*50}")
    print("🧪 EVALUACIÓN EN TEST SET")
    print(f"{'='*50}")
    
    # Cargar mejor modelo
    best_model_path = MODELS_DIR / "best.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    test_acc, test_f1 = test(model, test_loader, device)
    
    # Guardar historial
    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    history['elapsed_seconds'] = elapsed
    
    history_path = MODELS_DIR / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Historial guardado: {history_path}")
    
    # Visualizar entrenamiento
    plot_training_history(history)
    
    print(f"\n{'='*50}")
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*50}")
    print(f"Mejor época: {best_epoch}/{args.epochs}")
    print(f"Mejor Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"\nModelo: {best_model_path}")


def plot_training_history(history):
    """Graficar historial de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = MODELS_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=100)
    print(f"📈 Gráfica guardada: {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenar modelo de clasificación de incidentes"
    )
    parser.add_argument(
        '--epochs', type=int, default=25,
        help='Número de épocas (default: 25)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Tamaño de batch (default: 16)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    args = parser.parse_args()
    main(args)
