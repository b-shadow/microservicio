#!/usr/bin/env python3
r"""
🔍 Script de inferencia - Clasificar imágenes con el modelo entrenado

Uso:
    python inference/test_model.py --image path/to/image.jpg
    python inference/test_model.py --image "C:\ruta\imagen.jpg"
"""

import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import json

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

CLASSES = {
    0: 'COLISION_VISIBLE',
    1: 'PINCHAZO_LLANTA',
    2: 'HUMO_O_SOBRECALENTAMIENTO',
    3: 'VEHICULO_INMOVILIZADO',
    4: 'SIN_HALLAZGOS_CLAROS',
}

DESCRIPTIONS = {
    'COLISION_VISIBLE': '🚗 Colisión o daño visible en el vehículo',
    'PINCHAZO_LLANTA': '🛞 Neumático pinchado o dañado',
    'HUMO_O_SOBRECALENTAMIENTO': '💨 Humo o signos de sobrecalentamiento',
    'VEHICULO_INMOVILIZADO': '🛑 Vehículo detenido/inmovilizado',
    'SIN_HALLAZGOS_CLAROS': '✅ Sin problemas aparentes',
}

def load_model(model_path, device):
    """Cargar modelo entrenado"""
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return None
    
    # Detectar tipo de modelo por nombre
    if 'mobile' in model_path.name:
        print("📦 Cargando MobileNetV2...")
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)
        )
    else:
        print("📦 Cargando ResNet50...")
        model = models.resnet50(weights=None)
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
            nn.Linear(128, 5)
        )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def classify_image(image_path, model, device):
    """Clasificar una imagen"""
    
    # Cargar imagen
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error al cargar imagen: {e}")
        return None
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = CLASSES[predicted.item()]
    confidence_value = confidence.item()
    
    return {
        'class': class_name,
        'confidence': confidence_value,
        'probabilities': {CLASSES[i]: prob.item() for i, prob in enumerate(probabilities[0])},
        'description': DESCRIPTIONS[class_name],
        'image_shape': image.size
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Dispositivo: {device}\n")
    
    # Seleccionar modelo
    if args.model == 'auto':
        # Buscar mejor modelo disponible
        mobile_model = MODELS_DIR / "best_mobile.pt"
        resnet_model = MODELS_DIR / "best.pt"
        
        if mobile_model.exists():
            model_path = mobile_model
        elif resnet_model.exists():
            model_path = resnet_model
        else:
            print("❌ No se encontraron modelos entrenados en", MODELS_DIR)
            print("   Entrena un modelo primero con: python training/train.py")
            sys.exit(1)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            model_path = MODELS_DIR / args.model
    
    print(f"📦 Modelo: {model_path.name}")
    print(f"📊 Tamaño: {model_path.stat().st_size / (1024*1024):.2f} MB\n")
    
    # Ver historial del modelo
    history_path = model_path.parent / f"history{'_mobile' if 'mobile' in model_path.name else ''}.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print(f"📈 Entrenamiento:")
        print(f"   Época: {history.get('best_epoch', 'N/A')}")
        print(f"   Val Acc: {history.get('best_val_acc', 0):.4f}")
        print(f"   Test Acc: {history.get('test_acc', 0):.4f}\n")
    
    # Cargar modelo
    model = load_model(model_path, device)
    if model is None:
        sys.exit(1)
    
    print(f"✅ Modelo cargado\n")
    
    # Clasificar imagen
    if not args.image:
        print("❌ Especifica una imagen con: --image path/to/image.jpg")
        sys.exit(1)
    
    result = classify_image(args.image, model, device)
    
    if result is None:
        sys.exit(1)
    
    print("="*60)
    print("🔍 RESULTADO DE CLASIFICACIÓN")
    print("="*60)
    print(f"Imagen: {args.image}")
    print(f"Tamaño: {result['image_shape']}")
    print(f"\n🎯 Predicción: {result['class']}")
    print(f"📋 {result['description']}")
    print(f"🎲 Confianza: {result['confidence']*100:.2f}%")
    
    print(f"\n📊 Probabilidades:")
    for class_name, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        print(f"   {class_name:<30} {prob:>6.4f}  {bar}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificar imágenes con modelo entrenado")
    parser.add_argument('--image', type=str, help='Ruta a la imagen a clasificar')
    parser.add_argument('--model', type=str, default='auto', help='Modelo a usar (auto, best.pt, best_mobile.pt)')
    args = parser.parse_args()
    main(args)
