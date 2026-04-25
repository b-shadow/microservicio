#!/usr/bin/env python3
"""
Setup automático de carpetas y estructura para entrenamiento
Ejecutar: python setup_training.py
"""

import os
import sys
from pathlib import Path

def create_training_structure():
    """Crear estructura de carpetas para entrenamiento"""
    
    # Directorio base
    base_dir = Path(__file__).parent
    
    # Estructura de carpetas
    dirs_to_create = [
        'training',
        'dataset/raw',
        'dataset/processed/COLISION_VISIBLE',
        'dataset/processed/PINCHAZO_LLANTA',
        'dataset/processed/HUMO_O_SOBRECALENTAMIENTO',
        'dataset/processed/VEHICULO_INMOVILIZADO',
        'dataset/processed/SIN_HALLAZGOS_CLAROS',
        'dataset/splits',
        'models',
        'inference',
    ]
    
    print('📁 Creando estructura de carpetas para entrenamiento...\n')
    
    for dir_path in dirs_to_create:
        full_path = base_dir / dir_path
        os.makedirs(full_path, exist_ok=True)
        status = '✅ Creada' if full_path.exists() else '❌ Error'
        print(f'  {status}: {dir_path}/')
    
    # Crear archivos de configuración
    print('\n📝 Creando archivos de configuración...\n')
    
    # requirements_training.txt
    training_reqs = """# Dependencias para entrenamiento
torch==2.0.1
torchvision==0.15.2
fastai==2.7.12
pillow>=9.0.0
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.62.0
jupyter>=1.0.0
ipywidgets>=7.6.0

# Opcional: Para descargar datos
bing-image-downloader>=1.1.0
kaggle>=1.5.12
"""
    
    req_path = base_dir / 'training' / 'requirements_training.txt'
    with open(req_path, 'w', encoding='utf-8') as f:
        f.write(training_reqs)
    print(f'  ✅ training/requirements_training.txt')
    
    # .gitignore para directorio de datos
    gitignore = """dataset/raw/
dataset/processed/*/
dataset/splits/
models/*.pt
models/*.pkl
models/*.pth
models/*.onnx
models/history.json
runs/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
"""
    
    git_path = base_dir / '.gitignore'
    if not git_path.exists():
        with open(git_path, 'w', encoding='utf-8') as f:
            f.write(gitignore)
        print(f'  ✅ .gitignore')
    
    # Info README
    info_md = """# 📊 Estructura de Entrenamiento

Esta carpeta contiene la estructura para entrenar modelos de clasificación.

## 📁 Carpetas

- **dataset/raw/** - Imágenes descargadas sin procesar
- **dataset/processed/** - Imágenes organizadas por clase para entrenamiento
- **dataset/splits/** - Sets de train/val/test generados automáticamente
- **training/** - Scripts de entrenamiento
- **models/** - Modelos entrenados (best.pt, best.pkl, etc.)
- **inference/** - Scripts para probar el modelo

## 🚀 Inicio Rápido

1. **Descargar imágenes:**
   ```bash
   # Opción A: Descarga automática
   cd training
   pip install -r requirements_training.txt
   python data_preparation.py
   
   # Opción B: Manual (descargar y guardar en dataset/processed/)
   ```

2. **Explorar dataset:**
   ```bash
   python explore_dataset.py
   ```

3. **Entrenar modelo:**
   ```bash
   python train.py  # PyTorch
   # o
   python train_fastai.py  # FastAI (más simple)
   ```

4. **Evaluar:**
   ```bash
   python evaluate.py
   ```

5. **Usar en microservicio:**
   - Editar `.env`: `MODEL_PROVIDER=ultralytics`
   - Copiar modelo a `../models/best.pt`
   - Reiniciar servidor

## 📊 Requisitos Mínimos

- Mínimo 100 imágenes por clase
- Recomendado: 200-300 imágenes por clase
- Tamaño: 100x100px mínimo, 1024x1024 máximo
- Formato: PNG, JPG, JPEG
- Distribución: Balanceada entre clases

## 📚 Ver guía completa

Lee **TRAINING_GUIDE.md** para instrucciones detalladas.
"""
    
    info_path = base_dir / 'TRAINING_STRUCTURE.md'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(info_md)
    print(f'  ✅ TRAINING_STRUCTURE.md')
    
    # Script de ejemplo: explore_dataset.py
    explore_script = '''import os
from PIL import Image
from collections import defaultdict
from pathlib import Path

dataset_dir = Path('../../dataset/processed')
stats = defaultdict(int)
invalid_images = []

print('📊 Analizando dataset...\\n')

for class_dir in dataset_dir.iterdir():
    if not class_dir.is_dir():
        continue
    
    images = list(class_dir.glob('*.[jJ][pP][gG]')) + list(class_dir.glob('*.[pP][nN][gG]'))
    stats[class_dir.name] = len(images)
    
    # Validar imagenes
    for img_path in images:
        try:
            img = Image.open(img_path)
            
            # Verificar dimensiones
            if img.size[0] < 100 or img.size[1] < 100:
                invalid_images.append((img_path.name, 'Too small'))
            
            # Verificar tamaño file
            file_size_mb = img_path.stat().st_size / (1024*1024)
            if file_size_mb > 5:
                invalid_images.append((img_path.name, f'Too large: {file_size_mb:.2f}MB'))
                
        except Exception as e:
            invalid_images.append((img_path.name, str(e)))

# Reporte
print('📈 Distribución de imágenes por clase:')
print('-' * 50)
total = sum(stats.values())
for class_name in sorted(stats.keys()):
    count = stats[class_name]
    pct = (count/total*100) if total > 0 else 0
    bar = '█' * int(pct/2)
    print(f'{class_name:30} | {count:3} | {bar}')

print(f'\\n{"Total":30} | {total:3}\\n')

if invalid_images:
    print(f'⚠️  {len(invalid_images)} imágenes inválidas encontradas:')
    for name, reason in invalid_images[:5]:
        print(f'  - {name}: {reason}')
        
    if len(invalid_images) > 5:
        print(f'  ... y {len(invalid_images) - 5} más')
else:
    print('✅ Todas las imágenes son válidas')
    
# Recomendaciones
print('\\n📋 Recomendaciones:')
if total < 500:
    print(f'  ⚠️  Total muy bajo ({total} imágenes). Mínimo recomendado: 500')
elif total < 1000:
    print(f'  ⚠️  Dataset pequeño. Considere 1000+ imágenes para mejor precisión')
else:
    print(f'  ✅ Dataset tiene buen tamaño ({total} imágenes)')

unbalanced = max(stats.values()) / (min(stats.values()) + 1) if stats else 0
if unbalanced > 2:
    print(f'  ⚠️  Dataset desbalanceado (ratio {unbalanced:.1f}). Intente balancear clases')
else:
    print(f'  ✅ Clases balanceadas adecuadamente')
'''
    
    explore_path = base_dir / 'training' / 'explore_dataset.py'
    with open(explore_path, 'w', encoding='utf-8') as f:
        f.write(explore_script)
    print(f'  ✅ training/explore_dataset.py')
    
    # Script de ejemplo simple
    test_script = '''#!/usr/bin/env python3
"""
Script rápido para probar una imagen en el modelo local
"""
import sys
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

CLASS_NAMES = [
    'COLISION_VISIBLE',
    'PINCHAZO_LLANTA',
    'HUMO_O_SOBRECALENTAMIENTO',
    'VEHICULO_INMOVILIZADO',
    'SIN_HALLAZGOS_CLAROS'
]

def predict_image(image_path):
    """Predecir clase de una imagen"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo
    model = models.resnet50()
    model.fc = torch.nn.Linear(2048, len(CLASS_NAMES))
    
    try:
        model.load_state_dict(torch.load('../../models/best.pt', map_location=device))
    except FileNotFoundError:
        print('❌ Error: Modelo no encontrado en ../../models/best.pt')
        return
    
    model = model.to(device)
    model.eval()
    
    # Procesar imagen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predicción
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, class_idx = torch.max(probs, 1)
    
    clase = CLASS_NAMES[class_idx.item()]
    conf = confidence.item()
    
    print(f'📊 Predicción: {clase} ({conf:.2%})')
    print(f'\\n📈 Confianzas:')
    for i, prob in enumerate(probs[0]):
        bar = '█' * int(prob * 50)
        print(f'  {CLASS_NAMES[i]:30} | {prob.item():.2%} | {bar}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python test_model.py <image_path>')
        print('\\nEjemplo:')
        print('  python test_model.py ../../dataset/processed/PINCHAZO_LLANTA/imagen.jpg')
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f'❌ Archivo no encontrado: {image_path}')
        sys.exit(1)
    
    predict_image(image_path)
'''
    
    test_path = base_dir / 'inference' / 'test_model.py'
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    print(f'  ✅ inference/test_model.py')
    
    print('\n' + '='*60)
    print('✅ ESTRUCTURA CREADA EXITOSAMENTE')
    print('='*60)
    print('''
📍 Próximos pasos:

1. DESCARGA IMÁGENES:
   Opción A (Automática):
     cd training
     pip install -r requirements_training.txt
     python data_preparation.py
   
   Opción B (Manual):
     Descarga imágenes y colócalas en:
     dataset/processed/COLISION_VISIBLE/
     dataset/processed/PINCHAZO_LLANTA/
     etc.

2. VERIFICA DATASET:
   cd training
   python explore_dataset.py

3. ENTRENA MODELO:
   python train.py        # PyTorch (más control)
   # o
   python train_fastai.py # FastAI (más simple)

4. INTEGRA EN MICROSERVICIO:
   - Copia models/best.pt
   - Edita .env: MODEL_PROVIDER=ultralytics
   - Reinicia servidor

📚 Lee TRAINING_GUIDE.md para más detalles
''')

if __name__ == '__main__':
    create_training_structure()
