import os
from PIL import Image
from collections import defaultdict
from pathlib import Path

dataset_dir = Path('../../dataset/processed')
stats = defaultdict(int)
invalid_images = []

print('📊 Analizando dataset...\n')

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

print(f'\n{"Total":30} | {total:3}\n')

if invalid_images:
    print(f'⚠️  {len(invalid_images)} imágenes inválidas encontradas:')
    for name, reason in invalid_images[:5]:
        print(f'  - {name}: {reason}')
        
    if len(invalid_images) > 5:
        print(f'  ... y {len(invalid_images) - 5} más')
else:
    print('✅ Todas las imágenes son válidas')
    
# Recomendaciones
print('\n📋 Recomendaciones:')
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
