# 🚀 QUICK START: Cómo Entrenar tu Modelo

## 📁 Estructura de Carpetas - DÓNDE COLOCAR LAS IMÁGENES

```
microservicio/
│
├── dataset/                              📦 AQUÍ VAN TUS IMÁGENES
│   └── processed/
│       ├── COLISION_VISIBLE/             ← Coloca ~200-300 imágenes de colisiones
│       │   ├── colision_001.jpg
│       │   ├── colision_002.jpg
│       │   └── ...
│       │
│       ├── PINCHAZO_LLANTA/              ← Coloca ~200-300 imágenes con llantas dañadas
│       │   ├── pinchazo_001.jpg
│       │   ├── pinchazo_002.jpg
│       │   └── ...
│       │
│       ├── HUMO_O_SOBRECALENTAMIENTO/    ← Coloca ~200-300 imágenes con humo/daño térmico
│       │   ├── humo_001.jpg
│       │   └── ...
│       │
│       ├── VEHICULO_INMOVILIZADO/        ← Coloca ~200-300 imágenes de vehículos parados
│       │   ├── inmovilizado_001.jpg
│       │   └── ...
│       │
│       └── SIN_HALLAZGOS_CLAROS/         ← Coloca ~200-300 imágenes sin daños claros
│           ├── sin_hallazgos_001.jpg
│           └── ...
│
├── models/                               📊 AQUÍ IRÁN LOS MODELOS ENTRENADOS
│   ├── best.pt                           ← Modelo guardado automáticamente
│   └── history.json                      ← Métricas de entrenamiento
│
├── training/                             🏋️ SCRIPTS DE ENTRENAMIENTO
│   ├── train.py                          ← Script principal (PyTorch)
│   ├── train_fastai.py                   ← Script alternativo (FastAI)
│   ├── data_preparation.py               ← Descargar imágenes automáticamente
│   ├── explore_dataset.py                ← Verificar dataset
│   └── evaluate.py                       ← Evaluar modelo
│
└── inference/
    └── test_model.py                     ← Probar modelo localmente
```

## 📸 OPCIÓN 1: Descargar Imágenes Automáticamente

### Paso 1: Preparar Estructura
```bash
cd microservicio
python setup_training.py
```

### Paso 2: Instalar herramientas de descarga
```bash
cd training
pip install -r requirements_training.txt
```

### Paso 3: Descargar imágenes automáticamente
```bash
python data_preparation.py
```

**Espera a que se complete (~30 minutos dependiendo de velocidad de internet)**

## 📸 OPCIÓN 2: Descargar Imágenes Manualmente

### Para cada clase:

**1️⃣ COLISION_VISIBLE**
- Abre: https://images.google.com/
- Busca: "car collision damage", "vehicle crash"
- Descarga 200-300 imágenes
- Guarda en: `dataset/processed/COLISION_VISIBLE/`
- Elimina duplicados

**2️⃣ PINCHAZO_LLANTA**
- Busca: "tire damage", "burst tire", "damaged wheel"
- Descarga 200-300 imágenes
- Guarda en: `dataset/processed/PINCHAZO_LLANTA/`

**3️⃣ HUMO_O_SOBRECALENTAMIENTO**
- Busca: "car smoke", "engine fire", "overheating vehicle"
- Descarga 200-300 imágenes
- Guarda en: `dataset/processed/HUMO_O_SOBRECALENTAMIENTO/`

**4️⃣ VEHICULO_INMOVILIZADO**
- Busca: "broken down car", "stalled vehicle", "tow truck"
- Descarga 200-300 imágenes
- Guarda en: `dataset/processed/VEHICULO_INMOVILIZADO/`

**5️⃣ SIN_HALLAZGOS_CLAROS**
- Busca: "normal car", "car no damage", "parked vehicle"
- Descarga 200-300 imágenes
- Guarda en: `dataset/processed/SIN_HALLAZGOS_CLAROS/`

### 📋 Requisitos por imagen:
- ✅ Formato: JPG, PNG
- ✅ Tamaño: 100x100px mínimo (sin límite máximo)
- ✅ Tamaño archivo: Máximo 5MB por imagen
- ✅ Contenido: Una o múltiples instancias del incidente
- ❌ Sin: Screenshots, documentos, gráficos

## 🔍 Verificar Dataset

Una vez colocadas las imágenes, verifica que sean válidas:

```bash
cd training
python explore_dataset.py
```

**Salida esperada:**
```
📊 Analizando dataset...

📈 Distribución de imágenes por clase:
------------------------------------------------------
COLISION_VISIBLE               | 250 | ██████████████████████████
PINCHAZO_LLANTA                | 245 | █████████████████████████
HUMO_O_SOBRECALENTAMIENTO      | 200 | ████████████████████
VEHICULO_INMOVILIZADO          | 210 | █████████████████████
SIN_HALLAZGOS_CLAROS           | 295 | ██████████████████████████████

Total                          |1200

✅ Todas las imágenes son válidas
📋 Recomendaciones:
  ✅ Dataset tiene buen tamaño (1200 imágenes)
  ✅ Clases balanceadas adecuadamente
```

## 🏋️ Entrenar el Modelo

### Opción A: PyTorch (Recomendado - Más control)

```bash
cd training
python train.py
```

**Salida:**
```
Cargando dataset desde ../dataset/processed...
✅ Dataset cargado: 960 train, 240 val

🏗️  Cargando ResNet50...

🚀 Entrenando por 50 épocas...

Epoch 1/50  | Train Loss: 2.1523 | Val Loss: 1.8234 | Val Acc: 0.4567
Epoch 2/50  | Train Loss: 1.6234 | Val Loss: 1.2134 | Val Acc: 0.6234
...
Epoch 50/50 | Train Loss: 0.1234 | Val Loss: 0.4567 | Val Acc: 0.9234
  ✅ Mejor modelo guardado

✅ Entrenamiento completado!
Mejor precisión: 0.9234 (92.34%)
```

**Duración esperada:**
- Sin GPU: 30-60 minutos
- Con GPU (CUDA): 5-10 minutos

### Opción B: FastAI (Más simple)

```bash
cd training
python train_fastai.py
```

**Más rápido y automático, ideal para principiantes**

## 📊 Evaluar Modelo

Después del entrenamiento:

```bash
python evaluate.py
```

**Salida esperada:**
```
              precision    recall  f1-score   support

COLISION_VISIBLE                  0.95      0.92      0.93        50
PINCHAZO_LLANTA                   0.91      0.94      0.92        48
HUMO_O_SOBRECALENTAMIENTO         0.88      0.90      0.89        42
VEHICULO_INMOVILIZADO             0.93      0.91      0.92        44
SIN_HALLAZGOS_CLAROS              0.96      0.96      0.96        56

        accuracy                           0.93       240
       macro avg                    0.93      0.93      0.93       240
    weighted avg                    0.93      0.93      0.93       240

✅ Precisión global: 0.9300

Matriz de Confusión:
[[46  1  0  2  1]
 [ 1 45  1  1  0]
 [ 0  1 38  2  1]
 [ 2  0  2 40  0]
 [ 0  0  1  0 55]]
```

## 🚀 Integrar en Microservicio

### Paso 1: Copiar modelo
```bash
# El modelo ya está en models/best.pt después del entrenamiento
# Verifica que exista:
ls -l models/best.pt
```

### Paso 2: Editar `.env`
```env
APP_ENV=production
MODEL_PROVIDER=torch
MODEL_PATH=models/best.pt
LOG_LEVEL=WARNING
```

### Paso 3: Reiniciar servidor
```bash
venv_vision\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Paso 4: Probar
```bash
curl -X POST "http://localhost:8001/predict/image" \
  -F "image=@path/to/test_image.jpg"
```

**Respuesta esperada:**
```json
{
  "ok": true,
  "clase_predicha": "PINCHAZO_LLANTA",
  "confianza": 0.95,
  "especialidad_sugerida": "GOMERIA_LLANTAS",
  "servicio_sugerido": "CAMBIO_LLANTA",
  "nivel_urgencia_sugerido": "MEDIA",
  "modelo_utilizado": "resnet50-custom",
  "version_modelo": "1.0.0"
}
```

## 🎯 Checklist Completo

- [ ] Estructura creada: `python setup_training.py`
- [ ] Imágenes descargadas o colocadas manualmente
- [ ] Dataset verificado: `python explore_dataset.py`
- [ ] Modelo entrenado: `python train.py` (50-60 min)
- [ ] Models/best.pt existe: `ls models/best.pt`
- [ ] `.env` actualizado con MODEL_PROVIDER=torch
- [ ] Servidor reiniciado
- [ ] Endpoint pueba responde correctamente
- [ ] Version en README actualizada

## ⏱️ Tabla de Tiempos

| Tarea | Sin GPU | Con GPU (CUDA) |
|-------|---------|----------------|
| Descargar 1200 imágenes | 1-2 horas | - |
| Verificar dataset | 2-5 min | 2-5 min |
| Entrenar (50 épocas) | 30-60 min | 5-10 min |
| Evaluar | 2-5 min | 1-2 min |
| **TOTAL** | **~40-70 min** | **~15-25 min** |

ƒ## 💡 Tips Importantes

**Para mejor precisión:**
- ✅ Más imágenes = mejor modelo (1000+ es ideal)
- ✅ Dataset balanceado (similar cantidad por clase)
- ✅ Imágenes diversas (diferentes ángulos, iluminación, tamaños)
- ✅ Entrenar por más épocas si val_acc sigue mejorando

**Si el modelo no funciona bien:**
- ❌ Dataset muy pequeño (< 100 por clase)
- ❌ Imágenes irrelevantes (no muestran el incidente)
- ❌ Clases muy similares (ej: casi todos los autos dañados se ven parecido)
- ❌ Overfitting (train_acc alta pero val_acc baja)

**Soluciones:**
1. Agrega más imágenes relevantes
2. Aumenta épocas de entrenamiento
3. Reduce learning rate (0.0001 en lugar de 0.001)
4. Añade más data augmentation
5. Usa modelo más complejo (ResNet152 en lugar de ResNet50)

## 🔗 Recursos

- **Imágenes gratuitas**: Unsplash, Pexels, Pixabay
- **Datasets publicados**: Kaggle, Roboflow, GitHub
- **Herramientas**: Google Colab (GPU gratis), Roboflow (anotaciones)
- **Documentación**: PyTorch, FastAI, Hugging Face

---

**¿Necesitas ayuda?** 
1. Lee TRAINING_GUIDE.md para más detalles
2. Verifica que las imágenes estén en el lugar correcto
3. Usa `explore_dataset.py` para diagnosticar problemas
4. Revisa logs de entrenamiento en models/history.json
