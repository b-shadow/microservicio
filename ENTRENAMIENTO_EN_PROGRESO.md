# 🚀 ESTADO DE ENTRENAMIENTO - Microservicio de IA

## 📊 Estado Actual (13 de Abril 2026)

### Entrenamientos en Progreso

#### 1️⃣ **ResNet50** - ACTIVO ⏳
- **Estado**: Epoch 1/25 en progreso
- **Modelo**: ResNet50 (Transfer Learning)
- **Datos**: 1046 imágenes (836 train, 104 val, 106 test)
- **Configuración**:
  - Épocas: 25
  - Batch Size: 16
  - Learning Rate: 0.001
  - Dispositivo: CPU
- **Tiempo Estimado**: ~30 minutos por época ≈ 12-15 horas total
- **Terminal**: 7f9069a0-4991-4897-95d9-97f758310175
- **Ventaja**: Mayor precisión

#### 2️⃣ **MobileNetV2** - ACTIVO ⏳
- **Estado**: Descargando modelo preentrenado
- **Modelo**: MobileNetV2 (Más ligero)
- **Datos**: 1046 imágenes (836 train, 104 val, 106 test)
- **Configuración**:
  - Épocas: 15
  - Batch Size: 32
  - Learning Rate: 0.001
  - Dispositivo: CPU
- **Tiempo Estimado**: ~3-5 minutos por época ≈ 45-75 minutos total
- **Terminal**: 27839452-ac00-4d78-8850-8a37184a0263
- **Ventaja**: Más rápido, suficiente precisión

---

## 📂 Dataset Cargado

| Clase | Imágenes | Estado |
|-------|----------|--------|
| 🚗 COLISION_VISIBLE | 250 | ✅ |
| 🛞 PINCHAZO_LLANTA | 240 | ✅ |
| 💨 HUMO_O_SOBRECALENTAMIENTO | 100 | ✅ |
| 🛑 VEHICULO_INMOVILIZADO | 210 | ✅ |
| ✅ SIN_HALLAZGOS_CLAROS | 246 | ✅ |
| **TOTAL** | **1046** | **✅** |

---

## 🎯 Próximos Pasos

### Cuando MobileNet Complete (próximas 1-2 horas)
```bash
# Probar el modelo
python inference/test_model.py --image "ruta/imagen.jpg"

# Ver métricas
python training/monitor.py
```

### Cuando ResNet50 Complete (próximas 12-15 horas)
```bash
# Cambiar a mejor modelo en app
# Actualizar .env: MODEL_PROVIDER=resnet50
```

---

## 🔧 Scripts Disponibles

| Script | Descripción | Tiempo |
|--------|-------------|--------|
| `train.py` | ResNet50 (Más preciso) | 12-15h |
| `train_mobile.py` | MobileNetV2 (Rápido) | 45-75m |
| `monitor.py` | Ver progreso | 1s |
| `test_model.py` | Probar imagen | 5s |
| `explore_dataset.py` | Analizar dataset | 2m |

---

## 📁 Estructura de Archivos

```
microservicio/
├── models/
│   ├── best.pt              ← ResNet50 (cuando complete)
│   ├── best_mobile.pt       ← MobileNetV2 (cuando complete)
│   ├── history.json         ← Métricas ResNet
│   ├── history_mobile.json  ← Métricas MobileNet
│   └── training_history.png ← Gráficas
│
├── dataset/
│   └── processed/
│       ├── COLISION_VISIBLE/ (250 imgs)
│       ├── PINCHAZO_LLANTA/ (240 imgs)
│       ├── HUMO_O_SOBRECALENTAMIENTO/ (100 imgs)
│       ├── VEHICULO_INMOVILIZADO/ (210 imgs)
│       └── SIN_HALLAZGOS_CLAROS/ (246 imgs)
│
└── training/
    ├── train.py
    ├── train_mobile.py
    ├── monitor.py
    └── explore_dataset.py
```

---

## 🎓 Información de Entrenamiento

### Transfer Learning
- Ambos modelos usan **pesos preentrenados** de ImageNet
- Solo se reentrena la última capa
- Esto permite usar menos datos y entrenar más rápido

### Capas Congeladas
- **ResNet50**: Capas 1-2 congeladas, capas 3-4 entrenables
- **MobileNetV2**: Features[:-2] congeladas, últimas capas entrenables

### Augmentación de Datos
- Rotations: ±15°
- Flip horizontal: 50%
- Color jitter: ±20%

---

## 💡 Recomendaciones

### Ahora
- ✅ Dejar entrenamientos corriendo
- ✅ MobileNet terminará primero (~1-2 horas)
- ✅ ResNet50 terminará después (~12-15 horas)

### Cuando Complete MobileNet
```bash
# Probar con imagen de prueba
python inference/test_model.py --image dataset/processed/COLISION_VISIBLE/co001.jpg

# Ver métricas
python training/monitor.py
```

### Cuando Complete ResNet50
```bash
# Si ResNet es más preciso, cambiar en app/main.py
# Actualizar carga del modelo
```

---

## 📊 Métricas Esperadas

Basado en dataset balanceado (~200 imgs/clase):
- **MobileNet**: ~85-90% accuracy
- **ResNet50**: ~90-95% accuracy

---

## ⚠️ Notas Importantes

- ✅ Se usando **CPU** (sin GPU) - más lento pero funciona
- ✅ Los modelos se guardan automáticamente cuando mejora el validation accuracy
- ✅ Historial completo se guarda en JSON con gráficas
- ✅ Puedes cancelar con `Ctrl+C` en cualquier momento

---

**Actualizado**: 13 de Abril 2026, 01:30 GMT
