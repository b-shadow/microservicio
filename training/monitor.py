#!/usr/bin/env python3
"""
📊 Monitor de estado de entrenamiento
Verifica el progreso del modelo entrenado
"""

import json
import time
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

def check_training_status():
    """Verificar estado del entrenamiento"""
    
    print("\n" + "="*50)
    print("🔍 MONITOR DE ENTRENAMIENTO")
    print("="*50)
    
    # Verificar archivo de historial
    history_file = MODELS_DIR / "history.json"
    best_model = MODELS_DIR / "best.pt"
    
    if history_file.exists():
        print("✅ Entrenamiento completado!")
        with open(history_file) as f:
            history = json.load(f)
        
        print(f"\n📊 Resultados Finales:")
        print(f"  ✅ Mejor Época: {history.get('best_epoch', 'N/A')}")
        print(f"  ✅ Val Accuracy: {history.get('best_val_acc', 'N/A'):.4f}")
        print(f"  ✅ Test Accuracy: {history.get('test_acc', 'N/A'):.4f}")
        print(f"  ✅ Test F1-Score: {history.get('test_f1', 'N/A'):.4f}")
        elapsed = history.get('elapsed_seconds', 0)
        print(f"  ⏱️  Tiempo Total: {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")
        
    elif best_model.exists():
        print("⏳ Entrenamiento en progreso...")
        print(f"  📦 Modelo parcial guardado: {best_model}")
        print(f"  🕐 Tamaño: {best_model.stat().st_size / (1024*1024):.2f} MB")
        
    else:
        print("⏳ Iniciando entrenamiento...")
        print(f"  📁 Carpeta modelo: {MODELS_DIR}")
    
    # Listar archivos
    print(f"\n📁 Archivos en {MODELS_DIR}:")
    if MODELS_DIR.exists():
        for f in MODELS_DIR.iterdir():
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.2f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.2f} KB"
            else:
                size_str = f"{size} B"
            print(f"  📄 {f.name} ({size_str})")
    else:
        print("  ❌ Carpeta no existe")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    check_training_status()
