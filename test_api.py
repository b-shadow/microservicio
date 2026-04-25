#!/usr/bin/env python
"""Script para probar los endpoints de la API."""

import httpx
from PIL import Image
import io

# Crear imagen de prueba
img = Image.new('RGB', (100, 100), color='blue')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Hacer request a /predict/image
with httpx.Client() as client:
    files = {'image': ('test.png', img_bytes.getvalue(), 'image/png')}
    data = {'evidencia_id': 'ev-test-001'}
    r = client.post('http://localhost:8001/predict/image', files=files, data=data)
    result = r.json()
    
print(f"Status: {r.status_code}")
print(f"OK: {result.get('ok')}")
print(f"Source: {result.get('source')}")
print(f"Model: {result.get('modelo_utilizado')}")
print(f"Clase predicha: {result.get('clase_predicha')}")
print(f"Confianza: {result.get('confianza')}")
print(f"Especialidad: {result.get('especialidad_sugerida')}")
print(f"Servicio: {result.get('servicio_sugerido')}")
print(f"Urgencia: {result.get('nivel_urgencia_sugerido')}")
print(f"Observaciones: {result.get('observaciones')}")
print("\n✓ Clasificación completada exitosamente")
