# app.py

from flask import Flask, request, jsonify, render_template
from procesamiento.limpieza_datos import LimpiezaDatos
from procesamiento.modelo_ml import ModeloSeguridad

import os
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Estado global
datos_actuales = None
modelo_actual = ModeloSeguridad()

# Decorador para persistir estado (opcional)
def persistir_estado(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        resultado = func(*args, **kwargs)
        return resultado
    return wrapper

# ==================== ENDPOINTS ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cargar_datos', methods=['POST'])
@persistir_estado
def cargar_datos():
    global datos_actuales
    if 'archivo' not in request.files:
        return jsonify({'error': 'No se envió archivo'}), 400

    archivo = request.files['archivo']
    ruta = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
    archivo.save(ruta)

    limpiador = LimpiezaDatos(ruta)
    datos_actuales = limpiador.ejecutar_pipeline()
    
    # Guardar CSV limpio automáticamente
    limpio_path = os.path.join(app.config['UPLOAD_FOLDER'], "ACLED_Ecuador_limpio.csv")
    limpiador.guardar_datos_limpios(limpio_path)

    return jsonify({
        'mensaje': 'Datos cargados y limpios',
        'registros': len(datos_actuales),
        'csv_limpio': limpio_path
    })

@app.route('/entrenar_modelo', methods=['POST'])
@persistir_estado
def entrenar_modelo():
    if datos_actuales is None:
        return jsonify({'error': 'Primero cargue datos'}), 400

    resultados = modelo_actual.entrenar_modelo(datos_actuales)
    return jsonify({
        'mensaje': 'Modelo entrenado correctamente',
        'resultados': resultados
    })

@app.route('/predecir', methods=['POST'])
def predecir():
    if modelo_actual.modelo is None:
        return jsonify({'error': 'Modelo no entrenado'}), 400

    datos = request.json
    resultado = modelo_actual.predecir(datos)
    return jsonify(resultado)

# ==================== MAIN ====================
if __name__ == '__main__':
    print("🚀 App de Seguridad Territorial corriendo en http://127.0.0.1:5000")
    app.run(debug=True)


