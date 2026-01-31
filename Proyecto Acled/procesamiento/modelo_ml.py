# procesamiento/modelo_ml.py

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ModeloSeguridad:
    """Modelo de clasificación para predecir nivel de riesgo"""

    def __init__(self):
        self.modelo = None
        self.encoder_provincia = LabelEncoder()
        self.encoder_tipo_evento = LabelEncoder()
        self.scaler = StandardScaler()
        self.columnas_modelo = [
            'fatalidades_norm',
            'provincia_encoded',
            'tipo_evento_encoded'
        ]

    def preparar_datos(self, df: pd.DataFrame):
        df = df.copy()

        columnas_requeridas = ['fatalities', 'admin1', 'event_type']
        for col in columnas_requeridas:
            if col not in df.columns:
                raise ValueError(f"Falta columna obligatoria: {col}")

        condiciones = [
            df['fatalities'] == 0,
            (df['fatalities'] > 0) & (df['fatalities'] <= 2),
            df['fatalities'] > 2
        ]
        valores = ['bajo', 'medio', 'alto']
        df['nivel_riesgo'] = np.select(condiciones, valores)

        df['fatalidades_norm'] = df['fatalities']
        df['admin1'] = df['admin1'].fillna('DESCONOCIDO')
        df['event_type'] = df['event_type'].fillna('DESCONOCIDO')

        df['provincia_encoded'] = self.encoder_provincia.fit_transform(df['admin1'])
        df['tipo_evento_encoded'] = self.encoder_tipo_evento.fit_transform(df['event_type'])

        X = df[self.columnas_modelo]
        y = df['nivel_riesgo']

        X[['fatalidades_norm']] = self.scaler.fit_transform(X[['fatalidades_norm']])

        return X, y

    def entrenar_modelo(self, df: pd.DataFrame):
        X, y = self.preparar_datos(df)

        if len(X) < 10:
            return {"error": "Datos insuficientes para entrenamiento"}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.modelo.fit(X_train, y_train)

        y_pred = self.modelo.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
            "reporte": classification_report(y_test, y_pred, output_dict=True)
        }

    def predecir(self, datos: Dict[str, Any]):
        if self.modelo is None:
            return {"error": "Modelo no entrenado"}

        df = pd.DataFrame([datos])

        def encode_safe(encoder, valor):
            if valor in encoder.classes_:
                return encoder.transform([valor])[0]
            return 0

        df['provincia_encoded'] = encode_safe(
            self.encoder_provincia,
            datos.get('admin1', 'DESCONOCIDO')
        )
        df['tipo_evento_encoded'] = encode_safe(
            self.encoder_tipo_evento,
            datos.get('event_type', 'DESCONOCIDO')
        )

        df['fatalidades_norm'] = datos.get('fatalities', 0)
        df[['fatalidades_norm']] = self.scaler.transform(df[['fatalidades_norm']])

        X = df[self.columnas_modelo]

        pred = self.modelo.predict(X)[0]
        probs = self.modelo.predict_proba(X)[0]

        return {
            "prediccion": pred,
            "probabilidades": dict(zip(self.modelo.classes_, probs))
        }
