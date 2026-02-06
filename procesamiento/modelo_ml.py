"""
Este módulo implementa:
1. Modelo de Clasificación: Predice nivel de riesgo (bajo/medio/alto)
2. Modelo de Regresión: Predice número de fatalidades
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

class ModeloClasificacion:
    """Modelo de clasificación para predecir nivel de riesgo"""
    
    def __init__(self):
        self.modelo = None
        self.encoder_provincia = LabelEncoder()
        self.encoder_tipo_evento = LabelEncoder()
        self.scaler = StandardScaler()
        
    def preparar_datos_clasificacion(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para clasificación.
        
        Variable objetivo: nivel_riesgo (bajo/medio/alto)
        Se crea basado en el número de fatalidades:
        - Bajo: 0 fatalidades
        - Medio: 1-2 fatalidades  
        - Alto: 3+ fatalidades
        """
        df = df.copy()
        
        # Verificar columnas requeridas
        columnas_requeridas = ['fatalities', 'admin1', 'event_type']
        for col in columnas_requeridas:
            if col not in df.columns:
                raise ValueError(f"Falta columna obligatoria: {col}")
        
        # Crear variable objetivo (nivel de riesgo)
        condiciones = [
            df['fatalities'] == 0,
            (df['fatalities'] >= 1) & (df['fatalities'] <= 2),
            df['fatalities'] >= 3
        ]
        valores = ['bajo', 'medio', 'alto']
        df['nivel_riesgo'] = np.select(condiciones, valores, default='bajo')
        
        print("Distribución de clases:")
        print(df['nivel_riesgo'].value_counts())
        
        # Preparar características
        df['admin1'] = df['admin1'].fillna('DESCONOCIDO')
        df['event_type'] = df['event_type'].fillna('DESCONOCIDO')
        
        # Codificar variables categóricas
        df['provincia_encoded'] = self.encoder_provincia.fit_transform(df['admin1'])
        df['tipo_evento_encoded'] = self.encoder_tipo_evento.fit_transform(df['event_type'])
        
        # Crear características
        features = ['provincia_encoded', 'tipo_evento_encoded']
        if 'valor_riesgo' in df.columns:
            features.append('valor_riesgo')
        
        X = df[features]
        y = df['nivel_riesgo']
        
        # Estandarizar si hay características numéricas
        if 'valor_riesgo' in features:
            X[['valor_riesgo']] = self.scaler.fit_transform(X[['valor_riesgo']])
        
        return X, y
    
    def entrenar_modelo_clasificacion(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena modelo de clasificación con métricas"""
        
        # Preparar datos
        X, y = self.preparar_datos_clasificacion(df)
        
        if len(X) < 10:
            return {"error": "Datos insuficientes para entrenamiento"}
        
        # DIVISIÓN TRAIN/TEST (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nDivisión Train/Test:")
        print(f"  - Entrenamiento: {len(X_train)} registros")
        print(f"  - Prueba: {len(X_test)} registros")
        
        # Entrenar modelo
        self.modelo = DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        self.modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred = self.modelo.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        matriz_conf = confusion_matrix(y_test, y_pred)
        reporte = classification_report(y_test, y_pred, output_dict=True)
        
        # Importancia de características
        importancia = dict(zip(X.columns, self.modelo.feature_importances_))
        
        return {
            "accuracy": accuracy,
            "matriz_confusion": matriz_conf.tolist(),
            "reporte_clasificacion": reporte,
            "tamano_entrenamiento": len(X_train),
            "tamano_prueba": len(X_test),
            "importancia_variables": importancia,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "clases": list(self.modelo.classes_)
        }
    
    def visualizar_resultados_clasificacion(self, resultados: Dict[str, Any]):
        """Visualiza resultados del modelo de clasificación SIN seaborn"""
        
        # Crear figura con subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. Matriz de confusión (usando matplotlib en lugar de seaborn)
        matriz = np.array(resultados['matriz_confusion'])
        clases = resultados['clases']
        
        # Mostrar matriz como imagen
        im = axes[0].imshow(matriz, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0].set_title('Matriz de Confusión')
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Real')
        
        # Añadir etiquetas
        tick_marks = np.arange(len(clases))
        axes[0].set_xticks(tick_marks)
        axes[0].set_xticklabels(clases, rotation=45)
        axes[0].set_yticks(tick_marks)
        axes[0].set_yticklabels(clases)
        
        # Añadir texto en cada celda
        thresh = matriz.max() / 2.
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                axes[0].text(j, i, format(matriz[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if matriz[i, j] > thresh else "black")
        
        # Añadir barra de color
        plt.colorbar(im, ax=axes[0])
        
        # 2. Importancia de variables
        importancia = resultados['importancia_variables']
        features = list(importancia.keys())
        values = list(importancia.values())
        
        # Crear barras horizontales
        y_pos = np.arange(len(features))
        bars = axes[1].barh(y_pos, values, color='steelblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(features)
        axes[1].set_xlabel('Importancia')
        axes[1].set_title('Importancia de Variables')
        
        # Añadir valores en barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig('resultados/clasificacion_resultados.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

class ModeloRegresion:
    """Modelo de regresión para predecir número de fatalidades"""
    
    def __init__(self):
        self.modelo = None
        self.encoder_provincia = LabelEncoder()
        self.encoder_tipo_evento = LabelEncoder()
        self.scaler = StandardScaler()
        
    def preparar_datos_regresion(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para regresión.
        
        Variable objetivo: fatalities (número de fatalidades)
        """
        df = df.copy()
        
        # Verificar columnas requeridas
        columnas_requeridas = ['fatalities', 'admin1', 'event_type']
        for col in columnas_requeridas:
            if col not in df.columns:
                raise ValueError(f"Falta columna obligatoria: {col}")
        
        # Preparar características
        df['admin1'] = df['admin1'].fillna('DESCONOCIDO')
        df['event_type'] = df['event_type'].fillna('DESCONOCIDO')
        
        # Codificar variables categóricas
        df['provincia_encoded'] = self.encoder_provincia.fit_transform(df['admin1'])
        df['tipo_evento_encoded'] = self.encoder_tipo_evento.fit_transform(df['event_type'])
        
        # Crear características
        features = ['provincia_encoded', 'tipo_evento_encoded']
        if 'valor_riesgo' in df.columns:
            features.append('valor_riesgo')
        
        X = df[features]
        y = df['fatalities']  # Variable objetivo
        
        print(f"\nEstadísticas de la variable objetivo (fatalities):")
        print(f"  - Mínimo: {y.min()}")
        print(f"  - Máximo: {y.max()}")
        print(f"  - Promedio: {y.mean():.2f}")
        print(f"  - Desviación: {y.std():.2f}")
        
        # Estandarizar características
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def entrenar_modelo_regresion(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena modelo de regresión con métricas"""
        
        # Preparar datos
        X, y = self.preparar_datos_regresion(df)
        
        if len(X) < 10:
            return {"error": "Datos insuficientes para entrenamiento"}
        
        # DIVISIÓN TRAIN/TEST (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nDivisión Train/Test:")
        print(f"  - Entrenamiento: {len(X_train)} registros")
        print(f"  - Prueba: {len(X_test)} registros")
        
        # Entrenar modelo
        self.modelo = DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        )
        self.modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred = self.modelo.predict(X_test)
        
        # Calcular métricas de regresión
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\nMétricas del Modelo de Regresión:")
        print(f"  - MSE (Error Cuadrático Medio): {mse:.4f}")
        print(f"  - MAE (Error Absoluto Medio): {mae:.4f}")
        print(f"  - R² (Coeficiente de Determinación): {r2:.4f}")
        print(f"  - RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "tamano_entrenamiento": len(X_train),
            "tamano_prueba": len(X_test),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "modelo": "DecisionTreeRegressor"
        }
    
    def visualizar_resultados_regresion(self, resultados: Dict[str, Any]):
        """Visualiza resultados del modelo de regresión"""
        
        # Crear figura con subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. Valores reales vs predichos
        y_test = resultados['y_test']
        y_pred = resultados['y_pred']
        
        axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue')
        axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                    'r--', lw=2, label='Línea perfecta')
        axes[0].set_xlabel('Valores Reales (fatalities)')
        axes[0].set_ylabel('Valores Predichos')
        axes[0].set_title('Valores Reales vs Predichos')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residuales
        residuales = np.array(y_test) - np.array(y_pred)
        axes[1].scatter(y_pred, residuales, alpha=0.5, color='coral')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Valores Predichos')
        axes[1].set_ylabel('Residuales')
        axes[1].set_title('Análisis de Residuales')
        axes[1].grid(True, alpha=0.3)
        
        # Añadir métricas en el gráfico
        axes[0].text(0.05, 0.95, f'R² = {resultados["r2"]:.3f}',
                    transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1].text(0.05, 0.95, f'RMSE = {resultados["rmse"]:.3f}',
                    transform=axes[1].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig('resultados/regresion_resultados.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Clase unificadora para mantener compatibilidad
class ModeloSeguridad:
    """Clase unificada para mantener compatibilidad con código existente"""
    
    def __init__(self):
        self.clasificador = ModeloClasificacion()
        self.regresor = ModeloRegresion()
        self.modelo_clasificacion = None
        self.modelo_regresion = None
    
    def entrenar_modelo(self, df: pd.DataFrame, tipo: str = 'ambos'):
        """
        Entrena modelos según tipo especificado
        
        Args:
            df: DataFrame con datos
            tipo: 'clasificacion', 'regresion', o 'ambos'
        """
        resultados = {}
        
        if tipo in ['clasificacion', 'ambos']:
            print("\n" + "="*60)
            print("ENTRENANDO MODELO DE CLASIFICACIÓN")
            print("="*60)
            
            resultados_clasificacion = self.clasificador.entrenar_modelo_clasificacion(df)
            
            if "error" not in resultados_clasificacion:
                print(f"\n✓ Precisión (accuracy): {resultados_clasificacion['accuracy'] * 100:.2f}%")
                print(f"✓ Tamaño entrenamiento: {resultados_clasificacion['tamano_entrenamiento']}")
                print(f"✓ Tamaño prueba: {resultados_clasificacion['tamano_prueba']}")
                
                # Visualizar resultados
                try:
                    self.clasificador.visualizar_resultados_clasificacion(resultados_clasificacion)
                except Exception as e:
                    print(f"Advertencia: No se pudieron generar visualizaciones de clasificación: {str(e)}")
            
            resultados['clasificacion'] = resultados_clasificacion
        
        if tipo in ['regresion', 'ambos']:
            print("\n" + "="*60)
            print("ENTRENANDO MODELO DE REGRESIÓN")
            print("="*60)
            
            resultados_regresion = self.regresor.entrenar_modelo_regresion(df)
            
            if "error" not in resultados_regresion:
                print(f"\n✓ Modelo entrenado exitosamente")
                print(f"✓ R² Score: {resultados_regresion['r2']:.4f}")
                
                # Visualizar resultados
                try:
                    self.regresor.visualizar_resultados_regresion(resultados_regresion)
                except Exception as e:
                    print(f"Advertencia: No se pudieron generar visualizaciones de regresión: {str(e)}")
            
            resultados['regresion'] = resultados_regresion
        
        return resultados
    
    def predecir(self, datos: Dict[str, Any]):
        """Método para compatibilidad - predice usando clasificación"""
        if self.clasificador.modelo is None:
            return {"error": "Modelo de clasificación no entrenado"}
        
        try:
            # Preparar datos para predicción
            df_pred = pd.DataFrame([datos])
            
            # Codificar con manejo de errores
            try:
                df_pred['provincia_encoded'] = self.clasificador.encoder_provincia.transform(
                    [datos.get('admin1', 'DESCONOCIDO')]
                )[0]
            except ValueError:
                # Si la provincia no está en el entrenamiento, usar valor por defecto
                df_pred['provincia_encoded'] = 0
            
            try:
                df_pred['tipo_evento_encoded'] = self.clasificador.encoder_tipo_evento.transform(
                    [datos.get('event_type', 'DESCONOCIDO')]
                )[0]
            except ValueError:
                df_pred['tipo_evento_encoded'] = 0
            
            # Crear características
            features = ['provincia_encoded', 'tipo_evento_encoded']
            
            # Manejar valor_riesgo si existe
            if 'valor_riesgo' in datos and datos['valor_riesgo'] is not None:
                df_pred['valor_riesgo'] = datos['valor_riesgo']
                features.append('valor_riesgo')
                
                # Estandarizar solo si existe
                if hasattr(self.clasificador.scaler, 'transform'):
                    valor_riesgo_df = pd.DataFrame([[datos['valor_riesgo']]], columns=['valor_riesgo'])
                    valor_riesgo_scaled = self.clasificador.scaler.transform(valor_riesgo_df)
                    df_pred['valor_riesgo'] = valor_riesgo_scaled[0][0]
            
            X = df_pred[features]
            
            # Predecir
            pred = self.clasificador.modelo.predict(X)[0]
            probs = self.clasificador.modelo.predict_proba(X)[0]
            
            return {
                "prediccion": pred,
                "probabilidades": dict(zip(self.clasificador.modelo.classes_, probs)),
                "exito": True
            }
        
        except Exception as e:
            return {
                "error": f"Error en predicción: {str(e)}",
                "exito": False
            }