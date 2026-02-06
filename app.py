from flask import Flask, render_template
import pandas as pd
import folium
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Entrenar modelo al inicio
df = pd.read_csv("datos/ACLED_Ecuador.csv")


# Variables por ciudad
ciudades = df.groupby("location").agg({
    "event_type": lambda x: x.mode()[0],   # tipo de evento dominante
    "fatalities": "sum",                   # total fatalidades
    "event_date": "count"                  # número de eventos
}).reset_index()

# Crear columna de riesgo (ejemplo simple)
def asignar_riesgo(row):
    if row["fatalities"] > 3 or row["event_date"] > 50:
        return "ALTO"
    elif row["fatalities"] > 1 or row["event_date"] > 20:
        return "MEDIO"
    else:
        return "BAJO"

ciudades["riesgo"] = ciudades.apply(asignar_riesgo, axis=1)

# Preparar datos para ML
X = ciudades[["fatalities", "event_date"]]
y = ciudades["riesgo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template("index.html")

import os

@app.route('/mapa')
def mapa():
    # Centrar el mapa en Ecuador
    mapa = folium.Map(location=[-1.8312, -78.1834], zoom_start=7)

    # Recorrer los eventos del dataset y añadir marcadores
    for _, row in df.iterrows():
        # Ejemplo de predicción con tu modelo
        riesgo_predicho = modelo.predict([[row["fatalities"], 1]])[0]

        # Color según riesgo
        if riesgo_predicho == "BAJO":
            color = "green"
        elif riesgo_predicho == "MEDIO":
            color = "orange"
        else:
            color = "red"

        # Marcador con popup y tooltip
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"<b>{row['location']}</b><br>"
                  f"Eventos: {row['event_type']}<br>"
                  f"Fatalidades: {row['fatalities']}<br>"
                  f"Riesgo predicho: {riesgo_predicho}",
            tooltip=row['location']  # aparece al pasar el mouse
        ).add_to(mapa)

    # Guardar el mapa en templates
    ruta_mapa = os.path.join(app.root_path, "templates", "mapa.html")
    mapa.save(ruta_mapa)

    return render_template("mapa.html")


if __name__ == '__main__':
    app.run(debug=True)
