"""
Visualizaciones interactivas con Plotly para el proyecto
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

class VisualizadorSeguridad:
    """Generador de visualizaciones para datos de seguridad"""
    
    @staticmethod
    def crear_grafico_fatalidades_por_provincia(dataframe: pd.DataFrame) -> go.Figure:
        """Gráfico de barras: fatalidades por provincia"""
        
        if 'admin1' not in dataframe.columns or 'fatalities' not in dataframe.columns:
            return go.Figure()
        
        # Agrupar datos
        por_provincia = dataframe.groupby('admin1')['fatalities'].sum().reset_index()
        por_provincia = por_provincia.sort_values('fatalities', ascending=False)
        
        # Crear gráfico
        fig = px.bar(
            por_provincia,
            x='admin1',
            y='fatalities',
            title='Total de Fatalidades por Provincia',
            labels={'admin1': 'Provincia', 'fatalities': 'Número de Víctimas'},
            color='fatalities',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Provincia",
            yaxis_title="Víctimas",
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def crear_grafico_tipos_evento(dataframe: pd.DataFrame) -> go.Figure:
        """Gráfico de barras: tipos de evento"""
        
        if 'event_type' not in dataframe.columns or 'fatalities' not in dataframe.columns:
            return go.Figure()
        
        # Agrupar datos
        por_tipo = dataframe.groupby('event_type')['fatalities'].sum().reset_index()
        por_tipo = por_tipo.sort_values('fatalities', ascending=False)
        
        # Crear gráfico
        fig = px.bar(
            por_tipo,
            x='event_type',
            y='fatalities',
            title='Fatalidades por Tipo de Evento',
            labels={'event_type': 'Tipo de Evento', 'fatalities': 'Víctimas'},
            color='fatalities',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Tipo de Evento",
            yaxis_title="Víctimas"
        )
        
        return fig
    
    @staticmethod
    def crear_mapa_eventos(dataframe: pd.DataFrame) -> go.Figure:
        """Mapa interactivo de eventos"""
        
        if 'latitude' not in dataframe.columns or 'longitude' not in dataframe.columns:
            return go.Figure()
        
        # Filtrar datos con coordenadas válidas
        df_mapa = dataframe.dropna(subset=['latitude', 'longitude'])
        
        if len(df_mapa) == 0:
            return go.Figure()
        
        # Crear mapa
        fig = px.scatter_mapbox(
            df_mapa,
            lat='latitude',
            lon='longitude',
            color='fatalities',
            size='fatalities',
            hover_name='event_type',
            hover_data=['event_date', 'admin1', 'admin2', 'fatalities'],
            zoom=5,
            title='Mapa de Eventos de Seguridad en Ecuador',
            color_continuous_scale='OrRd'
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        
        return fig
    
    @staticmethod
    def crear_grafico_evolucion_temporal(dataframe: pd.DataFrame) -> go.Figure:
        """Gráfico de evolución temporal de eventos"""
        
        if 'event_date' not in dataframe.columns or 'fatalities' not in dataframe.columns:
            return go.Figure()
        
        # Agrupar por mes
        dataframe['mes'] = dataframe['event_date'].dt.to_period('M').astype(str)
        
        evolucion = dataframe.groupby('mes').agg({
            'fatalities': 'sum',
            'event_date': 'count'
        }).reset_index()
        
        evolucion.columns = ['mes', 'total_victimas', 'total_eventos']
        
        # Crear gráfico con dos ejes Y
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Agregar trazos
        fig.add_trace(
            go.Scatter(
                x=evolucion['mes'],
                y=evolucion['total_eventos'],
                name="Número de Eventos",
                line=dict(color='blue', width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=evolucion['mes'],
                y=evolucion['total_victimas'],
                name="Víctimas Totales",
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        # Actualizar layout
        fig.update_layout(
            title="Evolución Temporal de Eventos y Víctimas",
            xaxis_title="Mes",
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Número de Eventos", secondary_y=False)
        fig.update_yaxes(title_text="Víctimas Totales", secondary_y=True)
        
        return fig
    
    @staticmethod
    def crear_grafico_comparacion_algoritmos(resultados: Dict[str, Any]) -> go.Figure:
        """Gráfico comparativo de algoritmos de ordenamiento"""
        
        if not resultados:
            return go.Figure()
        
        # Preparar datos para gráfico
        algoritmos = []
        tiempos = []
        memorias = []
        
        for algo, stats in resultados.items():
            if algo != 'analisis' and 'tiempo' in stats:
                algoritmos.append(algo.replace('_', ' ').title())
                tiempos.append(stats['tiempo'] * 1000)  # Convertir a ms
                memorias.append(stats['memoria_bytes'] / 1024)  # Convertir a KB
        
        # Crear gráfico de barras agrupadas
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Tiempo de Ejecución (ms)", "Uso de Memoria (KB)"),
            horizontal_spacing=0.2
        )
        
        # Gráfico de tiempos
        fig.add_trace(
            go.Bar(
                x=algoritmos,
                y=tiempos,
                name="Tiempo",
                marker_color='lightblue',
                text=[f'{t:.2f} ms' for t in tiempos],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Gráfico de memoria
        fig.add_trace(
            go.Bar(
                x=algoritmos,
                y=memorias,
                name="Memoria",
                marker_color='lightgreen',
                text=[f'{m:.1f} KB' for m in memorias],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparación de Algoritmos de Ordenamiento",
            showlegend=False,
            height=500
        )
        
        return fig
    
    @staticmethod
    def crear_visualizacion_arbol(estadisticas_arbol: Dict[str, Any]) -> go.Figure:
        """Visualización simple del árbol de búsqueda"""
        
        # Crear gráfico de indicadores (gauges)
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("Altura del Árbol", "Total de Nodos", 
                          "Total de Registros", "Comparaciones Promedio")
        )
        
        # Agregar indicadores
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=estadisticas_arbol.get('altura', 0),
                title={"text": "Altura"},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=estadisticas_arbol.get('total_nodos', 0),
                title={"text": "Nodos"},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=estadisticas_arbol.get('total_registros', 0),
                title={"text": "Registros"},
                domain={'row': 1, 'column': 0}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=round(estadisticas_arbol.get('comparaciones_promedio', 0), 2),
                title={"text": "Comp. Promedio"},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Estadísticas del Árbol de Búsqueda Binaria",
            height=400,
            template="plotly_white"
        )
        
        return fig