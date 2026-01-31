"""
Implementación de Árbol Binario de Búsqueda (BST) para datos de seguridad territorial
"""
from typing import Any, List, Optional, Tuple
import time

class NodoBST:
    """Nodo del Árbol Binario de Búsqueda"""
    
    def __init__(self, clave: float, dato: dict):
        self.clave = clave  # Valor para ordenar (ej: fatalidades, riesgo)
        self.datos = [dato]  # Lista de registros con misma clave
        self.izquierdo: Optional[NodoBST] = None
        self.derecho: Optional[NodoBST] = None
        self.altura = 1

class ArbolBusquedaBinaria:
    """Árbol Binario de Búsqueda implementado iterativamente"""
    
    def __init__(self):
        self.raiz: Optional[NodoBST] = None
        self.contador_comparaciones = 0
    
    def insertar(self, clave: float, dato: dict) -> None:
        """Inserta un nuevo nodo en el árbol de forma iterativa"""
        nuevo_nodo = NodoBST(clave, dato)
        
        if self.raiz is None:
            self.raiz = nuevo_nodo
            return
        
        actual = self.raiz
        padre = None
        
        while actual is not None:
            padre = actual
            self.contador_comparaciones += 1
            
            if clave < actual.clave:
                actual = actual.izquierdo
            elif clave > actual.clave:
                actual = actual.derecho
            else:  # Clave igual, agregar a lista de datos
                actual.datos.append(dato)
                return
        
        # Insertar en posición correcta
        if clave < padre.clave:
            padre.izquierdo = nuevo_nodo
        else:
            padre.derecho = nuevo_nodo
    
    def buscar(self, clave: float) -> List[dict]:
        """Busca todos los registros con una clave específica"""
        actual = self.raiz
        self.contador_comparaciones = 0
        
        while actual is not None:
            self.contador_comparaciones += 1
            
            if clave < actual.clave:
                actual = actual.izquierdo
            elif clave > actual.clave:
                actual = actual.derecho
            else:
                return actual.datos
        
        return []  # No encontrado
    
    def buscar_rango(self, clave_min: float, clave_max: float) -> List[dict]:
        """Busca registros dentro de un rango de claves"""
        resultados = []
        
        def _buscar_rango_rec(nodo: Optional[NodoBST]):
            if nodo is None:
                return
            
            if clave_min < nodo.clave:
                _buscar_rango_rec(nodo.izquierdo)
            
            if clave_min <= nodo.clave <= clave_max:
                resultados.extend(nodo.datos)
            
            if clave_max > nodo.clave:
                _buscar_rango_rec(nodo.derecho)
        
        _buscar_rango_rec(self.raiz)
        return resultados
    
    def recorrido_inorden(self) -> List[dict]:
        """Recorrido inorden (izquierdo, raíz, derecho)"""
        resultados = []
        pila = []
        actual = self.raiz
        
        while actual or pila:
            while actual:
                pila.append(actual)
                actual = actual.izquierdo
            
            actual = pila.pop()
            # Agregar todos los datos del nodo
            for dato in actual.datos:
                resultados.append({
                    'clave': actual.clave,
                    **dato
                })
            
            actual = actual.derecho
        
        return resultados
    
    def obtener_altura(self) -> int:
        """Calcula la altura del árbol"""
        def _altura(nodo: Optional[NodoBST]) -> int:
            if nodo is None:
                return 0
            return 1 + max(_altura(nodo.izquierdo), _altura(nodo.derecho))
        
        return _altura(self.raiz)
    
    def obtener_estadisticas(self) -> dict:
        """Obtiene estadísticas del árbol"""
        datos = self.recorrido_inorden()
        total_nodos = len(set(d['clave'] for d in datos))
        total_registros = len(datos)
        
        return {
            'altura': self.obtener_altura(),
            'total_nodos': total_nodos,
            'total_registros': total_registros,
            'comparaciones_promedio': self.contador_comparaciones / max(1, total_registros)
        }

# Árbol especializado para datos de seguridad
class ArbolSeguridadTerritorial(ArbolBusquedaBinaria):
    """BST especializado para búsquedas en datos de seguridad"""
    
    def buscar_por_fatalidades(self, min_fatal: int, max_fatal: int) -> List[dict]:
        """Busca eventos por rango de fatalidades"""
        return self.buscar_rango(min_fatal, max_fatal)
    
    def buscar_por_provincia(self, provincia: str) -> List[dict]:
        """Busca eventos por provincia (recorrido completo)"""
        resultados = []
        for dato in self.recorrido_inorden():
            if dato.get('admin1') == provincia:
                resultados.append(dato)
        return resultados
    
    def buscar_por_tipo_evento(self, tipo_evento: str) -> List[dict]:
        """Busca eventos por tipo"""
        resultados = []
        for dato in self.recorrido_inorden():
            if dato.get('event_type') == tipo_evento:
                resultados.append(dato)
        return resultados