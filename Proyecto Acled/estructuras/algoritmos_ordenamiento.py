"""
Análisis comparativo de algoritmos de ordenamiento
"""
import time
import random
from typing import List, Tuple, Dict
import sys

class AnalisisOrdenamiento:
    """Clase para comparar algoritmos de ordenamiento"""
    
    @staticmethod
    def counting_sort(arr: List[int]) -> Tuple[List[int], Dict]:
        """
        Counting Sort para enteros con rango pequeño
        Complejidad: O(n + k) donde k es el rango
        """
        inicio = time.time()
        
        if not arr:
            return [], {'tiempo': 0, 'memoria': 0}
        
        max_val = max(arr)
        min_val = min(arr)
        rango = max_val - min_val + 1
        
        # Arreglo de conteo
        count = [0] * rango
        
        # Contar ocurrencias
        for num in arr:
            count[num - min_val] += 1
        
        # Reconstruir arreglo ordenado
        resultado = []
        for i in range(rango):
            resultado.extend([i + min_val] * count[i])
        
        tiempo = time.time() - inicio
        memoria = sys.getsizeof(count) + sys.getsizeof(resultado)
        
        return resultado, {
            'tiempo': tiempo,
            'memoria_bytes': memoria,
            'estable': True,
            'complejidad': 'O(n + k)',
            'rango_k': rango
        }
    
    @staticmethod
    def radix_sort(arr: List[int]) -> Tuple[List[int], Dict]:
        """
        Radix Sort para enteros grandes
        Complejidad: O(d * (n + b)) donde d es número de dígitos
        """
        inicio = time.time()
        
        if not arr:
            return [], {'tiempo': 0, 'memoria': 0}
        
        # Encontrar el número máximo para saber cantidad de dígitos
        max_num = max(arr)
        
        # Hacer counting sort para cada dígito
        exp = 1
        while max_num // exp > 0:
            # Counting sort para el dígito actual
            n = len(arr)
            output = [0] * n
            count = [0] * 10
            
            # Contar ocurrencias del dígito actual
            for i in range(n):
                index = (arr[i] // exp) % 10
                count[index] += 1
            
            # Cambiar count[i] para que contenga posición actual
            for i in range(1, 10):
                count[i] += count[i - 1]
            
            # Construir output array
            i = n - 1
            while i >= 0:
                index = (arr[i] // exp) % 10
                output[count[index] - 1] = arr[i]
                count[index] -= 1
                i -= 1
            
            # Copiar output a arr
            for i in range(n):
                arr[i] = output[i]
            
            exp *= 10
        
        tiempo = time.time() - inicio
        memoria = sys.getsizeof(arr) + sys.getsizeof(output) + sys.getsizeof(count)
        
        return arr, {
            'tiempo': tiempo,
            'memoria_bytes': memoria,
            'estable': True,
            'complejidad': 'O(d * (n + b))',
            'digitos_d': len(str(max_num))
        }
    
    @staticmethod
    def bucket_sort(arr: List[float]) -> Tuple[List[float], Dict]:
        """
        Bucket Sort para datos uniformemente distribuidos
        Complejidad: O(n + k) en caso promedio
        """
        inicio = time.time()
        
        if not arr:
            return [], {'tiempo': 0, 'memoria': 0}
        
        # Determinar rango
        min_val = min(arr)
        max_val = max(arr)
        rango = max_val - min_val
        
        # Número de buckets (usar n buckets)
        n = len(arr)
        buckets = [[] for _ in range(n)]
        
        # Distribuir elementos en buckets
        for num in arr:
            if rango == 0:
                index = 0
            else:
                index = int((num - min_val) / rango * (n - 1))
            buckets[index].append(num)
        
        # Ordenar cada bucket (usando sorted por simplicidad)
        for i in range(n):
            buckets[i].sort()
        
        # Concatenar buckets
        resultado = []
        for bucket in buckets:
            resultado.extend(bucket)
        
        tiempo = time.time() - inicio
        memoria = sys.getsizeof(buckets) + sys.getsizeof(resultado)
        
        return resultado, {
            'tiempo': tiempo,
            'memoria_bytes': memoria,
            'estable': True,
            'complejidad': 'O(n + k) promedio',
            'num_buckets': n
        }
    
    @staticmethod
    def comparar_algoritmos(datos: List, tipo_datos: str = 'enteros') -> Dict:
        """
        Compara los tres algoritmos en tiempo y memoria
        """
        resultados = {}
        
        # Preparar datos según tipo
        if tipo_datos == 'enteros':
            datos_copy = datos.copy()
            datos_int = [int(d) for d in datos_copy if isinstance(d, (int, float))]
            
            if len(datos_int) < 10:
                return {"error": "Datos insuficientes para comparación"}
            
            # Counting Sort
            arr_counting = datos_int.copy()
            sorted_counting, stats_counting = AnalisisOrdenamiento.counting_sort(arr_counting)
            resultados['counting_sort'] = stats_counting
            
            # Radix Sort
            arr_radix = datos_int.copy()
            sorted_radix, stats_radix = AnalisisOrdenamiento.radix_sort(arr_radix)
            resultados['radix_sort'] = stats_radix
            
            # Bucket Sort (convertir a float para prueba)
            arr_bucket = [float(d) for d in datos_int]
            sorted_bucket, stats_bucket = AnalisisOrdenamiento.bucket_sort(arr_bucket)
            resultados['bucket_sort'] = stats_bucket
        
        elif tipo_datos == 'decimales':
            datos_float = [float(d) for d in datos if isinstance(d, (int, float))]
            
            # Solo Bucket Sort es apropiado para decimales
            arr_bucket = datos_float.copy()
            sorted_bucket, stats_bucket = AnalisisOrdenamiento.bucket_sort(arr_bucket)
            resultados['bucket_sort'] = stats_bucket
            
            # Para comparación, también probamos counting con escalado
            scaled = [int(d * 100) for d in datos_float]  # Escalar a enteros
            arr_counting = scaled.copy()
            sorted_counting, stats_counting = AnalisisOrdenamiento.counting_sort(arr_counting)
            resultados['counting_sort_escalado'] = stats_counting
        
        # Análisis comparativo
        if len(resultados) > 1:
            mejor_tiempo = min(resultados.items(), key=lambda x: x[1]['tiempo'])
            mejor_memoria = min(resultados.items(), key=lambda x: x[1]['memoria_bytes'])
            
            resultados['analisis'] = {
                'mejor_tiempo': {
                    'algoritmo': mejor_tiempo[0],
                    'tiempo': mejor_tiempo[1]['tiempo']
                },
                'mejor_memoria': {
                    'algoritmo': mejor_memoria[0],
                    'memoria_bytes': mejor_memoria[1]['memoria_bytes']
                },
                'recomendacion': AnalisisOrdenamiento._generar_recomendacion(resultados, tipo_datos)
            }
        
        return resultados
    
    @staticmethod
    def _generar_recomendacion(resultados: Dict, tipo_datos: str) -> str:
        """Genera recomendación basada en resultados"""
        if tipo_datos == 'enteros':
            if 'counting_sort' in resultados and resultados['counting_sort']['rango_k'] < 1000:
                return "Counting Sort es mejor para enteros con rango pequeño"
            elif 'radix_sort' in resultados:
                return "Radix Sort es eficiente para enteros grandes con muchos dígitos"
        elif tipo_datos == 'decimales':
            return "Bucket Sort es ideal para datos decimales uniformemente distribuidos"
        
        return "Elija según el tamaño y tipo de datos específicos"