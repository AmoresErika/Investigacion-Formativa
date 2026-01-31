"""
Excepciones personalizadas para el proyecto de seguridad territorial
"""

class ErrorProyectoSeguridad(Exception):
    """Excepción base personalizada para el proyecto"""
    pass

class ErrorArchivo(ErrorProyectoSeguridad):
    """Excepción para errores relacionados con archivos"""
    pass

class ErrorColumnasFaltantes(ErrorProyectoSeguridad):
    """Excepción cuando faltan columnas necesarias"""
    pass

class ErrorModeloNoEntrenado(ErrorProyectoSeguridad):
    """Excepción cuando el modelo no está entrenado"""
    pass

class ErrorDatosInvalidos(ErrorProyectoSeguridad):
    """Excepción para datos inválidos o corruptos"""
    pass

class ErrorConfiguracion(ErrorProyectoSeguridad):
    """Excepción para errores de configuración"""
    pass
