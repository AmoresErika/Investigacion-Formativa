import pandas as pd


class LimpiezaDatos:
    """Clase para limpieza y preprocesamiento de datos ACLED"""

    def __init__(self, archivo_csv: str):
        self.archivo = archivo_csv
        self.dataframe = None
        self.estadisticas = {}

    def cargar_datos(self) -> pd.DataFrame:
        """Carga el dataset desde CSV"""
        self.dataframe = pd.read_csv(self.archivo, encoding="utf-8")
        print(
            f"Datos cargados: {len(self.dataframe)} registros, "
            f"{len(self.dataframe.columns)} columnas"
        )
        return self.dataframe.copy()

    def filtrar_ecuador(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra datos solo para Ecuador"""
        if "country" in df.columns:
            df = df[df["country"].str.upper() == "ECUADOR"].copy()
            print(f"Registros de Ecuador: {len(df)}")
        return df.copy()

    def seleccionar_columnas(
        self, df: pd.DataFrame, columnas_relevantes: list | None = None
    ) -> pd.DataFrame:
        """Selecciona columnas relevantes"""
        if columnas_relevantes is None:
            columnas_relevantes = [
                "event_date",
                "year",
                "event_type",
                "sub_event_type",
                "actor1",
                "actor2",
                "admin1",
                "admin2",
                "admin3",
                "location",
                "latitude",
                "longitude",
                "fatalities",
                "notes",
                "source",
                "timestamp",
            ]

        columnas_existentes = [
            col for col in columnas_relevantes if col in df.columns
        ]
        return df[columnas_existentes].copy()

    def manejar_valores_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manejo de valores nulos"""
        if "fatalities" in df.columns:
            df = df.dropna(subset=["fatalities"])

        if "latitude" in df.columns and "longitude" in df.columns:
            df = df[df["latitude"].notna() & df["longitude"].notna()]

        for col in ["admin1", "admin2", "event_type", "actor1"]:
            if col in df.columns:
                df[col] = df[col].fillna("DESCONOCIDO")

        return df.copy()

    def convertir_tipos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conversión de tipos de datos"""
        if "fatalities" in df.columns:
            df["fatalities"] = (
                pd.to_numeric(df["fatalities"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(
                df["event_date"], errors="coerce"
            )

        if "year" in df.columns:
            df["year"] = (
                pd.to_numeric(df["year"], errors="coerce")
                .fillna(2023)
                .astype(int)
            )

        return df.copy()

    def eliminar_outliers(
        self,
        df: pd.DataFrame,
        columna: str = "fatalities",
        umbral_superior: float = 0.995,
    ) -> pd.DataFrame:
        """Elimina outliers por cuantiles"""
        if columna in df.columns:
            q_high = df[columna].quantile(umbral_superior)
            df = df[df[columna] <= q_high]

        return df.copy()

    def crear_variable_riesgo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea una variable de riesgo normalizada (0–10)"""
        if "fatalities" in df.columns:
            max_val = df["fatalities"].max()
            if max_val > 0:
                df["valor_riesgo"] = (df["fatalities"] / max_val) * 10
            else:
                df["valor_riesgo"] = 0

        return df.copy()

    def ejecutar_pipeline(
        self, archivo_salida: str = "ACLED_Ecuador_limpio.csv"
    ) -> pd.DataFrame:
        """Ejecuta el pipeline completo"""
        df = self.cargar_datos()
        df = self.filtrar_ecuador(df)
        df = self.seleccionar_columnas(df)
        df = self.manejar_valores_nulos(df)
        df = self.convertir_tipos(df)
        df = self.eliminar_outliers(df)
        df = self.crear_variable_riesgo(df)

        # Guardar estado final
        self.dataframe = df

        # Guardar CSV
        df.to_csv(archivo_salida, index=False, encoding="utf-8")
        print(f"Pipeline completado: {len(df)} registros listos para ML")
        print(f"Datos limpios guardados en: {archivo_salida}")

        return df

    def obtener_estadisticas(self) -> dict:
        """Genera estadísticas descriptivas del dataset limpio"""
        if self.dataframe is None:
            raise ValueError("Primero debes ejecutar el pipeline")

        df = self.dataframe

        estadisticas = {
            "registros_totales": len(df),
            "fatalidades": {
                "total": int(df["fatalities"].sum())
                if "fatalities" in df.columns
                else 0,
                "promedio": float(df["fatalities"].mean())
                if "fatalities" in df.columns
                else 0,
                "maximo": int(df["fatalities"].max())
                if "fatalities" in df.columns
                else 0,
            },
            "eventos_por_provincia": (
                df["admin1"].value_counts().to_dict()
                if "admin1" in df.columns
                else {}
            ),
        }

        self.estadisticas = estadisticas
        return estadisticas
