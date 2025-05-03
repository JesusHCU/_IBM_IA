# Importación de librerías necesarias
# numpy: para generar y manipular arrays numéricos
# pandas: para crear y manejar DataFrames
# matplotlib.pyplot: para visualización de datos
# LinearRegression: modelo de regresión lineal de scikit-learn

# Escribe aquí las importaciones necesarias:
# __________________________________________________________________

class VehicleRecord:
    def __init__(self, hours_used, wear_level):
        """
        Inicializa un registro de vehículo con las horas de uso y el nivel de desgaste.

        Args:
            hours_used (float): Horas que ha sido utilizado el vehículo.
            wear_level (float): Porcentaje de desgaste observado.
        """
        # Escribe aquí la inicialización de los atributos:
        # __________________________________________________________________

    def to_vector(self):
        """
        Convierte el atributo de horas de uso en una lista para que pueda ser utilizado por el modelo.

        Returns:
            list: Lista con un único valor, las horas de uso.
        """
        # Escribe aquí la conversión del dato a lista/vector:
        # __________________________________________________________________


class VehicleDataGenerator:
    def __init__(self, num_samples=100):
        """
        Inicializa el generador con la cantidad de muestras deseadas.

        Args:
            num_samples (int): Número de datos sintéticos a generar.
        """
        # Escribe aquí la inicialización de num_samples:
        # __________________________________________________________________

    def generate(self):
        """
        Genera datos sintéticos de uso y desgaste de vehículos.

        Returns:
            list: Lista de objetos VehicleRecord con datos generados.
        """
        # Genera horas aleatorias entre 50 y 500
        hours = np.random.uniform(50, 500, self.num_samples)
        # Calcula el desgaste con una fórmula lineal más algo de ruido aleatorio
        wear = 10 + 0.18 * hours + np.random.normal(0, 5, self.num_samples)
        # Asegura que el desgaste esté entre 0% y 100%
        wear = np.clip(wear, 0, 100)
        # Crea los objetos VehicleRecord
        # Escribe aquí la creación de la lista de datos:
        # __________________________________________________________________
        return data


class VehicleWearRegressor:
    def __init__(self):
        """
        Inicializa el modelo de regresión lineal.
        """
        # Escribe aquí la creación del modelo LinearRegression:
        # __________________________________________________________________

    def fit(self, records):
        """
        Entrena el modelo de regresión con los datos de entrada.

        Args:
            records (list): Lista de objetos VehicleRecord.
        """
        # Prepara los datos en arrays para entrenamiento
        X = np.array([r.to_vector() for r in records])
        y = np.array([r.wear_level for r in records])
        # Entrena el modelo
        # Escribe aquí el llamado a fit:
        # __________________________________________________________________

    def predict(self, hours):
        """
        Predice el nivel de desgaste basado en las horas de uso.

        Args:
            hours (float): Horas de uso del vehículo a predecir.

        Returns:
            float: Nivel estimado de desgaste.
        """
        # Escribe aquí el llamado a predict:
        # __________________________________________________________________

    def get_model(self):
        """
        Devuelve el modelo entrenado.

        Returns:
            LinearRegression: Modelo de regresión lineal entrenado.
        """
        # Escribe aquí la devolución del modelo:
        # __________________________________________________________________


class VehicleWearPredictionExample:
    def run(self):
        """
        Ejecuta todo el proceso de generación, entrenamiento, predicción y visualización.
        """
        # Genera los datos sintéticos
        generator = VehicleDataGenerator(100)
        records = generator.generate()

        # Crea y entrena el modelo
        regressor = VehicleWearRegressor()
        regressor.fit(records)

        # Realiza una predicción para 250 horas
        test_hours = 250
        prediction = regressor.predict(test_hours)
        print(f"⏱ Horas de uso estimadas: {test_hours}")
        print(f"⚙️ Nivel de desgaste estimado: {prediction:.2f}%")

        # Crea un DataFrame para graficar
        df = pd.DataFrame({
            "Horas de uso": [r.hours_used for r in records],
            "Nivel de desgaste": [r.wear_level for r in records]
        })

        # Visualiza los datos y la línea de regresión
        plt.figure(figsize=(8, 5))
        plt.scatter(df["Horas de uso"], df["Nivel de desgaste"], label="Datos Reales")
        x_line = np.linspace(40, 520, 100).reshape(-1, 1)
        y_line = regressor.get_model().predict(x_line)
        plt.plot(x_line, y_line, color='red', label="Línea de Regresión")
        plt.axvline(test_hours, color='green', linestyle='--', label="Predicción")

        # Personalización del gráfico
        plt.title('Predicción del Desgaste de Vehículos Militares')
        plt.xlabel('Horas de Uso')
        plt.ylabel('Nivel de Desgaste (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Ejecuta el ejemplo completo
if __name__ == "__main__":
# Escribe aquí la creación del objeto y la ejecución del método run():
# __________________________________________________________________
