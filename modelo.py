import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Cargar los datos simulados
df = pd.read_csv("costos_simulados.csv")

# Definir características (X) y etiqueta (y)
X = df[["vm_instances", "storage_gb", "network_gb"]]
y = df["costo_total"]

# Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Guardar el modelo entrenado en un archivo
joblib.dump(modelo, "modelo_costos.pkl")

print("✅ Modelo entrenado y guardado como 'modelo_costos.pkl'")

