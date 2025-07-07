# Crear una copia de seguridad
cp main.py main.py.backup

# Crear el nuevo main.py mejorado
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI(title="🏗️ ML Cost Optimizer")

# Variables globales
modelo = None
df = None

def entrenar_modelo_inline():
    """Entrenar modelo inline si no existe o no se puede cargar"""
    global modelo
    try:
        # Cargar o generar datos
        if not os.path.exists("costos_simulados.csv"):
            print("🔄 Generando datos simulados...")
            import random
            data = {
                "vm_instances": [random.randint(1, 20) for _ in range(100)],
                "storage_gb": [random.randint(100, 1000) for _ in range(100)],
                "network_gb": [round(random.uniform(0.1, 50.0), 2) for _ in range(100)],
                "costo_total": [round(random.uniform(10.0, 200.0), 2) for _ in range(100)]
            }
            df_temp = pd.DataFrame(data)
            df_temp.to_csv("costos_simulados.csv", index=False)
            print("✅ Datos generados")
        
        # Cargar datos
        df_temp = pd.read_csv("costos_simulados.csv")
        
        # Entrenar modelo
        from sklearn.linear_model import LinearRegression
        X = df_temp[["vm_instances", "storage_gb", "network_gb"]]
        y = df_temp["costo_total"]
        
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Guardar modelo
        joblib.dump(modelo, "modelo_costos.pkl")
        print("✅ Modelo entrenado y guardado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error entrenando modelo: {e}")
        return False

def inicializar_app():
    """Inicializar modelo y datos"""
    global modelo, df
    
    # Intentar cargar modelo existente
    try:
        modelo = joblib.load("modelo_costos.pkl")
        print("✅ Modelo cargado desde archivo")
    except Exception as e:
        print(f"⚠️ No se pudo cargar modelo existente: {e}")
        print("🔄 Entrenando nuevo modelo...")
        if not entrenar_modelo_inline():
            print("❌ Error crítico: No se pudo entrenar modelo")
            modelo = None
    
    # Cargar datos
    try:
        df = pd.read_csv("costos_simulados.csv")
        print("✅ Datos cargados correctamente")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        df = None

# Inicializar al arrancar la app
inicializar_app()

@app.get("/")
def inicio():
    return {
        "mensaje": "🚀 ML Cost Optimizer API",
        "version": "1.0",
        "endpoints": ["/", "/salud", "/total-costos", "/predecir-costo"]
    }

@app.get("/salud")
def check_salud():
    return {
        "status": "OK",
        "modelo_cargado": modelo is not None,
        "datos_cargados": df is not None
    }

@app.get("/total-costos")
def obtener_total_costos():
    if df is None:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    return {
        "total_costos": round(df["costo_total"].sum(), 2),
        "promedio": round(df["costo_total"].mean(), 2),
        "maximo": round(df["costo_total"].max(), 2),
        "minimo": round(df["costo_total"].min(), 2),
        "total_registros": len(df)
    }

@app.get("/predecir-costo")
def predecir_costo(vm_instances: int, storage_gb: int, network_gb: float):
    if modelo is None:
        # Intentar entrenar modelo en tiempo real
        print("🔄 Intentando entrenar modelo en tiempo real...")
        if entrenar_modelo_inline():
            print("✅ Modelo entrenado exitosamente")
        else:
            raise HTTPException(status_code=500, detail="Modelo no disponible y no se pudo entrenar")
    
    try:
        # Crear DataFrame con los parámetros de entrada
        entrada = pd.DataFrame({
            "vm_instances": [vm_instances],
            "storage_gb": [storage_gb], 
            "network_gb": [network_gb]
        })
        
        # Hacer predicción
        prediccion = modelo.predict(entrada)[0]
        
        return {
            "prediccion_costo": round(prediccion, 2),
            "parametros": {
                "vm_instances": vm_instances,
                "storage_gb": storage_gb,
                "network_gb": network_gb
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
EOF