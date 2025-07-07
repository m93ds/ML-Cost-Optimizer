from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI(title="🏗️ ML Cost Optimizer")

# Cargar modelo al iniciar la app
try:
    modelo = joblib.load("modelo_costos.pkl")
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    modelo = None

# Cargar datos
try:
    df = pd.read_csv("costos_simulados.csv")
    print("✅ Datos cargados correctamente")
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    df = None

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
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
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