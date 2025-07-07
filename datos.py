import pandas as pd  
import random  
from datetime import datetime, timedelta  

def crear_datos_falsos():  
    servicios = ["VM-Instances", "Storage", "Database", "Networking"]  
    regiones = ["us-east-1", "europe-west1", "asia-southeast1"]  

    datos = []  
    for i in range(100):  
        dato = {  
            "fecha": datetime.now() - timedelta(days=random.randint(1, 30)),  
            "servicio": random.choice(servicios),  
            "region": random.choice(regiones),  
            "costo": round(random.uniform(10, 500), 2),  
            "horas_uso": random.randint(1, 744)  
        }  
        datos.append(dato)  

    return pd.DataFrame(datos)  

if __name__ == "__main__":  
    df = crear_datos_falsos()  
    print("✅ Datos creados:")  
    print(df.head())  
    df.to_csv("costos_simulados.csv", index=False)  
    print("✅ Archivo guardado: costos_simulados.csv")  