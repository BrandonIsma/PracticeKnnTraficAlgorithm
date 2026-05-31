import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
# Fluido (17 registros)
fluido_vel = np.random.randint(60, 81, 17)
fluido_veh = np.random.randint(10, 31, 17)

# Moderado (17 registros)
moderado_vel = np.random.randint(35, 61, 17)
moderado_veh = np.random.randint(30, 61, 17)

# Congestionado (16 registros)
cong_vel = np.random.randint(5, 36, 16)
cong_veh = np.random.randint(60, 101, 16)

velocidad = np.concatenate([
    fluido_vel,
    moderado_vel,
    cong_vel
])

vehiculos = np.concatenate([
    fluido_veh,
    moderado_veh,
    cong_veh
])

estado = (
    ["Fluido"] * 17 +
    ["Moderado"] * 17 +
    ["Congestionado"] * 16
)

df = pd.DataFrame({
    "Velocidad": velocidad,
    "Vehiculos": vehiculos,
    "Estado": estado
})
X = df[["Velocidad", "Vehiculos"]]
y = df["Estado"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=10,
    random_state=42,
    stratify=y
)

k = 3
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("RESULTADOS DEL MODELO KNN")

print(f"\nDatos de entrenamiento: {len(X_train)}")
print(f"Datos de prueba: {len(X_test)}")

print(f"\nPrecisión: {accuracy*100:.2f}%")
print(f"Error: {(1-accuracy)*100:.2f}%")

resultados = X_test.copy()

resultados["Real"] = y_test.values
resultados["Predicho"] = y_pred

resultados["Correcto"] = (
    resultados["Real"] == resultados["Predicho"]
)
print("CASOS DE PRUEBA")

print(resultados)

plt.figure(figsize=(10,7))
# ENTRENAMIENTO
colores = {
    "Fluido": "green",
    "Moderado": "orange",
    "Congestionado": "red"
}

for clase in y_train.unique():

    indices = y_train == clase

    plt.scatter(
        X_train.loc[indices, "Velocidad"],
        X_train.loc[indices, "Vehiculos"],
        c=colores[clase],
        s=80,
        alpha=0.7,
        label=f"Entrenamiento {clase}"
    )
# PRUEBA CORRECTA
correctos = resultados["Correcto"] == True

plt.scatter(
    resultados.loc[correctos, "Velocidad"],
    resultados.loc[correctos, "Vehiculos"],
    marker="o",
    facecolors="none",
    edgecolors="black",
    s=250,
    linewidths=2,
    label="Prueba correcta"
)
# PRUEBA INCORRECTA
incorrectos = resultados["Correcto"] == False

plt.scatter(
    resultados.loc[incorrectos, "Velocidad"],
    resultados.loc[incorrectos, "Vehiculos"],
    marker="x",
    c="black",
    s=250,
    linewidths=3,
    label="Prueba incorrecta"
)
plt.title(
    f"KNN para Clasificación de Tráfico\nPrecisión = {accuracy*100:.2f}%"
)

plt.xlabel("Velocidad promedio (km/h)")
plt.ylabel("Vehículos por minuto")

plt.grid(True)
plt.legend()

plt.show()
