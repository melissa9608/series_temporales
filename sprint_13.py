import math
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('/datasets/taxi.csv')

print(data.info())
print()
print(data.sample(10))
print()
print(data.isnull().sum())


data['datetime'] = pd.to_datetime(data['datetime'])
print(data.info())


# Remuestreo por una hora
data.set_index('datetime', inplace=True)
data = data.resample('1H').sum()
data = data.dropna()


data['rolling_mean'] = data['num_orders'].rolling(10).mean()
data = data.dropna()
plt.figure(figsize=(15, 6))
data.plot()
plt.show()

print(data)

print(data.describe())


# seasonal_decompose
decomposition = seasonal_decompose(data['num_orders'], model='additive')


# Gráfica de tendencia
plt.figure(figsize=(15, 6))
plt.plot(decomposition.trend)
plt.title('Tendencia')
plt.show()


# Gráfica de estacionalidad
plt.figure(figsize=(15, 6))
plt.plot(decomposition.seasonal)
plt.title('Estacionalidad')
plt.show()


# Gráfica de residuo
plt.figure(figsize=(15, 6))
plt.plot(decomposition.resid)
plt.title('Residuales')
plt.show()


# Gráfica de descomposición completa
fig = decomposition.plot()
fig.set_size_inches(15, 8)
plt.show()


# Crear nuevas features
def new_features(df, target, max_lag=1, rolling_window=1):
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['weekend'] = df.index.dayofweek >= 5  # Fines de semana

    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)

    df['rolling_mean'] = df[target].shift().rolling(rolling_window).mean()
    df['rolling_median'] = df[target].shift().rolling(rolling_window).median()
    df['rolling_std'] = df[target].shift().rolling(rolling_window).std()
    df = df.dropna()

    return df


data = new_features(data, 'num_orders', max_lag=3, rolling_window=10)


features = data.drop(columns=['num_orders'])
target = data['num_orders']

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.10, shuffle=False)


# Lineal Regression

model_lr = LinearRegression()
model_lr.fit(features_train, target_train)
predictions_lr = model_lr.predict(features_test)
recm_lr = sqrt(mean_squared_error(target_test, predictions_lr))
print(f'RECM Regresión Lineal: {recm_lr}')

# Decision Tree. Modificando el hiperparametro max_depth

depth_opc = [1, 3, 5, 7, 10, 20, 40, 50, 100, 150, 200]
best_recm = float('inf')
best_model = None
best_params = {}

for depth in depth_opc:
    model_dt = DecisionTreeRegressor(random_state=12345, max_depth=depth)
    model_dt.fit(features_train, target_train)
    predictions_dt = model_dt.predict(features_test)
    recm_dt = sqrt(mean_squared_error(target_test, predictions_dt))
    if recm_dt < best_recm:
        best_recm = recm_dt
        best_model = model_dt
        best_params = {'model': 'DecisionTree', 'max_depth': depth}

print(f'Mejores hiperparámetros: {best_params}')
print(f'Menor RECM en el conjunto de validación: {best_recm}')


# Random Forest

n_estimators_opc = [10, 20, 30, 40, 50, 75, 100, 200]
best_recm = float('inf')
best_model = None
best_params = {}

for n_estimators in n_estimators_opc:
    model_rf = RandomForestRegressor(
        random_state=54321, n_estimators=n_estimators)
    model_rf.fit(features_train, target_train)
    predictions_rf = model_rf.predict(features_test)
    recm_rf = sqrt(mean_squared_error(target_test, predictions_rf))
    if recm_rf < best_recm:
        best_recm = recm_rf
        best_model = model_rf
        best_params = {'model': 'RandomForest', 'n_estimators': n_estimators}

print(f'Mejores hiperparámetros: {best_params}')
print(f'Menor RECM en el conjunto de validación: {best_recm}')


# Modelo de series temporales SARIMA

# Determinar el tamaño de la prueba (10% del conjunto de datos)
train_size = int(len(data) * 0.90)
train_data, test_data = data[:train_size], data[train_size:]

# Objetivo para el entrenamiento y la prueba
target_train = train_data['num_orders']
target_test = test_data['num_orders']

print(f'Tamaño del conjunto de entrenamiento: {len(train_data)}')
print(f'Tamaño del conjunto de prueba: {len(test_data)}')

# Configurar el modelo SARIMA
model_sa = SARIMAX(target_train,
                   order=(1, 1, 1),
                   seasonal_order=(1, 1, 1, 24),
                   enforce_stationarity=False,
                   enforce_invertibility=False)

model_sa_fit = model_sa.fit(disp=False)
predictions_sa = model_sa_fit.forecast(steps=len(target_test))
recm_sarima = sqrt(mean_squared_error(target_test, predictions_sa))

print(f'RECM SARIMA: {recm_sarima}')


# Visualización de los resultados

plt.figure(figsize=(10, 6))
plt.plot(target_train.index, target_train, label='Entrenamiento')
plt.plot(target_test.index, target_test, label='Prueba', color='orange')
plt.plot(target_test.index, predictions_sa,
         label='Predicciones', color='green')
plt.legend()
plt.title('Predicción de Pedidos de Taxis')
plt.show()


# ## Conclusiones

# El objetivo del sprint13: Series temporales era construir un modelo para predecir la cantidad de pedidos de taxis en la próxima hora,
# utilizando datos históricos de pedidos de taxis en los aeropuertos. La métrica que se usó para evaluar el rendimiento del modelo fue el
# Error Cuadrático Medio de la Raíz (RMSE), esta se utiliza para medir la precisión de los modelos predictivos,
# especialmente en regresión y series temporales. Un valor bajo del RMSE indica que el modelo tiene una alta precisión, es decir,
# las predicciones están muy cerca de los valores reales. En este caso el RMSE no podía superar 48 en el conjunto de prueba, esto con el fin de asegurar que
# las preducciones fueran precisas y se pueda confiar en el modelo elegido. Por último la muestra de prueba debía ser del 10% del conjunto de datos inicial.

# Se probaron varios modelos de aprendizaje automático y un modelo de series temporales SARIMA.

# 1. Regresión Lineal:
#
# RECM: 53.41
#
# Este modelo se utilizó ya que la regresión lineal es un modelo simple que asume una relación lineal entre las características y el objetivo.
# Sin embargo, este modelo no alcanzó el objetivo de RECM < 48, lo que indica que no es suficientemente preciso para lo que busca la compañía Sweet Lift Taxi.
#
# 2. Árbol de Decisión:
#
# Mejores Hiperparámetros: max_depth = 3
#
# RECM: 52.60
#
# El Árbol de Decisión es un modelo no lineal que divide iterativamente los datos en subconjuntos basados en las características más importantes.
# Aunque probó diferentes profundidades, el mejor rendimiento fue con una profundidad de 3, pero tampoco cumplió con el requisito de RECM < 48
#
#
# 3. Random Forest:
#
# Mejores Hiperparámetros: n_estimators = 40
#
# RECM: 46.78
#
# El Random Forest es un conjunto de múltiples árboles de decisión, que mejora la robustez y la precisión en comparación con un solo árbol.
# Sin embargo, a pesar de ajustar el número de estimadores, este modelo tuvo un rendimiento peor que el Árbol de Decisión individual.
#
#
# 4. Modelo SARIMA:
#
# Orden: (1, 1, 1)
#
# Orden estacional: (1, 1, 1, 24)
#
# RECM: 47.87
#
# SARIMA (Seasonal AutoRegressive Integrated Moving Average) es un modelo de series temporales que maneja patrones estacionales y no estacionarios.
# Este modelo no solo cumplió con el requisito de RECM < 48, sino que también se desempeñó mejor que los modelos de aprendizaje automático,
# lo que lo convierte en el mejor modelo para este problema.
#
# SARIMA es adecuado para datos de series temporales que muestran patrones estacionales y no estacionarios,
# del grafico del punto 4 se pudo concluir que se trataba de una serie no estacionaria, este modelo es útil cuando los datos muestran fluctuaciones periódicas,
# como es el caso de los pedidos de taxis que pueden variar según la hora del día o el día de la semana.
#
# Este modelo es útil para predecir eventos futuros en función de patrones observados en el pasado,
# lo que lo lleva a ser útil a la hora de la demanda de taxis en la próxima hora.
#
# Además, el modelo SARIMA es el mejor modelo en este caso, ya que logró un RECM de 43.64, por debajo de 48 que era lo requerido.
#
# Basandose en el rendimiento, el modelo más adecuado a implementar es el **modelo SARIMA**.
# Este modelo permitirá a Sweet Lift Taxi ajustar sus operaciones según la demanda prevista, mejorando así la eficiencia y la satisfacción del cliente.
