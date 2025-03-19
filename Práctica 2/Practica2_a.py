#%% [markdown]
## Práctica 2-a
#---
# Esta práctica ha sido llevada a cabo en solitario por Jaime Alonso Fernández (2024/2025) para la asignatura optativa 
# "Aprendizaje Automático y Big Data" en el grado de "Ingeniería del Software - plan 2019" cursado en la 
# Universidad Complutense de Madrid (UCM).
#
#
#---
### Importamos las librerías
#%%
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt

#%% [markdown]
#---
# Parte A: Regresión Lineal
# Para esta parte comenzaremos definiendo una función para obtener de manera fácil y consistente los valores 
# temporales x -> [1...60] y la frecuencia cardíaca asociada al mismo. Para que la tendencia sea creciente vamos a 
# multiplicar sumar a la frecuencia obtenida aleatoriamente un 70% del tiempo actual.
#%%
# Método para generar los datos de la frecuencia cardíaca
def generar_datos_FC():
    # np.arange utiliza los datos [start, stop). Por ello y para tener 60 valores lo detenemos en 61
    x = np.arange(1,61)
    # Fórmula dada en la transparencia de la pŕactica
    y = [0.7*i + 60 + np.random.uniform(-5.9, 5.9) for i in x]
    return x, y

#%% [markdown]
# Definida la función, vamos a usarla para generar datos aleatorios. Estos datos serán utilizados 
#%%
# Obtenemos los datos de x y de y
x, y = generar_datos_FC()

# Como necesitamos que x esté en formato de listas anidadas, lo transformamos
x_transformed = [[x_i] for x_i in x]

# Ahora con estos datos podemos calcular la regresión lineal
lin_reg = lm.LinearRegression()
# Le pasamos los datos de x y de y
lin_reg.fit(x_transformed, y)

# Mostramos por pantalla la información:
print(f"Coeficiente de la regresión lineal: {lin_reg.coef_}")
print(f"Intercepto (independiente) de la regresión lineal: {lin_reg.intercept_}")

# Definimos la función acorde a la documentación de sklearn
func_y = lambda x: lin_reg.intercept_ + lin_reg.coef_ * x

# Mostramos los resultados en una gráfica
plt.figure()
# Mostramos la línea usando la regresión lineal
plt.plot(x, func_y(x))
# Mostramos los puntos generados aleatoriamente
plt.scatter(x, y)
plt.show()

# Obtenemos la precisión del modelo
precision = lin_reg.score(x_transformed, y)*100
print(f"La precisión del modelo es de {precision}%.") 

#%% [markdown]
# A la vista de los resultados y con una precisión de en torno al 93%, podemos concluir que el modelo de regresión 
# lineal es bastate preciso cuando tratamos un set de datos de características similares a este, y aunque haya varios 
# puntos fuera de la linea en si, podríamos hacer una estimación bastante precisa de valores sin datos. 
