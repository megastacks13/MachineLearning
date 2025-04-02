#%% [markdown]
### Práctica 1
# Esta práctica ha sido llevada a cabo en solitario por Jaime Alonso Fernández (2024/2025) para la asignatura optativa 
# "Aprendizaje Automático y Big Data" en el grado de "Ingeniería del Software - plan 2019" cursado en la 
# Universidad Complutense de Madrid (UCM).
#
#
### Importamos las librerías necesarias
#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from surprise import model_selection as ms, Dataset, Reader, prediction_algorithms as pa, accuracy as acu
from sklearn import metrics as met

#%% [markdown]
## *APARTADO A*
### Parte 1: Lectura de datos y adecuación al contexto
# Para esta primera parte, comenzamos usando la librería 'pandas' para leer y eliminar todos los archivos 
# duplicados o con información vacía. Todo este proceso se lleva a cabo para trabajar solo con información nueva
# y precisa en el futuro.

#%%

# Importamos los datos del archivo
titanic_data = pd.read_csv('titanic.csv')

# Vamos a trabajar con otra variable cuando modifiquemos los datos
# Eliminamos las filas duplicadas
titanic_data_cleaned = titanic_data.drop_duplicates()
# Eliminamos también aquellas filas con información vacía
titanic_data_cleaned = titanic_data_cleaned.dropna()

# Reiniciamos los índicies de esta información "limpia"
titanic_data_cleaned = titanic_data_cleaned.reset_index(drop=True)

# Si ahora comparamos las longitudes de titanic_data_cleaned y de titanic_data, sabremos el número de filas eliminadas
print(len(titanic_data)-len(titanic_data_cleaned)) # $708

#%% [markdown]
# Como podemos ver en el resultado de la ejecución, se han eliminado un total de 708 filas. Cabe mencionar que para el 
# caso específico de este dataset, no había ningún elemento duplicado.
#
# Ahora con esta información ya "limpia" podemos proceder a trabajar con los datos sin que salgan comparativas raras
# pues no todos los valores son reales y válidos.
#
#
#---
### Parte 2: Atributos Redundantes
# Antes de pasar a comparar datos, es importante entender que podemos esperarnos de los mismos y tratar de hacer 
# una computación eficiente con ellos. Para eso, tenemos que valorar que columnas pueden contener información 
# relevante y sobre todo, cuáles no, de manera que podamos descartar estas últimas.
#
# En este caso en particular, yo he determinado que las siguientes columnas como redudantes:
# - PassengerId: No contiene información importante para el modelo.
# - Name: No considero relevante sacar la tasa de supervivencia de los "Juanes" o las "Raqueles", aunque socialmente pudiera estar ligado a la clase de los mismos.
# - Ticket: La información que pudiera ser relevante está asociada a la clase, las cuales van aparte.
# - Cabin: Entendiendo que la gente no tenía porque estar en su cabina en el momento del impacto, esta información es inútil.
# 
#%%
# Guardo los valores de los atributos redundantes en una lista para poder omitirlos
atributos_redudantes = ["PassengerId", "Name", "Ticket", "Cabin"]
# Eliminamos los atributos
titanic_data_cleaned = titanic_data_cleaned.drop(atributos_redudantes, axis=1)
# Y los mostramos en el notebook
titanic_data_cleaned

#%% [markdown]
#
#---
### Parte 7: Numerizando los atributos categóricos
# *DISCLAIMER: Esta celda requiere de la ejecución de la parte 1 para poder funcionar*
#
# Antes de nada aclarar que debido a la naturaleza del ejercicio 3, dónde se necesita calcular la matriz de correlación, 
# he decidido adelantar este apartado ya que es necesario que todas las variables categóricas estén "numerizadas" para 
# calcular su correlación. 
#
# En este caso vamos a "numerizar" las variables mediante dos métodos diferentes: get_dummies y LabelEncoder; y 
# compararlos para decidir cual considero mejor para un entrenamiento de IA.
#%%
# Definimos las variables categóricas:
categorical = ["Sex", "Embarked", "Survived","Pclass"]
categorical_textual = ["Sex", "Embarked"]

label_encoder = LabelEncoder()

# Declaramos una variable donde almacenar los resultados del LabelEncoder copiando si referencia "titanic_data_cleaned"
titanic_label_encoded = titanic_data_cleaned.copy(deep=True)

# Por cada columna categorica, la transformamos a numérica y la guardamos en la variable anterior
for c in categorical:
    titanic_label_encoded[c] = label_encoder.fit_transform(titanic_data_cleaned[c])

# Utilizamos pandas para hacer un símil del OneHotEncoder
titanic_dummified = pd.get_dummies(titanic_data_cleaned, categorical_textual)

#%% [markdown]
# Mostramos los datos en el notebook del OneHotEncoder o get_dummy
#%%
titanic_dummified
#%% [markdown]
# Mostramos los datos en el notebook del LabelEncoder
#%%
titanic_label_encoded
#%% [markdown]
# Finalmente podemos observar lo siguiente:
#
# Por una parte el OneHotEncoder crea una nueva columna booleana por cada elemento que conforma esa categoría,
# alargando la tabla y sobrando siempre una de las columnas (Puesto que se puede inferir al haber solo un "True" por fila).
#
# Por otra parte, el LabelEncoder decide asignar un valor entero para cada elemento que conforma esa categoría,
# manteniendo la longitud original de la tabla y sin columnas redundantes.
#
# En resumen y en lo personal, considero que es más limpia y correcta la opción que te ofrece el LabelEncoder, pues 
# en términos de espacio es más eficiente y por ello es la opción que usaré de cara a esta práctica.
#
#
#---
### Parte 3: Relaciones entre atributos
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1 y 7 para poder funcionar*
#
# En este apartado vamos a analizar que variables comparten un vínculo más fuerte entre si y que variables darán 
# son más solitarias. Esto permite entender la relación entre las diferentes variables, capacitandonos para entrenar 
# un modelo con información más precisa. De todas formas esta información tambien puede contener sesgos que transciendan 
# la vericidad de los datos por lo que habría que actuar de una manera cauta ante esta información. 
#
# El rango de valores susceptibles a ser tomados es de [1, -1]. Los valores más próximos a 0 serán aquellos que menos tengan 
# que ver mientras que los valores cuyo valor absoluto se aproxime más a 1 (los maś cercanos a los extremos), serán más similares.
#%%
# Sacamos la matriz de correlación:
titanic_corr = titanic_label_encoded.corr()
titanic_corr

#%% [markdown]
# Mostramos la gráfica
#%%
sns.pairplot(titanic_label_encoded)
plt.show()

#%% [markdown]
# Como podemos observar, especialmente en la variable de correlación, la mayoría de las variables no se afectan en 
# gran medida entre si. Sin embargo podemos destacar las siguientes que si lo hacen:
# 
# Relaciones relevantes (>0,3)
# - Sexo y Supervivencia (0.532418): Atendiendo al protocolo tomado dónde las mujeres y los niños desembarcaron 
# primero, tiene sentido que se de esta situacion. Quiero mencionar que en el DataSet se puede observar que no había 
# una gran población infantil ni mayor, por lo que limita la capacidad de la relación de edad y supervivencia.
#
# - Tarifa y Congéneres (0.389740): Es posible que existieran ciertos descuentos a la hora de comprar los billetes 
# en familia. De todas formas esto no se puede llegar a probar y la correlación no es tan alta como para afirmarlo 
# rotundamente. Otra opción es que hubieran promociones o concursos realizados por terceros que dieran esta opción. 
# De todas formas no se puede probar.
#
# - Clase y tarifa (0.315235): Las clases más bajas tendían a ser más baratas mientras que las más altas contaban 
# con dos opciones: Que fueras un invitado, por lo que pagarías menos; o que pagases más que las clases más bajas. 
# Personalmente sospecho que esto es lo que causa una correlación baja entre estos elementos, o que al menos esta
# correlación sea más baja de lo que cabría esperar.
# 
# - Edad y PClass (0.306514): Es lógico que hasta cierto punto la edad determine el poder adquisitivo de cada uno 
# y mientras que familias con infantes es más probable que optasen por un billete de clase inferior (y como los 
# infantes contribuyen a las estadisticas, estas bajaran), gente que fuera sola o solo con su pareja se podrían 
# permitir mejores billetes, y para ir solo has de tener cierta edad. 
#
#
#---
### Parte 4: Métricas de las variables numéricas
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1 y 7 para poder funcionar*
#
# En este apartado procederemos a calcular las estadísticas de las variables numéricas de nuestro DataSet. Para ello 
# haremos uso de la función *pandas.describe* la cual nos devuelve todas las estdísticas para cada una de las variables. Este 
# resultado será mostrado en la pantalla y guardado en un diccionario que asocie cada resultado a su variable, permitiendo 
# un fácil acceso a la información en caso de que se necesite a futuro.
#%%

# Definimos las variables numéricas:
numerical = ["Age", "SibSp", "Parch", "Fare"]
# Definimos el diccionario para los resultados
statistics = {}

# Aplicamos el comando describe() de pandas sobre todos los valores numericos
for variable in numerical:
    print(f"{variable}: ")
    description = titanic_data_cleaned[variable].describe()
    print(description)
    print()
    statistics[variable] = description

#%% [markdown]
#
#---
### Parte 5: Métricas de las variables categóricas
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1, 7 y 4 para poder funcionar*
#
# En este apartado procederemos a calcular las estadísticas de las variables categoricas de nuestro DataSet. Para ello 
# haremos uso de la función *pandas.describe* la cual nos devuelve todas las estdísticas para cada una de las variables. Este 
# resultado será mostrado en la pantalla y guardado en el diccionario anterior que asocia cada resultado a su variable, permitiendo 
# un fácil acceso a la información en caso de que se necesite a futuro.
#%%

# Aplicamos el comando describe() de pandas sobre todos los valores numericos
for variable in categorical:
    print(f"{variable}: ")
    description = titanic_data_cleaned[variable].astype('category').describe()
    print(description)
    print()
    statistics[variable] = description
    # Hacemos también una representación por cada una de las variables en formato histograma
    plt.figure()
    sns.histplot(titanic_data_cleaned[variable].astype('category'))
    plt.show()
    

#%% [markdown]
#
#---
### Parte 6: Determinando outliers
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1, 7 y 4 y 5 para poder funcionar*
#
# En este apartado vamos a trabajar en base a los datos obtenidos anteriormente para valorar la presencia de outliers 
# en el DataSet en que nos encontramos. 
#
# Para ello podemos utilizar dos maneras, una gráfica (empleando boxplot o scatterplot en función del tipo de variable) 
# o mediante unas fórmulas ligadas al rango entre quartiles (IQR), la cual emplea el quartil 1 y 3 (percentiles 25 y 75).
# En esta iteración, voy a hacer una representación mediante boxplot ya que considero que se destacan más los outliers
# que en el scatterplot.
#
# Los valores de los quartiles los tenemos almacenados en el diccionario 'statistics', lo que nos provee de un fácil 
# acceso a la inforamción. Destaquemos que no se hará el cálculo de las variables categóricas ya que la definición de 
# outliers para estas variables no posee la misma relevancia (si poseé alguna) que las variables numéricas.
#%%

for variable_name, variable_stat in statistics.items():
    try:
        # Probamos a sacar los quartiles 3 y 1 de las estadísticas. Saltará un error si las variables son categóricas
        Q3, Q1 = variable_stat['75%'], variable_stat['25%']
        # Cálculo del rango entre cuartiles
        IQR =  Q3 - Q1
        count = sum(1 for x in titanic_label_encoded[variable_name] if x >= 1.5*IQR+Q3 or x <= Q1-1.5*IQR)
        # Hacemos el boxplot
        plt.figure()
        sns.boxplot(titanic_data_cleaned[variable_name])
        plt.show()
        # Mostramos los resultados numéricos
        if count == 0:
            print(f"La variable {variable_name} no cuenta con outliers.")
        else:
            print(f"La variable {variable_name} cuenta con {count} outliers.")
    # No existe campo '75%' ni '25%' -> Variable categórica
    except KeyError:
        print("----")
        print(f"Omitida variable {variable_name} por ser categórica.")
        
#%% [markdown]
#
#---
### Parte 8: Normalizando y estandarizando el DataSet
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1, 7 y 4 y 5 para poder funcionar*
#
# En este apartado vamos a hacer uso de las estadísticas del DataSet de pandas (mínimo, máximo, media, dev_est) para 
# normalizar y estandarizar el DataSet.
#
# La normalización se hará mediante la fórmula de escalamiento de Min-Max. Esta dice así:
# X_normalizada = (X-X_min)/(X_max-X_min) 
#
#
# La estandarización se hará mediante la fórmula de estandarización o Z-scoring. Esta dice así:
# X_estandarizada = (X-mean(X))/desv_est(X)
#
# Mencionar también que haremos uso del set tras haber aplicado el label encoder ya que necesitamos las variables numéricas.
#%%

t_min, t_max = titanic_label_encoded.min(), titanic_label_encoded.max()
titanic_normalized = (titanic_label_encoded-t_min)/(t_max-t_min)
sns.pairplot(titanic_normalized)
plt.show()
titanic_normalized
#%%

t_mean, t_std = titanic_label_encoded.mean(), titanic_label_encoded.std()
titanic_standarized = (titanic_label_encoded-t_mean)/(t_std)

sns.pairplot(titanic_standarized)
plt.show()
titanic_standarized
#%% [markdown]
# Como podemos observar en los gráficos, la estructura es idéntica en ambos casos (e igual que en la Parte 3). Esto es 
# sucede debido a la naturaleza de la transformación lineal que realizamos. De todas formas los plots si que nos permiten ver
# cambios que no son perceptibles tan facilmente solo viendo los números: el rango.
#
# Este es al factor más determinante a la hora de elegir una fórmula para trabajar con los datos, detacando que sea como 
# fuere la estructura inicial, la normalización siempre te deja el rango entre 0 y 1, lo cual facilita los cálculos estadísitcos.
# Por otra parte la estandarización escala los datos a un rango más pequeño pero variable con cada medida. Esto puede 
# provocar problemas de inconsistencia de escalas a la hora de realizar los cálculos (obligando a que estos sean porcentuales)
# lo cual no es tan cómodo para trabajar con ello.
#
# Como conclusión, optaré por utilizar en la mayoría de los casos una estructura normalizada ante una estandarizada.
#
#
## *APARTADO B*
#
#
### Parte 1:
#
# Comenzamos cargando los datos usando pandas. De esta manera vamos a tener un DataFrame que contenga toda nuestra 
# información siguiendo la estructura de columnas presente en el archivo.
#%%

# Determinamos la seed 
SEED = 22082022

# Cargar los datos
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ml100_dataframe = pd.read_csv('ml100.data', sep='\t', names=column_names)

#%% [markdown]
#---
### Parte 2:
#
# *DISCLAIMER: Esta celda requiere de la ejecución de las parte 1 para poder funcionar*
#
# En este apartado establecemos un lector que delimita los posibles valores de la variable rating de nuestros datos 
# entre 1 y 5. Seguidamente convertimos el DataFrame de pandas a un Dataset de surprise, otorgando la posibilidad 
# de trabajar con surprise sin problema. 
#%%
# Declaramos el reader
reader = Reader(rating_scale=(1, 5))
# Convertimos a Dataset omitiendo la última columna pues no será relevante para este apartado
ml100_dataset = Dataset.load_from_df(ml100_dataframe[['user_id', 'item_id', 'rating']], reader)
# Usamos la función division para entreno de surprise para dividir "aleatoriamente" (mediante SEED) el dataset en 75-25
train_set, test_set = ms.train_test_split(ml100_dataset, test_size=0.25, random_state=SEED)

# Verificamos que la división se haya realizado correctamente
print(f"Longitud datos: {len(ml100_dataframe)}")
print(f"Longitud set de entrenamiento: {train_set.n_ratings}")
print(f"Longitud set de evaluación: {len(test_set)}")

#%% [markdown]
#---
### Parte 3: 
#
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1 y 2 para poder funcionar*
#
# Para este apartado vamos a emplear diferentes algoritmos de recomendación y compararlos con la idea de elegir un 
# mejor candidato para nuestro trabajo.
#
# Por una parte usaremos el Filtrado Colaborativo de Vecinos KNN de la biblioteca surprise usando la métrica de Pearson.
#
# Por otra parte emplearemos la factorización de matrices usando los algoritmos SVD (Single Value Decomposition) y 
# NMF (non Negative Matrix Factorization). 
#%%

# Vamos a crear un diccionario para almacenar las opciones de simulación
sim_options = {
    "name": "pearson",
    "user_based": True
}
# Comenzamos por KNN aplicado a los usuarios  
knn_user_based = pa.knns.KNNBasic(sim_options=sim_options, random_state=SEED)

# Reconfiguramos el diccionario
sim_options["user_based"] = False
# Hacemos KNN aplicado a productos
knn_product_based = pa.knns.KNNBasic(sim_options=sim_options, random_state=SEED)

# Pasamos a hacer la sección de SVD
SVD_set = pa.matrix_factorization.SVD(random_state=SEED)

# Y finalmente NMF
NMF_set = pa.matrix_factorization.NMF(random_state=SEED)

#%% [markdown]
#---
### Parte 4: 
#
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1, 2 y 3 para poder funcionar*
#
# Ya preparado el espacio de trabajo, hay que entrenar los diferentes modelos con los datos de entreno (train_data)
#
# Para esto usaremos el método fit que tienen todos los modelos de la librería surprise que estamos utilizando
#%%
# Entrenamos los KNN
knn_user_trained = knn_user_based.fit(trainset=train_set)
knn_product_trained = knn_product_based.fit(trainset=train_set)

# Entrenamos los algoritmos de factorización de matrices
SVD_trained = SVD_set.fit(trainset=train_set)
NMF_trained = NMF_set.fit(trainset=train_set)
#%% [markdown]
#---
### Parte 5:
#
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1, 2, 3 y 4 para poder funcionar* 
#
# Con los modelos entrenados, pasaremos a probar que funciones bien mendiante la opción test con los datos de set (test_set).
#
# Finalmente tomaremos las 5 primeras predicciones y discutiremos los resultados. 
#%%

# Comenzamos realizando los tests:
resultados_test = {}
# Ejecutamos los test y los 
resultados_test['knn_user'] = knn_user_trained.test(test_set)
resultados_test['knn_product'] = knn_product_trained.test(test_set)
resultados_test['SVD'] = SVD_trained.test(test_set)
resultados_test['NMF'] = NMF_trained.test(test_set)

# Mostramos los resultados de las predicciones
for i in range(5):
    print(f"Elemento {i}:")
    for key, value in resultados_test.items():
        print(f"\tPara {key}:")
        print(f"\t{value[i]}")

#%% [markdown]
#
# Los resultados se muestran de la siguiente forma (acorde a las columnas):
# - ID usuario
# - ID película
# - Nota real
# - Nota predicha
# - Detalles (sin importancia)
#
#---
### Parte 6: 
# *DISCLAIMER: Esta celda requiere de la ejecución de las partes 1, 2, 3, 4 y 5 para poder funcionar*
#
# En este apartado usaremos 4 métricas de comparación para finalmente determinar que modelo es más preciso para 
# nuestra valoracion. Estás métricas serán las siguientes:
# - RMSE -> Cuánto menos mejor
# - precision -> Cuánto más mejor
# - recall -> Cuánto más mejor
# - NDCG (solo para 10 elementos) -> Cuánto más mejor
# 
# Para todos los algortimos salvo el RMSE, necesitaremos usar dos listas. Una de las listas contendrá todos los 
# elementos con ranking real >= 4 y la otra, todos los elementos con ranking estimado >=4. Para el caso del RMSE vamos 
# a coger solo los valores que cumplan ambas propiedades a la vez, pues este solo recibe una lista. 
# 
# Cabe mencionar que a la hora de aplicar del filtro para los valores >= 4, también vamos a ordenar la lista de manera 
# que aquellos con la mejor nota estimada estén arriba del todo. Esto se hace para que cuando apliquemos el NDCG sobre
# 10 elementos, estos elementos sean aquellos con la mejor nota estimada.
# 
#%%

# Comenzamos seleccionando los valores que tengan un ranking >= 4 y aislandolos en un diccionario.
resultados_filtrados = {}

# El diccionario guardará por cada key una tupla (relevantes reales, relevantes estimados)
# de manera que ambas listas de la tupla esten ordenadas de manera descendente en base al mismo criterio, x.est.
for key, resultado in resultados_test.items():
    # Ordenamos en base a r_est
    resultado.sort(key=(lambda x: x.r_ui), reverse=True)
    # Guardamos la tupla en el diccionario
    resultados_filtrados[key] = ([int(result.r_ui >= 4) for result in resultado], 
                                 [int(result.est >= 4) for result in resultado])

print("|-RMSE (<)-------------------------------------------------")
# Aplicamos el RMSE sobre todos los elementos
for key, resultado in resultados_test.items():
    # Filtramos los resultados de manera que solo tomemos aquellos con ambos valores >= 4
    resultado = [result for result in resultado if result.r_ui >= 4 and result.est >= 4]
    print(f"\t{key} - {acu.rmse(resultado, verbose=False)}")

print("")
print("|-Precision, Recall y NDCG (>)-----------------------------")   
# Aplicamos el resto de los algoritmos sobre los elmentos filtrados y ordenados
for key, resultado in resultados_filtrados.items():
    print(f"\t{key}:")
    print(f"\t\tprecision: {met.precision_score(resultado[0], resultado[1])}")
    print(f"\t\trecall: {met.recall_score(resultado[0], resultado[1])}")
    print(f"\t\tNDCG: {met.ndcg_score([resultado[0]], [resultado[1]], k=10)}")
    
#%% [markdown]
#---
### Parte 7: Conclusiones
#
# Analizando os datos podemos observar las siguientes tendencias:
# 1. SVD y NMF superan a los métodos KNN en todas las métricas:
#
#   - - Hay que tener en cuenta que el RMSE es el único que cuanto menos mejor, por ello, podemos ver que en 
#   todas las categorías, estos dos algortimos destacan.
# 
# 2. Igualdad en los KNN: 
#
#   - - Para todas las métricas, el valor obtenido para el KNN de usuario iguala aquel obtenido para el KNN de producto. 
#   Estas similitudes señalan que pueda haber una relación muy fuerte entre productos y usuarios. 
#
# 3. SVD es el mejor algortimo para nuestro caso:
# 
#   - - Y aunque dijimos antes que el NMF y el SVD están muy a la par, es el SVD el que gana en todas las métricas. 
#       De todas formas, ninguno ha tenido un RMSE lo suficientemente bajo como para determinar que es universalmente 
#       bueno.
