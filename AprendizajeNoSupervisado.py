# Importamos las librerias
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargamos y normalizamos el dataset
data = pd.read_csv('transmilenio.csv')

X = data[['Distancia', 'Paradas', 'Tiempo']]
X_norm = (X - X.mean()) / X.std()


# Segmentamos los datos
kmeans = KMeans(n_clusters=4, n_init=1, random_state=0)
kmeans.fit(X_norm)

# Imprimimos la matriz
print(kmeans.cluster_centers_)

# Creamos la gráfica

data['Cluster'] = kmeans.labels_

plt.title('Tiempo de recorrido de ruta Transmilenio')
plt.scatter(data['Distancia'], data['Tiempo'], c=data['Cluster'])
plt.xlabel('Distancia')
plt.ylabel('Tiempo')
plt.show()
