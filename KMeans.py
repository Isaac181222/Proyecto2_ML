import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Inicialización de Datos
X = np.array([
    [1, 2], [2, 5], [2, 10], [4, 9],
    [5, 8], [6, 4], [7, 5], [8, 4]
])

# Inicialización manual de los centroides
centroids = np.array([[4, 5], [5, 7], [6, 6]])

plt.scatter(X[:, 0], X[:, 1], marker='o', edgecolors='k', label="Puntos")
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroides Iniciales')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Paso 1: Inicialización de Centroides")
plt.legend()
plt.show(block=True)

print("Centroides iniciales:")
print(centroids)

labels = np.argmin(cdist(X, centroids), axis=1)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o', edgecolors='k', label="Puntos")
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroides')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Paso 2: Primera Asignación de Clusters")
plt.legend()
plt.show()

print("\nAsignación de puntos en la primera iteración:")
for i in range(3):
    print(f"Cluster {i + 1}: {X[labels == i]}")

new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(3)])

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o', edgecolors='k', label="Puntos")
plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=200, c='red', marker='X', label='Nuevos Centroides')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Paso 3: Recalcular Centroides")
plt.legend()
plt.show()

print("\nNuevos centroides después de la primera iteración:")
print(new_centroids)

labels = np.argmin(cdist(X, new_centroids), axis=1)
new_centroids_2 = np.array([X[labels == i].mean(axis=0) for i in range(3)])

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o', edgecolors='k', label="Puntos")
plt.scatter(new_centroids_2[:, 0], new_centroids_2[:, 1], s=200, c='black', marker='X', label='Centroides Finales')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Paso 4: Segunda Iteración de K-Means")
plt.legend()
plt.show()

print("\nCentroides después de la segunda iteración:")
print(new_centroids_2)

if np.allclose(new_centroids, new_centroids_2):
    print("\n Convergencia alcanzada: los centroides ya no cambian.")
else:
    print("\n Aún no ha convergido, se necesita otra iteración.")