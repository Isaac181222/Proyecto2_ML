import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

X = np.array([[1, 2], [2, 5], [2, 10], [4, 9], [5, 8], [6, 4], [7, 5], [8, 4]])

complete_link = linkage(X, method='complete')

plt.figure(figsize=(8, 5))
dendrogram(complete_link)
plt.title('Dendrograma - Enlace Completo (Complete Link)')
plt.xlabel('Índice del Punto')
plt.ylabel('Distancia')
plt.show()