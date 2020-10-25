import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

def visualize(X, labels, Z, idx=1, n_samples=1000):
    plt.figure(idx)

    embedding = TSNE(n_components=2, random_state=42, verbose=1)
    X_comp = embedding.fit_transform(X[:n_samples])

    plt.tricontourf(X_comp[:, 0], X_comp[:, 1], Z[:n_samples],
                    cmap='rainbow', alpha=0.3)

    plt.scatter(X_comp[:, 0], X_comp[:, 1], c=labels[:n_samples], cmap='rainbow')

    count = 0
    plt.tight_layout()
    for label, x, y in zip(labels, X_comp[:, 0], X_comp[:, 1]):
        if count % 10 == 0:
            plt.annotate(str(int(label)), xy=(x,y), color='black',
                        weight='normal', size=10)
        count = count + 1

def show():
    plt.show()
