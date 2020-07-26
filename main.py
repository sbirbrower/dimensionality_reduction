import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsne import *

class Iris_Data:

    def labeling(df):
        if df['Iris-setosa'] == 'Iris-setosa':
            return 1
        elif df['Iris-setosa'] == 'Iris-versicolor':
            return 2
        elif df['Iris-setosa'] == 'Iris-virginica':
            return 3
        
    def __init__(self):
        """
        returns 149x4 matrix from Iris Data Set
        rows are datapoints, columns are dimensions
        also returns original pandas dataframe with the original labelings
        """
        df = pd.read_csv('iris.data')
        df['label'] = df.apply(Iris_Data.labeling, axis=1)
        del df['Iris-setosa']
        df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
        self.original = df.copy()
        del df['label']
        self.matrix = df.values

    def get_matrix(self):
        return self.matrix, self.original


def pca(X, output_dims=50):
    """
    preprocessing step
    """
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:output_dims])
    return Y    

if __name__ == "__main__":
    data = Iris_Data()
    X, original = data.get_matrix()
    components = pca(X)
    Y = t_sne(components)

    plt.scatter(Y[:, 0], Y[:, 1], c=list(original['label']))
    plt.show()   