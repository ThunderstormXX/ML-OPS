from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 

class SklearnDataset:
    def __init__(self) :
        self.data = []
        self.X = []
        self.y = []
    def example_data_generate(self) :
        num_samples = 100
        self.X = np.random.rand(num_samples)
        self.y = 5 * self.X + np.random.normal(0, 1, num_samples)
    def split(self) :
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, y_train , X_test , y_test
