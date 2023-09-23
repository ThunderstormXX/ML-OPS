import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class SklearnLinreg :
    def __init__(self) :
        self.model = LinearRegression()
    def train_model(self , X_train ,y_train) :
        X = X_train.reshape(-1,1)
        self.model.fit(X, y_train)
    def test_model(self , X_test , y_test):
        y_preds = self.model.predict(X_test.reshape(-1,1))
        mse = mean_squared_error(y_test, y_preds)
        plt.scatter(X_test, y_test, label='Тестовые данные с шумом')
        plt.plot(X_test, y_preds, color='red', linewidth=3, label='Линейная регрессия')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        return mse
