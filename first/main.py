from models.linearregression import SklearnLinreg
from dataloader.loaddatasets import SklearnDataset

if __name__ == '__main__' :

    data = SklearnDataset()
    data.example_data_generate()
    X_train ,y_train ,X_test,y_test = data.split()


    model = SklearnLinreg()

    model.train_model(X_train ,y_train)

    result = model.test_model(X_test, y_test)
    print('MSE :' ,result)

    print('hello')
