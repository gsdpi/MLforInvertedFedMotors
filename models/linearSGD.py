
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class linearSGD(object):
    def __init__(self,data, params: dict, **kwargs) -> None:
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train
        self.y_test  = data.y_test
        self.n_feats = data.X_train.shape[-1]
        # Model Params
        self.alpha             = params.get("alpha",0.001) # Regularization term     
        self.max_iter          = params.get("max_iter",200)
        self.learning_rate     = params.get("learning_rate","adaptive")
        self.n_iter_no_change  = params.get("n_iter_no_change",10)
        # building the model
        self.model  = self.create_model()
    def train(self):
        print(f"training {self.__class__}")
        self.model.fit(self.X_train,self.y_train)
        self.y_test_est = self.model.predict(self.X_test)
        self.test_MSE, self.test_MAE = mean_squared_error(self.y_test,self.y_test_est), mean_absolute_error(self.y_test,self.y_test_est)
        print(f'Test MSE: {self.test_MSE}    Test MAE: {self.test_MAE}  ')
        self.train_hist = None
        return None

    def create_model(self):
        model = SGDRegressor(alpha=self.alpha,
                             learning_rate=self.learning_rate,
                             early_stopping=True,
                             n_iter_no_change=self.n_iter_no_change,
                             max_iter= self.max_iter)
        return model

    def predict(self,X):
        return self.model.predict(X)

    # Class methods for hiperparameter tuning
    @classmethod
    def get_model_type(cls):
        return "sklearn"
    @classmethod
    def get_model_name(cls):
        return "linearSGD"
    @classmethod
    def get_randomSearch_params(cls):
        param_grid = {'alpha':[0.0001, 0.001,0.01,0.1,0.5],
                     'max_iter':np.arange(30,330,30,dtype='int').tolist(),
                     'learning_rate':['optimal','invscaling','adaptive'],
                     'n_iter_no_change' : [10]
                    }
        return param_grid
    @classmethod
    def get_model_obj(cls):
        return SGDRegressor()

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass