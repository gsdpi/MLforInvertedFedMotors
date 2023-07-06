
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class svr(object):
    def __init__(self,data, params: dict, **kwargs) -> None:
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test

        self.y_train = data.y_train 
        self.y_test  = data.y_test
        self.n_feats = data.X_train.shape[-1]
        # Model Params
        self.kernel = params.get("kernel","rbf")     
        self.C      = params.get("C",3000)   
        self.gamma  = params.get("gamma",0.001)  
        self.max_iter = params.get("max_iter",1000)
        # building the model
        self.model  = self.create_model()
    def train(self):
        self.model.fit(self.X_train,self.y_train)
        self.y_test_est = self.model.predict(self.X_test)
        self.test_MSE, self.test_MAE = mean_squared_error(self.y_test,self.y_test_est), mean_absolute_error(self.y_test,self.y_test_est)
        print(f'Test MSE: {self.test_MSE}    Test MAE: {self.test_MAE}  ')
        self.train_hist = None
        return None

    def create_model(self):
        model = SVR(kernel=self.kernel,
                    C     = self.C,
                    gamma = self.gamma,
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
        return "svr"
    @classmethod
    def get_randomSearch_params(cls):
        param_grid = {'kernel':["poly","rbf","sigmoid"],
                     'max_iter':np.arange(30,330,30,dtype='int').tolist(),
                     'C':np.arange(50,3050,50,dtype='int').tolist(),
                     'gamma' : np.arange(0.001,0.2,0.001,dtype='float').tolist()
                    }
        return param_grid
    @classmethod
    def get_model_obj(cls):
        return SVR()


# Unit testing
if __name__ == "__main__":
    import ipdb
    pass