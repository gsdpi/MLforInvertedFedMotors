
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
        self.epochs = params.get("epochs",1000)
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
                    max_iter= self.epochs)
        return model

    def predict(self,X):
        return self.model.predict(X)

    @classmethod
    def type(cls):
        return "sklearn"

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass