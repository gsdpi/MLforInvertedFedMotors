# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from .rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ipdb
from sklearn.base import BaseEstimator

class rocket(object):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train[:,-1]
        self.y_test  = data.y_test[:,-1]
        self.n_feats = self.X_train.shape[-1]
        self.n_timesteps= self.X_train.shape[1]
        
        # Params: units, num. layers, 
        self.n_kernels          = params.get("n_kernels",1000)
        self.alpha              = params.get("alpha",1)

        # List with all keras layers
        
        self.model = self.create_model()

    def apply_kernel_multi(self,X,kernels):
        act = []
        N = X.shape[0]
        for feat in range(self.n_feats):
            act.append(apply_kernels(X[:,:,feat],kernels))
        return np.stack(act,axis=2).reshape(N,-1)
    
    def create_model(self,verbose=True):
        
        # Creating layers
        self.kernels = generate_kernels(self.n_timesteps,self.n_kernels)        
        self.model   = Ridge(alpha=self.alpha)
        return self.model
    
    def train(self):
        self.X_train_ = self.apply_kernel_multi(self.X_train,self.kernels)
        self.X_test_ = self.apply_kernel_multi(self.X_test,self.kernels)
        self.model.fit(self.X_train_,self.y_train)   

        self.y_test_est = self.model.predict(self.X_test_)
        
        self.test_MSE, self.test_MAE = mean_squared_error(self.y_test,self.y_test_est), mean_absolute_error(self.y_test,self.y_test_est)   
        print(f'Test MSE: {self.test_MSE}    Test MAE: {self.test_MAE}  ')

    def predict(self,X):
        X_ = self.apply_kernel_multi(X,self.kernels)
        return  self.model.predict(X_) 
        


    @classmethod
    def get_model_type(cls):
        return "rocket"
    
    @classmethod
    def get_model_name(cls):
        return "rocket"

    @classmethod
    def get_randomSearch_params(cls):
        param_grid = {'n_kernels':np.arange(100,500,100,dtype='int').tolist(),
                     'alpha' : np.arange(0.1,5,0.1,dtype='float').tolist()
                    }
        return param_grid

#https://stackoverflow.com/questions/74852797/wrap-model-with-sklearn-interface
    @classmethod
    def get_model_obj(cls,data,params):

        class Wrapper(BaseEstimator):
            
            def __init__(self, data,n_kernels,alpha):
                self.data = data
                self.n_kernels = n_kernels
                self.alpha = alpha
                self.params = {"n_kernels":self.n_kernels,"alpha":self.alpha}
                self.model = cls(data,params)
            
            def fit(self, X,y):
                self.model.X_train = X
                self.model.y_train = y[:,-1]
                self.model.train()
                return self

            def predict(self, X):
                return self.model.predict(X)
            
            def score(self, X,y):
                y = y[:,-1]
                y_ = self.model.predict(X)
                ipdb.set_trace()
                return mean_squared_error(y,y_)

            def get_params(self, deep=True):
                return {"data":self.data,"n_kernels":self.n_kernels,"alpha":self.alpha}

            def set_params(self, **params):
                self.data = params.get("data")
                self.n_kernels = params.get("n_kernels")
                self.alpha = params.get("alpha")
                return self
        return Wrapper(data,params["n_kernels"],params["alpha"])
    

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass