# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from featureExtraction import featureExtraction
from .rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ipdb

class rocket(object):
    def __init__(self,data: featureExtraction, params: dict, **kwargs) -> None:      
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
        return "sklearn"
    
    @classmethod
    def get_model_name(cls):
        return "rocket"

    @classmethod
    def get_randomSearch_params(cls,hp):
        param_grid = {'hiddenLayerUnits':[hp.Int("neurons_l1", min_value=10, max_value=200, step=10),
                                          hp.Int("neurons_l2", min_value=10, max_value=200, step=10),
                                          hp.Int("neurons_l3", min_value=10, max_value=200, step=10)],
                     'activation':        hp.Choice("activation", ["relu", "tanh","sigmoid"]),
                     'initializer':       "glorot_uniform",
                     'lr' :                hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
                     'beta' :              0.8,
                     'epochs':             300,
                     'batch_size':         hp.Int("batch_size", min_value=4, max_value=32, step=4),
                     'n_layers':           hp.Choice("n_layers", [1,2,3])

                    }
        
        return param_grid
    
    @classmethod
    def get_model_obj(cls,data):

        def build_model(hp):
            params = cls.get_randomSearch_params(hp)
            model  = cls(data,params).model
            return model

        return build_model
    @classmethod 
    def get_params_from_hp(cls,best_hp):
        params = {'hiddenLayerUnits':[best_hp["neurons_l1"],
                                          best_hp["neurons_l2"],
                                          best_hp["neurons_l3"]],
                       'activation':      best_hp["activation"],
                        'initializer':    "glorot_uniform",
                        'lr' :            best_hp["lr"],
                        'beta' :          0.8,
                        'epochs':         300,
                        'batch_size':     best_hp["batch_size"],
                        'n_layers':       best_hp["n_layers"]
                    }

        return params
    

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass