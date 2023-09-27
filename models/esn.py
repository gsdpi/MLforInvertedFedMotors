# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from featureExtraction import featureExtraction

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import ipdb
import reservoirpy as rpy
rpy.verbosity(0)  
rpy.set_seed(42)  
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import Ridge
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator

class esn(object):
    def __init__(self,data: featureExtraction, params: dict, **kwargs) -> None:      
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train[:,-1]
        self.y_test  = data.y_test[:,-1]
        self.n_feats = self.X_train.shape[-1]
        self.n_timesteps= self.X_train.shape[1]
        
        # Params: units, num. layers, 
        
        self.n_states       = params.get("n_states",300)
        self.rho            = params.get("rho",0.95)
        self.sparsity       = params.get("sparsity",0.01)
        self.lr             = params.get("lr",0.025)
        self.Win_scale      = params.get("Win_scale",150)
        self.input_scale    = params.get("input_scale",5)
        self.Warmup         = params.get("Warmup",100)
        self.alpha          = params.get("alpha",0.01)
        # List with all keras layers
        self.model = self.create_model()

    def get_states(self,X):
        N = X.shape[0]
        S = []
        for idx_sample in range(N):
            s = self.reservoir.run(X[idx_sample,...])
            s = s[self.Warmup:]
            S.append(s)
        return np.vstack(S)
    
    def create_model(self,verbose=True):
        
        # Creating layers
        #self.input = Input()
        self.reservoir = Reservoir( units = self.n_states,
                                    lr=self.lr,
                                    sr=self.rho,
                                    input_scaling=self.input_scale,
                                    rc_connectivity=self.sparsity,
                                    Win=rpy.mat_gen.bernoulli(input_scaling = self.Win_scale))

        
        self.model   = Ridge(alpha=self.alpha)
        
        return self.model
    
    def train(self):
        print("getting training states")
        S = self.get_states(self.X_train)
        self.y_train = np.repeat(self.y_train,self.n_timesteps-self.Warmup,axis=0) 
        print("Training readout")
        self.model.fit(S,self.y_train)   
        # print(self.reservoir.is_initialized, self.readout.is_initialized, self.readout.fitted)

        self.y_test_est = self.predict(self.X_test)
        
        self.test_MSE, self.test_MAE = mean_squared_error(self.y_test,self.y_test_est), mean_absolute_error(self.y_test,self.y_test_est)   
        print(f'Test MSE: {self.test_MSE}    Test MAE: {self.test_MAE}  ')

    def predict(self,X,agg = "mean"):
        w_size = self.n_timesteps - self.Warmup
        print("Getting  states")
        S = self.get_states(X)
        print("Getting outputs")
        y_ = self.model.predict(S) 
        y_ = sliding_window_view(y_, window_shape = w_size)[::w_size]
        
        return np.mean(y_,axis=1)
        


    @classmethod
    def get_model_type(cls):
        return "reservoir"
    
    @classmethod
    def get_model_name(cls):
        return "esn"

    @classmethod
    def get_randomSearch_params(cls):
        param_grid = {'input_scale':[5],
                     'n_states':[100,300,500],
                     'rho':np.arange(0.2,0.95,0.1).tolist(),
                     'sparsity':[0.01,0.03,0.1,0.3],
                     'lr':[0,0.01,0.03,0.1,0.3,0.6],
                     'Win_scale':[150],
                     "Warmup":[100],
                     'alpha':[0.0001, 0.001,0.01,0.1,0.5]
                    }
        return param_grid
    
    @classmethod
    def get_model_obj(cls,data,params):

        class Wrapper(BaseEstimator):
            
            def __init__(self,data, input_scale, n_states, rho, sparsity, lr, Win_scale, Warmup,alpha):
                self.data        = data 
                self.input_scale = input_scale
                self.n_states    = n_states    
                self.rho         = rho
                self.sparsity    = sparsity    
                self.lr          = lr
                self.Win_scale   = Win_scale    
                self.Warmup      = Warmup
                self.alpha       = alpha
                
                self.params = {'input_scale':self.input_scale,
                                'n_states':self.n_states,
                                'rho':self.rho,
                                'sparsity':self.sparsity,
                                'lr':self.lr,
                                'Win_scale':self.Win_scale,
                                "Warmup":self.Warmup,
                                'alpha':self.alpha
                                }
                self.model = cls(self.data,self.params)
            
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
                
                return r2_score(y,y_)

            def get_params(self, deep=True):
                
                return {        'data':self.data,
                                'input_scale':self.input_scale,
                                'n_states':self.n_states,
                                'rho':self.rho,
                                'sparsity':self.sparsity,
                                'lr':self.lr,
                                'Win_scale':self.Win_scale,
                                "Warmup":self.Warmup,
                                'alpha':self.alpha
                                }

            def set_params(self, **params):
                self.data        = params.get("data")
                self.input_scale = params.get("input_scale")
                self.n_states    = params.get("n_states")    
                self.rho         = params.get("rho")
                self.sparsity    = params.get("sparsity")    
                self.lr          = params.get("lr")
                self.Win_scale   = params.get("Win_scale")    
                self.Warmup      = params.get("Warmup")
                self.alpha       = params.get("alpha")

                self.params = {'input_scale':self.input_scale,
                                'n_states':self.n_states,
                                'rho':self.rho,
                                'sparsity':self.sparsity,
                                'lr':self.lr,
                                'Win_scale':self.Win_scale,
                                "Warmup":self.Warmup,
                                'alpha':self.alpha
                                }                
                self.model = cls(self.data,self.params)
                return self
       
        return Wrapper(data, params["input_scale"], params["n_states"], params["rho"], params["sparsity"], params["lr"], params["Win_scale"], params["Warmup"],params["alpha"])
    

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass