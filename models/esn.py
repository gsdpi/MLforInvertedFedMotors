# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from featureExtraction import featureExtraction

from sklearn.metrics import mean_squared_error, mean_absolute_error
import ipdb
import reservoirpy as rpy
rpy.verbosity(0)  
rpy.set_seed(42)  
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import Ridge
from numpy.lib.stride_tricks import sliding_window_view

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
        self.scale_inputs   = params.get("scale_input",5)	
        self.n_states       = params.get("n_states",300)
        self.rho            = params.get("rho",0.95)
        self.sparsity       = params.get("sparsity",0.01)
        self.lr             = params.get("lr",0.025)
        self.Win_scale      = params.get("Win_scale",150)
        self.Wfb_scale      = params.get("Wfb_scale",0.)
        self.input_scale    = params.get("input_scale",5)
        self.Washout        = params.get("Washout",0)
        self.Warmup         = params.get("Washout",100)
        self.alpha          = params.get("alpha",0.01)
        self.set_bias       = params.get("Washout",False)


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