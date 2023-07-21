# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf

import tensorflow as tf
import numpy as np
from .TempBloq import *
from featureExtraction import featureExtraction
layers = tf.keras.layers
K = tf.keras.backend

class tcn(object):
    def __init__(self,data: featureExtraction, params: dict, **kwargs) -> None:      
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train[:,-1]
        self.y_test  = data.y_test[:,-1]
        self.n_feats = self.X_train.shape[-1]
        self.n_timesteps= self.X_train.shape[1]
        # Params: units, num. layers, 

        self.kernel_size    = params.get("kernel_size",4)
        self.n_kernels      = params.get("n_kernels",16)
        self.dilation_base  = params.get("dilation_base",2)
        self.n_tempBlocks   = self.get_receptive_field()

        self.numKernels         = [ self.n_kernels*(i+1) for i in range(self.n_tempBlocks)]        
        self.numDilatations     = [2**i  for i in range(self.n_tempBlocks)]
        #self.numDilatations = [1, 2, 2, 2, 2]
        self.useSkips           =  params.get("useSkips",True)
        self.dropout_rate       =  params.get("dropout_rate",0.2)
        self.lr                 = params.get("lr",0.001)
        self.beta               = params.get("beta",0.8)
        self.epochs             = params.get("epochs",300)
        self.batch_size         = params.get("batch_size",16)
        self.min_delta          = params.get("min_delta",0.001)
        self.patience           = params.get("pacience",30)
        
        # List with all keras layers
        self.layers = []
        self.skips  = []
        self.model = self.create_model()


    def create_model(self,verbose=True):
        
        # Creating layers
        self.input_layer = layers.Input(dtype = tf.float32,shape=[self.n_timesteps,self.n_feats],name='input')
        
        for ll in range(self.n_tempBlocks):
            self.layers.append(TemporalBlock(dilation_rate=self.numDilatations[ll],
                                             nb_filters=self.numKernels[ll],
                                             kernel_size=self.kernel_size,
                                             dropout_rate=self.dropout_rate))
        
        self.output_layer = layers.Dense(units =1, activation= None,
                                         kernel_initializer='RandomNormal',
                                         bias_initializer='RandomNormal')
        #Building model
        x = self.input_layer
        for layer in self.layers:
            x = layer(x)
            self.skips.append(x)

        if self.useSkips:
            x = layers.concatenate(self.skips,axis=-1)
            #x = layers.add(self.skips)   
        
        y = self.output_layer(x[:,-1,:])
        
        self.model = tf.keras.Model(inputs = [self.input_layer], outputs = [y])
        self.optimizer =  tf.keras.optimizers.Adam(
                                            learning_rate=self.lr,
                                            beta_1=0.8,
                                            beta_2=0.999,
                                            epsilon=1e-07
                                        )
        self.model.compile(optimizer=self.optimizer,
                         loss=tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
                         
        if verbose:
            self.model.summary()

        return self.model
    
    def train(self):
            ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                            min_delta=0.001,
                                            patience=self.patience,                                            
                                            baseline=None,
                                            restore_best_weights=True)
            self.training_hist = self.model.fit(self.X_train, self.y_train,
                                                batch_size=self.batch_size,
                                                epochs=self.epochs,
                                                validation_data=(self.X_test, self.y_test),
                                                callbacks=[ES_cb])
            
            
            
            test_loss, self.test_MSE,self.test_MAE= self.model.evaluate(self.X_test, self.y_test, verbose=2)
            print(f'Test MSE: {self.test_MSE}    Test MAE: {self.test_MAE}  ')
    def predict(self,X):
        y_est = self.model.predict(X, batch_size=128) 
        return y_est.squeeze()

    # https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
    def get_receptive_field(self):
        k = self.kernel_size
        b = self.dilation_base 
        l = self.n_timesteps
        r = np.ceil(np.log2(((l-1)*(b-1))/((k-1)*2) +1))
        return int(r)
    @classmethod
    def get_model_type(cls):
        return "keras"
    
    @classmethod
    def get_model_name(cls):
        return "tcn"


    @classmethod
    def get_randomSearch_params(cls,hp):
        param_grid = {'kernel_size'   : hp.Choice("kernel_size", [3,5,7,11,15]),
                     'n_kernels'      : hp.Choice("n_kernels", [16,32,64]),
                     'useSkips'       : hp.Choice("useSkips", [True, False]),
                     'dropout_rate'   : hp.Float("dropout_rate", min_value=0., max_value=0.3),
                     'lr'             : hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
                     'beta'           : 0.8,
                     'epochs'         : 300,
                     'batch_size'     : hp.Int("batch_size", min_value=4, max_value=32, step=4)
                    
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

        params = {'kernel_size'   : best_hp["kernel_size"],
                     'n_kernels'      : best_hp["n_kernels"],
                     'useSkips'       : best_hp["useSkips"],
                     'dropout_rate'   : best_hp["dropout_rate"],
                     'lr'             : best_hp["lr"],
                     'beta' :          0.8,
                     'epochs':         300,
                     'batch_size':     best_hp["batch_size"]
                    
                    }

        return params
    

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass