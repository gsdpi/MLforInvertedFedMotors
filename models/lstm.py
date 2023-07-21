# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf

import tensorflow as tf
import numpy as np
from .TempBloq import *
from featureExtraction import featureExtraction
layers = tf.keras.layers
K = tf.keras.backend

class lstm(object):
    def __init__(self,data: featureExtraction, params: dict, **kwargs) -> None:      
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train[:,-1]
        self.y_test  = data.y_test[:,-1]
        self.n_feats = self.X_train.shape[-1]
        self.n_timesteps= self.X_train.shape[1]
        
        # Params: units, num. layers, 
        self.conv               = params.get("conv",True)
        self.kernel_size        = params.get("kernel_size",7)
        self.n_kernels          = params.get("n_kernels",32)
        self.units_lstm         = params.get("units_lstm",120)
        self.return_seq         = params.get("return_seq",True)
        self.lr                 = params.get("lr",0.001)
        self.beta               = params.get("beta",0.8)
        self.epochs             = params.get("epochs",300)
        self.batch_size         = params.get("batch_size",16)
        self.min_delta          = params.get("min_delta",0.001)
        self.patience           = params.get("pacience",30)

        # List with all keras layers
        self.layers = []
        self.model = self.create_model()


    def create_model(self,verbose=True):
        
        # Creating layers
        self.input_layer = layers.Input(dtype = tf.float32,shape=[self.n_timesteps,self.n_feats],name='input')
        
        if self.conv:
            self.layers.append(layers.Conv1D(filters=self.n_kernels,kernel_size=self.kernel_size))
            self.layers.append(layers.MaxPooling1D(pool_size=2))
        self.layers.append(layers.LSTM(units = self.units_lstm,input_shape=(self.n_timesteps,self.n_feats),return_sequences=True))

        self.output_layer = layers.Dense(units =1, activation= None,
                                         kernel_initializer='RandomNormal',
                                         bias_initializer='RandomNormal')
        #Building model
        x = self.input_layer
        for layer in self.layers:
            x = layer(x)
            
        x = layers.Flatten()(x)
        y = self.output_layer(x)
        
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


    @classmethod
    def get_model_type(cls):
        return "keras"
    
    @classmethod
    def get_model_name(cls):
        return "lstm"


    @classmethod
    def get_randomSearch_params(cls,hp):
        param_grid = {'conv'       :  hp.Choice("conv", [True,False]),
                     'kernel_size' :  hp.Choice("kernel_size", [3,5,7,11,15,23,32]),
                     'n_kernels'   :  hp.Choice("n_kernels", [16,32,64]),
                     'units_lstm'  :  hp.Choice("units_lstm", [32,64,128,256]),
                     'return_seq'  :  hp.Choice("return_seq", [True,False]),                    
                     'lr'          :  hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
                     'beta'        :  0.8,
                     'epochs'      :  300,
                     'batch_size'  :  hp.Int("batch_size", min_value=4, max_value=32, step=4)

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
        params = {      'conv'        :  best_hp["conv"],                  
                        'kernel_size' :  best_hp["kernel_size"],
                        'n_kernels'   :  best_hp["n_kernels"],
                        'units_lstm'  :  best_hp["units_lstm"],
                        'return_seq'  :  best_hp["return_seq"],
                        'lr'          :  best_hp["lr"],
                        'beta'        :  0.8,
                        'epochs'      :  300,
                        'batch_size'  :  best_hp["batch_size"]
                    }

        return params
    

# Unit testing
if __name__ == "__main__":
    import ipdb
    pass