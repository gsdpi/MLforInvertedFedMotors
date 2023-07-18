# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf

import tensorflow as tf
import numpy as np

layers = tf.keras.layers
K = tf.keras.backend

class seq2point(object):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        self.data    = data
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train[:,-1]
        self.y_test  = data.y_test[:,-1]
        self.n_feats = self.X_train.shape[-1]
        self.n_timesteps= self.X_train.shape[1]
        # Params: units, num. layers, 

        self.kernel_size    = params.get("kernel_size",20)
        self.numKernels     = params.get("numKernels",[16,16,16,16])
        self.n_convs   = params.get("n_conv",4)
        
        
        self.activation         = params.get("activation","relu")
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
        
        for ll in range(self.n_convs):
            self.layers.append(layers.Conv1D(filters=self.numKernels[ll],
                                            kernel_size=self.kernel_size,
                                            strides = 1,
                                            padding = "same",
                                            activation = self.activation,
                                            name = f"conv_{ll}"
                                            ))
            self.layers.append(layers.MaxPooling1D(pool_size=2))
        self.layers.append(layers.Flatten())
        self.output_layer = layers.Dense(units =1, activation= None,
                                         kernel_initializer='RandomNormal',
                                         bias_initializer='RandomNormal')
        #Building model
        x = self.input_layer
        for layer in self.layers:
            x = layer(x)
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
        return "seq2point"

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