# -*- coding: utf-8 -*-
import tensorflow as tf


layers = tf.keras.layers
LayerNormalization = layers.LayerNormalization
K = tf.keras.backend
class TemporalBlock(tf.keras.models.Model):
    def __init__(self, dilation_rate, nb_filters, kernel_size, 
                    padding='causal', dropout_rate=0.0,**kwargs): 
		
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate

        #self.initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.initializer = "he_normal"
        assert padding in ['causal', 'same']

        super(TemporalBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        
        # block1
        self.conv1 = layers.Conv1D(filters=self.nb_filters,
                                    kernel_size=self.kernel_size,
                                    dilation_rate=self.dilation_rate,
                                    padding=self.padding,
                                    kernel_initializer=self.initializer,
                                    name = "{}_Conv_1".format(self.name))

        self.batch1 = layers.BatchNormalization(name="{}_norm1".format(self.name))
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.SpatialDropout1D(rate=self.dropout_rate,name="{}_drop1".format(self.name))

        # block2
        self.conv2 = layers.Conv1D(filters=self.nb_filters,
                                    kernel_size=self.kernel_size,
                                    dilation_rate=self.dilation_rate,
                                    padding=self.padding,
                                    kernel_initializer=self.initializer,
                                    name = "{}_Conv_2".format(self.name))

        self.batch2 = layers.BatchNormalization(name="{}_norm2".format(self.name))		
        self.ac2 = layers.Activation('relu')
        self.drop2 = layers.SpatialDropout1D(rate=self.dropout_rate,name="{}_drop2".format(self.name))

        # in order to set the same amount of channels in the residual connection
        if self.nb_filters != input_shape[-1]:  
            self.downsample = layers.Conv1D(filters=self.nb_filters,
                                            kernel_size=1, 
                                            padding='same',
                                            kernel_initializer=self.initializer,
                                            name = "{}_downsample".format(self.name))

        else:
            self.downsample = None

        self.ac3 = layers.Activation('relu')

        super(TemporalBlock, self).build(input_shape)

    def get_config(self):
        config ={}
        return config
    def call(self, x,training=None):

        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x,training=training) 
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x,training=training) 

        if self.downsample!=None:    # match the dimention
            prev_x = self.downsample(prev_x)
        #assert prev_x.shape == x.shape
        res = self.ac3(prev_x + x)
        return res             # skip connection

if __name__ == "__main__":
	# Input layer
    n_layers =4
    n_timesteps = 1050
    input_layer = layers.Input(dtype = tf.float32,shape=[n_layers,n_timesteps],name='input')

    # Defining the model
    kernel_size = 3
    numKernels = [32,64,128]
    numDilatations = [2,2,2]
    tempBlocks = []
    output_size = 1
    for kk,n_kernel in enumerate(numKernels):
        tempBlocks.append(TemporalBlock(numDilatations[kk],nb_filters=n_kernel,kernel_size=kernel_size))
	
    #Building the model
    x = input_layer
    for layer in tempBlocks:
        x = layer(x)
    output_layer = layers.Dense(output_size, kernel_initializer="RandomNormal")
	
    y = output_layer(x[:,-1,:])
    model = tf.keras.Model(inputs=[input_layer],outputs=[y])
    model.summary()