
from models import *


# Model factory pattern
def modelGenerator(modelID:str,data,params:dict={}):
    '''
    ARGUMENTS
        modelID (str)                       ID that indicates the model type
        data    (featExtraction object)     Data object needed to train
        params  (dict)                      the params that define the model 
    '''
    data = data
    modelID = modelID
    params  = params
    #TODO: Make it more generic: https://stackoverflow.com/questions/456672/class-factory-in-python 
    if modelID == "mlp":
        model = mlp(data,params)
    elif modelID == "svr":
        model = svr(data,params)
    elif modelID == "linearSGD":
        model = linearSGD(data,params)
    else:
        model = None
        raise Exception("Model not implemented")
    return model
    


if __name__ == "__main__":
    import ipdb
    from featureExtraction import featureExtraction
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils import reset_seeds
    plt.ion()
    reset_seeds(seed_value=39)
    dataID = "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
    modelID = "svr"
    params = {}
    data = featureExtraction(dataID,statorFreqs=[37],testsID=[21,24])  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27
    
    model = modelGenerator(modelID=modelID, data=data,params=params)
    model.train()
    # training_hist = pd.DataFrame(model.training_hist.history)
    # training_hist.plot()