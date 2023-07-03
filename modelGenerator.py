
from models import mlp


# Model factory pattern
def modelGenerator(modelID:str,data,params:dict={})->None:
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
    if modelID== "mlp":
        model = mlp(data,params)
    return model
    


if __name__ == "__main__":
    import ipdb
    from featureExtraction import featureExtraction
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.ion()
    dataID = "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
    modelID = "mlp"
    params = {"layers":[10,10]}
    data = featureExtraction(dataID,statorFreqs=[37])  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27
    model = modelGenerator(modelID=modelID, data=data,params=params)
    model.train()
    training_hist = pd.DataFrame(model.training_hist.history)
    training_hist.plot()