
from models import *
import pandas as pd
from address import *
import ast


# Model factory pattern
def modelGenerator(modelID:str,data,params:dict={},verbose=False,debug = False):
    '''
    ARGUMENTS
        modelID (str)                       ID that indicates the model type
        data    (featExtraction object)     Data object needed to train
        params  (dict)                      the params that define the model 
    '''
    data = data
    modelID = modelID
    params  = params

    if verbose:
        print("Building model")

    if not params and not debug:
        if verbose:
            print("loading best hyperparameters")
        params_path  = get_param_path(modelID)
        df_params    = pd.read_csv(params_path,index_col=0)
        params       = ast.literal_eval(df_params.loc[data.dataID,'params'])[0]


    #TODO: Make it more generic: https://stackoverflow.com/questions/456672/class-factory-in-python 
    if modelID == "mlp":
        model = mlp(data,params)
    elif modelID == "svr":
        model = svr(data,params)
    elif modelID == "linearSGD":
        model = linearSGD(data,params)
    elif modelID == "tcn":
        model = tcn(data,params)
    elif modelID == "seq2point":
        model = seq2point(data,params)
    elif modelID == "lstm":
        model = lstm(data,params)
    elif modelID == "rocket":
        model = rocket(data,params)
    elif modelID == "esn":
        model = esn(data,params)
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

    # Test freq models
    # modelID = "mlp"
    # params = {}
    # data = featureExtraction(dataID,statorFreqs=[37],testsID=[21,24])  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27
    
    # model = modelGenerator(modelID=modelID, data=data,params=params)
    # model.train()


    # Test time models
    modelID = "lstm"
    params = {}
    data = featureExtraction(dataID,featsDomain="time",statorFreqs=[37],testsID=[21,24],timesteps=1100)  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27 
    
    model = modelGenerator(modelID=modelID, data=data,params=params,debug=False)
    model.train()

    if model.get_model_type=="keras":
        training_hist = pd.DataFrame(model.training_hist.history)
        training_hist.plot()
    
    y_est = model.predict(data.X)
    plt.figure()
    plt.plot(data.y[:,-1])
    plt.plot(y_est)