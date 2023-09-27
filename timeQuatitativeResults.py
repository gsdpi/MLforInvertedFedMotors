import copy
from sklearn.model_selection import KFold
from featureExtraction import featureExtraction
from modelGenerator import modelGenerator
from keras import backend as K 
from address import *

N_splits = 5
DataID = "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
modelsID = ["lstm","esn","tcn","seq2point","rocket"]
data = featureExtraction(DataID,statorFreqs=[37],testsID=[21,24],featsDomain="time",timesteps=800,Fm=20000,Fm_target=2000)  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27

METRICS = []

MODEL_LABELS = {"lstm": "LSTM",
                "rocket": "rocket",
                "seq2point":"CNN",
                "tcn": "TCN",
                "esn":"ESN"}

for modelID in modelsID:
    X =copy.deepcopy(data.X)
    y =copy.deepcopy(data.y)

    skf = KFold(n_splits=N_splits,shuffle=True)
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
            data.X_train,data.y_train = X[train_index],y[train_index]
            data.X_test, data.y_test  = X[test_index],y[test_index]
            model = modelGenerator(modelID=modelID, data=data,params={})
            model.train()
            METRICS.append([MODEL_LABELS[modelID],k,"MAE",model.test_MAE])
            METRICS.append([MODEL_LABELS[modelID],k,"MSE",model.test_MSE])
            del model
            K.clear_session()

import pandas as pd
import numpy as np
df_metrics = pd.DataFrame(METRICS,columns = ['Model','k','METRICS','VALUES'])
table_metrics = df_metrics.pivot_table(index='Model',columns='METRICS',values="VALUES",aggfunc=[np.mean, np.std])
table_metrics = table_metrics.swaplevel(axis=1)
table_metrics.sort_index(axis=1,inplace=True)
table_metrics.columns.names = (None,None)
table_metrics = table_metrics.reindex([MODEL_LABELS[model] for model in modelsID])
print(table_metrics.to_latex(float_format="%1.3f"))
table_metrics.to_csv(path_results_metrics+"metricsTime.csv")