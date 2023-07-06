from featureExtraction import featureExtraction
from modelGenerator import modelGenerator
from models import *
from address import *
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
import keras_tuner
import warnings
import csv
warnings.filterwarnings("ignore")
from utils import reset_seeds
import ipdb

callbacks = tf.keras.callbacks
reset_seeds(seed_value=39)

# Program params
STATOR_FREQs = [37]
DATAID       = "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
TESTIDs      = [21,24]
MODELID      = "mlp"
params       = {}
N_ITER_MAX   =  50
N_ITER_PER_SAMPLE = 2



print("Reading data")
data = featureExtraction(DATAID,statorFreqs=STATOR_FREQs,testsID=TESTIDs) 
print("Creating model")
params = {"_":None} # TODO: improve this with subclassing
model = modelGenerator(modelID=MODELID, data=data,params=params)
results = pd.DataFrame(index=[DATAID], columns=['params','score'])
print("Searching params")
if model.get_model_type() == "sklearn":
    params_grid = model.get_randomSearch_params()
    model_obj   = model.get_model_obj()
    # Searching params
    sk_searcher = RandomizedSearchCV(model_obj, param_distributions=params_grid,cv=5, n_iter=N_ITER_MAX)
    sk_searcher.fit(data.X_train,data.y_train)
    #Formatting paramas
    results.loc[DATAID,'params'] = [sk_searcher.best_params_]
    results.loc[DATAID,'score'] = sk_searcher.best_score_

elif model.get_model_type() == "keras":

    # building hyper-model 
    hyper_model   = model.get_model_obj(data)
    # Keras tuner obj
    tuner = keras_tuner.RandomSearch(
                            hypermodel=hyper_model,
                            objective="val_loss",
                            max_trials=N_ITER_MAX,
                            executions_per_trial=N_ITER_PER_SAMPLE,
                            overwrite=True,
                            directory= results_grid_search+'/tmpRandomSearch/',
                            project_name=f"{DATAID}"
                            )
    tuner.search_space_summary()
    # Searchinfg params
    ES_cb = callbacks.EarlyStopping( monitor="val_loss",
                                    min_delta=0.001,
                                    patience=10,                                            
                                    baseline=None,
                                    restore_best_weights=True)
    
    tuner.search(data.X_train, data.y_train, epochs=300, validation_data=(data.X_test, data.y_test),callbacks=[ES_cb])
    # Formatting the params 
    bestParams = tuner.get_best_hyperparameters(1)[0]
    bestModel = tuner.get_best_models(1)[0]
    scores = bestModel.evaluate(data.X_test,data.y_test)
    results.loc[DATAID,'params'] = [model.get_params_from_hp(bestParams)]
    results.loc[DATAID,'score'] = scores[0] if type(scores) == 'list' else scores 

# Saving/updating best hyperparameters
results.to_csv(f'{results_grid_search}grid_searchCNN.csv',mode='a')    
params_path = get_param_path(model.get_model_name())
if os.path.exists(params_path):
    print(f"Updating params in {params_path}")
    df_ = pd.read_csv(params_path,index_col=0)
    df_.loc[DATAID] = results.loc[DATAID]
    df_.to_csv(params_path)
else:
    print(f"Writting params in {params_path}")
    results.to_csv(params_path)
    