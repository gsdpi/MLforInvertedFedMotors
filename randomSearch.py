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
import argparse



callbacks = tf.keras.callbacks
reset_seeds(seed_value=39)

# Defining program params
parser = argparse.ArgumentParser(description="Script to reandomly search the optimal hyperparameters of the models.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#run -i randomSearch -dataid raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27 -stator_freq 37 -testIDs 21,24 -domain time -modelID tcn -n_iter 50 -iter_sample 2" 

parser.add_argument("-dataid", action="store", help="string with name of .h5 used for training",default="raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27")
parser.add_argument("-stator_freq", action="store", help="list with the stator freqs used to extract features",type = str,default="37")
parser.add_argument("-testIDs", action="store", help="list with the number os tests used for training", type = str, default="21,24")
parser.add_argument("-domain", action="store", help="features domain", default="freq")
parser.add_argument("-timesteps", action="store", help="timesteps in time domain", type = int,default=1100)
parser.add_argument("-modelID", action="store", help="string with id of the model to be fine tunned", default="mlp")
parser.add_argument("-n_iter", action="store", help="max. number of iterations in the search", type = int,  default=50)
parser.add_argument("-iter_sample", action="store", help=" number of runs per sample in the search",type =int,  default=2)

args = parser.parse_args()


# Program params
STATOR_FREQs = [int(item) for item in args.stator_freq.split(',')]
DATAID       = args.dataid
TESTIDs      = [int(item) for item in args.testIDs.split(',')]
DOMAIN       = args.domain
TIMESTEPS    = args.timesteps
MODELID      = args.modelID
N_ITER_MAX   =  args.n_iter #50
N_ITER_PER_SAMPLE = args.iter_sample # 2
params       = {}

print("Reading data")
data = featureExtraction(DATAID,statorFreqs=STATOR_FREQs,testsID=TESTIDs,featsDomain=DOMAIN,timesteps=1100) 
print("Creating model")
params = {"_":None} # TODO: improve this with subclassing
model = modelGenerator(modelID=MODELID, data=data,params=params)
results = pd.DataFrame(index=[DATAID], columns=['params','score'])
print("Searching params")
if model.get_model_type() == "sklearn":
    params_grid = model.get_randomSearch_params()
    model_obj   = model.get_model_obj()
    # Searching params
    sk_searcher = RandomizedSearchCV(model_obj, param_distributions=params_grid,cv=5, n_iter=N_ITER_MAX,verbose=1,error_score='raise')
    sk_searcher.fit(data.X_train,data.y_train)
    #Formatting paramas
    results.loc[DATAID,'params'] = [sk_searcher.best_params_]
    results.loc[DATAID,'score'] = sk_searcher.best_score_

elif model.get_model_type() == "rocket":
    params_grid = model.get_randomSearch_params()
    params_grid["data"] = [data]
    params = {"n_kernels":100,"alpha":1}
    model_obj   = model.get_model_obj(data,params)
    # Searching params
    sk_searcher = RandomizedSearchCV(model_obj, param_distributions=params_grid,cv=5, n_iter=N_ITER_MAX)
    sk_searcher.fit(data.X_train,data.y_train)
    #Formatting paramas
    results.loc[DATAID,'params'] = [sk_searcher.best_params_]
    results.loc[DATAID,'score'] = sk_searcher.best_score_
    print(f"Best score: {sk_searcher.best_score_}")

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



