from featureExtraction import featureExtraction
from modelGenerator import modelGenerator
from models import *
from address import *
from sklearn.model_selection import RandomizedSearchCV
import warnings
import csv
warnings.filterwarnings("ignore")
from utils import reset_seeds

reset_seeds(seed_value=39)

STATOR_FREQs = [37]
DATAID       = "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
TESTIDs      = [21,24]
MODELID      = "linearSGD"
params       = {}
N_ITER_MAX   =  50
N_ITER_PER_SAMPLE = 2
print("Reading data")
data = featureExtraction(DATAID,statorFreqs=STATOR_FREQs,testsID=TESTIDs) 
print("Creating model")
model = modelGenerator(modelID=MODELID, data=data,params=params)
results = pd.DataFrame(index=[DATAID], columns=['params','score'])
print("Searching params")
if model.get_model_type() == "sklearn":
    params_grid = model.get_randomSearch_params()
    model_obj   = model.get_model_obj()
    sk_searcher = RandomizedSearchCV(model_obj, param_distributions=params_grid,cv=5, n_iter=N_ITER_MAX)
    sk_searcher.fit(data.X,data.y)
    results.loc[DATAID,'params'] = [sk_searcher.best_params_]
    results.loc[DATAID,'score'] = sk_searcher.best_score_

params_path = get_param_path(model.get_model_name())
if os.path.exists(params_path):
    print(f"Updating params in {params_path}")
    df_ = pd.read_csv(params_path,index_col=0)
    df_.loc[DATAID] = results.loc[DATAID]
    df_.to_csv(params_path)
else:
    print(f"Writting params in {params_path}")
    results.to_csv(params_path)