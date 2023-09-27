#!/bin/bash

#Common params
DATAID="raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
STATOR_FREQ="37"
TESTIDS="21,24"
N_ITER=20
ITER_SAMPLE=1
TIMESTEPS=800




# TCN
#python randomSearch.py -dataid $DATAID -stator_freq $STATOR_FREQ -testIDs $TESTIDS -domain time -timesteps $TIMESTEPS -modelID tcn -n_iter $N_ITER -iter_sample $ITER_SAMPLE 

# Seq2point
#python randomSearch.py -dataid $DATAID -stator_freq $STATOR_FREQ -testIDs $TESTIDS -domain time -timesteps $TIMESTEPS -modelID seq2point -n_iter $N_ITER -iter_sample $ITER_SAMPLE 

# Lstm
#python randomSearch.py -dataid $DATAID -stator_freq $STATOR_FREQ -testIDs $TESTIDS -domain time -timesteps $TIMESTEPS -modelID lstm -n_iter $N_ITER -iter_sample $ITER_SAMPLE 

#rocket
python randomSearch.py -dataid $DATAID -stator_freq $STATOR_FREQ -testIDs $TESTIDS -domain time -timesteps $TIMESTEPS -modelID rocket -n_iter $N_ITER -iter_sample $ITER_SAMPLE 

# ESN
#python randomSearch.py -dataid $DATAID -stator_freq $STATOR_FREQ -testIDs $TESTIDS -domain time -timesteps $TIMESTEPS -modelID esn -n_iter $N_ITER -iter_sample $ITER_SAMPLE 


#python randomSearch.py -dataid "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27" -stator_freq "37" -testIDs "21,24" -domain time -timesteps 800 -modelID esn -n_iter 20 -iter_sample 1