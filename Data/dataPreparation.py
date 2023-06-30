#Title: preprocesamiento en frecuencia de las pruebas térmicas 23 
#Author: Diego García Pérez
#Date:  26-04-23
#Description: Procesamiento en frecuencia de los datos. Los datos temporales se conviertes en una representación de Park.
#             Sobre eso se calcula la fft y se trabaja con la parte real y las fases de la impedancia
#             
#             En este script solo busca comprender el vector de park y el procesamiento realizado previamente por Nacho.


import numpy as np
import pandas as pd
import os
import ipdb
import matplotlib.pyplot as plt
from datetime import datetime
import locale
from scipy.io import loadmat
locale.setlocale( locale.LC_TIME, 'C' )

DEBUG = True

plt.ion()
############################################################################################
# PARAMETERS 
############################################################################################
DEBUG = False
VERBOSE = True
# PATH a la carpeta con todos los ensayos
DATAPATH = "/home/datos/motoresFernando/23_01_11_CoupledMotors/2_Pruebas_termicas/"

# Solo proceso el experimento 21. Es el que 
EXP_NUMBER = [21,24,27]
# campos de los .mat no deseados
NOT_FEATS = ['__header__', '__version__', '__globals__']
#Campos del mat a procesar
FEATS_OF_INTEREST = ["data2","time2","data1","time1"]
LABELS = ["Ia","Ib","Ic","Va","Vb","Vc"]
# ENCODER
PPR_ENC = 4096   # Quad encoder with 4 counts per pulse and 4096 Pulses Per Revolution rate
P = 2            # Motor pole pairs

DATOS = []
METADATOS = []
# Creamos la lista de paths a procesar 
availableTests = [TestFolder for TestFolder in os.listdir(DATAPATH) if "Prueba" in TestFolder]
availableTests.sort()
selTests   = [selTest for selTest in availableTests if int(selTest.split("_")[1]) in EXP_NUMBER]
if VERBOSE:
    print("\n"*2)
    print("Tests disponibles")
    for i in availableTests: print(i)
    print("\n"*2)

    print("\n"*2)
    print("Tests seleccionados")
    for i in selTests: print(i)
    print("\n"*2)

for tt,test in enumerate(selTests):
    
    datos_mat_list = []
    metadatos_list = []
    rootTestPath = os.path.join(DATAPATH,test)
    testPath = os.path.join(DATAPATH,test,"sorted")

    # Lectura de la temperatura
    tempFile = [i for i in os.listdir(rootTestPath) if "xlsx" in i][0] 
    Temps = pd.read_excel(os.path.join(rootTestPath,tempFile),index_col=0,usecols=np.arange(7),parse_dates=True)
    
    # Creamos timestamps según los archivos .mat 
    availableFiles = os.listdir(testPath)
    availableFiles = [i for i in availableFiles if not "transient" in i and i.endswith('.mat')]
    availableFiles.sort()
    timestamps_mats = [ "_".join(name.split(".")[0].split("_")[:-1]) for name in availableFiles]
    timestamps_mats.sort()
    timestamps_mats = [datetime.strptime(date,"%Y_%B_%d_%H_%M_%S") for date in timestamps_mats]
    timestamps_mats_df = pd.DataFrame(index = timestamps_mats, data = timestamps_mats)
    
    # Reindex de la temperatura de acuerdo al los timestamps de los .mat (sincronización de los ejes de tiempos)
    Temps.index = Temps.index.tz_localize(None)
    Temps = Temps.loc[timestamps_mats_df.index]
    Temps_df = Temps[[ temp for temp in Temps.columns if temp.startswith('Channel 6')]]
    Temps_df.columns = ["Temperature stator winding (Ch6)"]

    STATOR_FREQ = []
    ROTOR_FREQ = []
    for ff,file in enumerate(availableFiles):
        print(f"Procesando {file} {ff} / {len(availableFiles)}")
        datos_mat = loadmat(os.path.join(testPath, file))
        # Obtenemos la velocidad del rotor para saber los puntos de mayor carga.
        ppr_encoder = 4096
        pole_pairs  = 2
        stator_freq = float(file.split('_')[-1][:2])
        rotor_freq  = -datos_mat['data1'][-1,0]/datos_mat['time1'][-1,0]/ppr_encoder*pole_pairs
        STATOR_FREQ.append(stator_freq)
        ROTOR_FREQ.append(rotor_freq)
		
        # Leemos los datos de tensiones y corrientes
        df_batch = pd.DataFrame(datos_mat['data2'],
                                index=pd.to_datetime(datos_mat['time2'].reshape(-1),unit='s'),                        
                                columns=['iu','iv','iw','vu','vv','vw'])

		# remuestreo a frecuencia fm
        fm = 20000
        tm = 1/fm
        df_batch = df_batch.resample(pd.Timedelta(tm, 's')).mean()

        # muestras por cada *.mat
        N = 10000
        X = df_batch[['vu', 'vv', 'vw', 'iu', 'iv', 'iw']].values[:N,:]
        datos_mat_list.append(X)
        
        metadatos_list.append({
            'idx_reverse': ff,
			'timestamp': datetime.strptime(file[:-9],'%Y_%B_%d_%H_%M_%S'),
			'stator_freq':stator_freq,
			'rotor_freq': rotor_freq, 
			'prueba': selTests[tt]})


        
        # Sacar los armónicos 
        # Calcular la admitancia y sacar el vector de características.
        # guardar todo en un .h5
    
    # Ordenamos la tabla metadatos
    df_meta = pd.DataFrame(metadatos_list).set_index('timestamp')
    df_meta.sort_index(inplace=True)

    # Concatenamos los metadatos
    df_metadatos = pd.concat([df_meta,Temps_df],axis=1)

    # Ordenamos los datos 
    idx =  df_meta['idx_reverse'].values
    datos = np.array(datos_mat_list)[idx,:,:]

    DATOS.append(datos)
    METADATOS.append(df_metadatos)
 


METADATOS = pd.concat(METADATOS,axis=0)
DATOS = np.concatenate(DATOS,axis=0)



pruebas_str = '_'.join([str(_) for _ in selTests])
outFileName = f'raw_data_{N}_samples_fm_{int(fm)}_tests_{pruebas_str}.h5'
print(outFileName)

# guardamos los datos
import h5py
with h5py.File(outFileName, 'w') as hf:
    hf.create_dataset("datos",data=DATOS)

# guardamos los metadatos
METADATOS.to_hdf(outFileName,key='metadatos')


# DEBUG
if DEBUG:
    idx = np.random.choice(DATOS.shape[0])

    plt.close("debug proc signals")
    plt.figure("debug proc signals")
    for feat in np.arange(DATOS.shape[-1]):
        plt.plot(DATOS[idx,:,feat])

    plt.close("debug proc metadatos")
    plt.figure("debug proc metadatos")
    plt.plot(METADATOS['stator_freq'].values)
    plt.plot(METADATOS['rotor_freq'].values)
    plt.plot(METADATOS["Temperature stator winding (Ch6)"].values)

    
    plt.close("debug proc all")
    plt.figure("debug proc all",figsize=(16,9))
    ax = plt.subplot(211)
    plt.plot(np.max(DATOS[:,:,3],axis=-1))
    plt.subplot(212,sharex=ax)
    plt.plot(METADATOS['stator_freq'].values)
    plt.plot(METADATOS['rotor_freq'].values)
    plt.plot(METADATOS["Temperature stator winding (Ch6)"].values)





    # Prueba para comprobar que están sincronizados
    # plt.figure()
    # plt.plot(Temps.iloc[-200:,5].values,c='r',alpha=0.4,label = "temp")
    # plt.plot(STATOR_FREQ +140,c='yellow',alpha = 0.4,label="stator freq")
    # plt.plot(ROTOR_FREQ +140,c = 'orange',alpha = 0.4,label="rotor freq")
    # plt.legend()


    # EJEMPLO DE LECTURA 

