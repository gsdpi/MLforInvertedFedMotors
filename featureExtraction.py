
# Authors: Diego García/ Ignacio Díaz
# Description: Read and process data to train, evaluate and use the models 
# 
# TO DO:
#   - manage time/freq
#   - manage feature extraction
import h5py
import pandas as pd
import numpy as np
import os
from utils import get_low_leakage_fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class featureExtraction(object):
    def __init__(self, dataID:str,featsDomain: str='freq',statorFreqs:list=[37],testsID:list=[],testRatio:float=0.2,random_state:int=14,scaler_params:dict={})->None:
        '''
            PARAMS:
                dataID (str)          String ID of the .h5 file to be processed.
                featsDomain (str)     Time or frequency feats  ['time','freq']
                statorFreqs (list)    Stator frequencies [34,36,37,...]
        '''
        self.DATAPATH = os.path.join("./Data",f"{dataID}.h5")
        self.metadata = pd.read_hdf(self.DATAPATH,key="metadatos")
        self.N        = self.metadata.shape[0]
        self.statorFreqs = statorFreqs
        self.testRatio = testRatio
        with h5py.File(self.DATAPATH, 'r') as hf:
            self.data = hf['datos'][:]
        # Filtering data by the selected tests in testsID
        if len(testsID)>0:
            idx_test = np.zeros_like(self.N)
            for testID in testsID:
                idx_test = idx_test | (self.metadata.prueba == f"Prueba_{testID}")
            self.metadata = self.metadata[idx_test]
            self.data = self.data[idx_test]
        
        if featsDomain=="freq":
            self.M,self.X = self.get_freq_feats()
            self.y = self.M[:,3]
        
        
        # Normalization
        #scaler = MinMaxScaler()
        if scaler_params =={}:
            self.scaler_params = (self.X.min(axis=0),self.X.max(axis=0))
        else:
            self.scaler_params = scaler_params

        self.X = (self.X - self.scaler_params[0])/(self.scaler_params[1]-self.scaler_params[0])
        self.X_train,self.X_test,self.y_train,self.y_test,self.M_train,self.M_test = train_test_split(self.X,self.y,self.M,
                                                                                                      test_size=self.testRatio,
                                                                                                       random_state=random_state)

    def get_scaler_params(self):
        return self.scaler_params

    def get_freq_feats(self):
        M = []
        F = []
        # selecting data with the indicated stator freq
        idx = []
        for statorFreq in self.statorFreqs: 
            idx.append(np.where(self.metadata.iloc[:,1]==statorFreq)[0])
        idx = np.concatenate(idx,axis=0)
        for sample_idx in idx:
            sample = self.data[sample_idx,...]
            Fs   = self.metadata.iloc[sample_idx,1]
            Fr   = self.metadata.iloc[sample_idx,2]
            Temp = self.metadata.iloc[sample_idx,4]

            i_a,u_ab,i_ab = self.get_alpha_beta(sample)
            x = np.array([i_ab, u_ab]).T 
            ll_fft = get_low_leakage_fft(x,i_a,Fs=Fs) 
            S       = ll_fft["espectro"]
            Fs_est  = ll_fft['frecuencia_estimada']
            # TODO:
            #   - Use class arguments to select the harmonics used to compute Zs
            if np.abs(Fs-Fs_est)<0.1:
                F.append([S[-5][1]/S[-5][0], S[7][1]/S[7][0], S[13][1]/S[13][0]])
                M.append([Fs_est, Fs, Fr, Temp])
        
        # Frequency feats
        F = np.array(F).squeeze()
        # System/context/aux feats
        M = np.array(M)
        X = X = np.concatenate([np.abs(F),np.angle(F)],axis=1)
        return M,X

    def get_alpha_beta(self,sample):
        a = np.exp(1j*2*np.pi/3)
        u_a  = sample[:,0]
        u_b  = sample[:,1]
        u_c  = sample[:,2]
        i_a  = sample[:,3]
        i_b  = sample[:,4]
        i_c  = sample[:,5]
        i_ab = i_a + a*i_b + a**2*i_c 
        u_ab = u_a + a*u_b + a**2*u_c 

        return i_a,u_ab,i_ab       
        
    def get_time_feats(self):
        pass

# Unit testing
if __name__ == "__main__":
    import ipdb

    # Freq test
    dataID = "raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27"
    data2 = featureExtraction(dataID,statorFreqs=[37,35],testsID=[21])  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27
    data = featureExtraction(dataID,statorFreqs=[37],testsID=[24],scaler_params=data2.scaler_params)  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure("X")
    for feat in range(data.X.shape[-1]):
        plt.subplot(2,3,feat+1)
        plt.plot(data.X[:,feat].T)
    plt.figure('y')
    plt.plot(data.y)

    # Ejemplo de procesamiento con normalización procedente de otro procesamiento previo
    