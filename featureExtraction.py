
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


class featureExtraction(object):
    def __init__(self, dataID:str,featsDomain: str='freq',statorFreqs:list=[37],testRatio:float=0.2,random_state:int=14)->None:
        '''
            PARAMS:
                dataID (str)          String ID of the .h5 file to be processed.
                featsDomain (str)     Time or frequency feats  ['time','freq']
                statorFreqs (list)    Stator frequencies [34,36,37,...]
        '''
        self.DATAPATH = os.path.join("./Data",f"{dataID}.h5")
        self.metadata = pd.read_hdf(self.DATAPATH,key="metadatos")
        self.statorFreqs = statorFreqs
        self.testRatio = testRatio
        with h5py.File(self.DATAPATH, 'r') as hf:
            self.data = hf['datos'][:]

        if featsDomain=="freq":
            self.M,self.X = self.get_freq_feats()
            self.y = self.M[:,3]

        self.X_train,self.X_test,self.y_train,self.y_test,self.M_train,self.M_test = train_test_split(self.X,self.y,self.M,
                                                                                                      test_size=self.testRatio,
                                                                                                       random_state=random_state)

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
    data = featureExtraction(dataID,statorFreqs=[37,35])  # raw_data_10000_samples_fm_20000_tests_Prueba_21_Prueba_24_Prueba_27

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure("X")
    for feat in range(data.X.shape[-1]):
        plt.subplot(2,3,feat+1)
        plt.plot(data.X[:,feat].T)
    plt.figure('y')
    plt.plot(data.y)
