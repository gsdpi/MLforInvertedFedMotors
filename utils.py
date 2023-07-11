
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import random as python_random
import tensorflow as tf
import os
from scipy.signal import find_peaks, firwin, filtfilt

# Function to get the peaks of a periodic signal:

def get_peaks(x,fm,low_cut=30,high_cut=40):
	"""
	PARAMS
		x:         signal where the peaks will be found
		fm:        sampling freq.
		low_cut:   Low cut in the band pass filter
		high_cut:  High cut in the band pass filter

	OUTPUT
		- index where the peaks take place in the input signal x
		- filtered signal
	"""
	b = firwin(500,[low_cut,high_cut],fs = fm, pass_zero='bandpass')
	yf = filtfilt(b,1,x)
	idx_peaks = np.where(np.diff(np.sign(yf))>0)[0]
	return idx_peaks, yf


def estimate_fs_from_peaks(y_peaks,fs):
	tm = 1/fs
	return 1/(np.mean(np.diff(y_peaks))*tm)


# Function to get the data in first harmonics in a alpha beta representation.
# Author: Ignacio Díaz Blanco
def get_low_leakage_fft(y,ypicos,fm=20000,Fs=37, num_periodos=8,primer_periodo=5,debug=False):
	'''
	Procesa periodos exactos de la señal en tiempo y frecuencia

	     y: nd-array con los datos (samples, timesteps, channels)
	ypicos: señal de la que se obtienen los picos
		fm: frecuencia de muestreo de los datos
	
	toma "num_periodos" de la señal a partir del "primer_periodo" inclusive

	'''

	# forzamos shape (timesteps,channels)
	if len(y.shape)==1:
		y = y.reshape(-1,1)

	# tamaño de la señal original
	Q = y.shape[0]


	# BUSCAMOS PICOS
	# buscamos los picos de la señal en la señal ypicos (elegir la más adecuada para buscar picos)
	#    - que sobresalen de un umbral del 90%
	#    - que están entre sí a una distancia al menos del 80% del periodo
	# 
	# los tramos entre picos contienen periodos exactos

	

	metodo='filtrado'
	if metodo=='picos':
		picos = find_peaks(ypicos,height=np.max(ypicos)*0.7,distance=fm/Fs*0.99)[0]
		# picos = find_peaks(ypicos,distance=fm/Fs*0.9)[0]

	if metodo=='filtrado':
		b = firwin(500,[30,40],fs = fm, pass_zero='bandpass')
		yf = filtfilt(b,1,ypicos)
		picos = np.where(np.diff(np.sign(yf))>0)[0]



	# concatenamos varios periodos "super-exactos" (de pico a pico)
	kp = np.arange(picos[primer_periodo],picos[primer_periodo+num_periodos])
	yp = y[kp,:]

	# tamaño del tramo de señal con varios periodos exactos
	N = yp.shape[0]

	# periodo de muestreo
	tm = 1/fm

	# vectores de frecuencias y de tiempos
	t  = np.arange(Q)*tm
	tp = kp*tm
	fp = np.arange(N)*fm/N

	# armónicos
	Yp = np.fft.fft(yp,axis=0)/N


	# orden del armónico fundamental (asumimos que "mean(diff(picos))" contiene un periodo exacto)
	idx_fundamental = int(np.round(N/np.mean(np.diff(picos))))
	frecuencia_estimada = 1/(np.mean(np.diff(picos))*tm)


	# armónicos positivos y negativos en p.u. respecto al fundamental
	from math import remainder
	n = np.array([remainder(i,N) for i in np.arange(N)])/idx_fundamental

	# subconjunto con los armónicos múltiplo exacto del fundamental (1x, 2x, 3x, ...)
	idx = np.where(np.round(n) == n)[0]
	na = n[idx]
	Ya = Yp[idx]

	# breakpoint()

	espectro = dict(zip(na,Ya))

	resultados = {
	'tp':tp, 
	'yp':yp, 
	'fp':fp, 
	'Yp':Yp, 
	'na': na,
	'Ya': Ya,
	'idx_fundamental': idx_fundamental, 
	'n':n, 
	'frecuencia_estimada': frecuencia_estimada,
	'espectro': espectro
	}

	return resultados

def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)