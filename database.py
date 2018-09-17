import wave
import os
import subprocess as sp
import itertools
import numpy as np
import random

class Database:

	def __init__(self,path,decode):
		self.classes = {'neutral':1,'calm':2,'happy':3,'sad':4,'angry':5,'fear':6,'disgust':7,'surprised':8}
		self.get_database(path)

	def get_database(self,path):
		males = ['01','03','05','07','09','11','13','15','17','19','21','23']
		females = ['02','04','06','08','10','12','14','16','18','20','22','24']
		self.targets = [] 
		self.data = []
		self.train_sets = [] 
		self.test_sets = [] 
		get_data = True
		crossP = itertools.product(males,females)
		print('please wait while we are loading the database......')
		for j in crossP:
			i = 0; train = []; test = [] 
			for audio in os.listdir(path):
				audio_path = os.path.join(path,audio)
				data = wave.open(audio_path,'r')
				rate = data.getframerate()
				x = np.fromstring(data.readframes(data.getnframes()),dtype=np.int16)
				if get_data :
					self.data.append((x,rate))
					self.targets.append(int(audio[7]))
				if audio[18:20] in j:
					test.append(i)
				else:
					train.append(i)
				i = i + 1	
			if train:
				self.train_sets.append(train)
			if test:
				self.test_sets.append(test)
			get_data = False
		print('thanks for waiting.....database has been loaded....')