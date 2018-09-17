import os
import Algorithms.extraction as Extraction
from sklearn.externals import joblib
import numpy as np

def single_extaction(sig,rate):
    win_size = 0.04
    step_size = 0.01
    F = Extraction.featureExtraction(sig,rate,win_size*rate,step_size*rate)
    return F

def batch_extraction():
    win_size = 0.04
    step_size = 0.01
    database = joblib.load(open('Metadata/database.pkl','rb'))
    i=0
    num_samples = len(database.targets)
    F_global=[]
    for (sig,rate) in database.data:
        F = Extraction.featureExtraction(sig,rate,win_size*rate,step_size*rate)
        F_global.append(np.concatenate((np.mean(F,axis=1),np.std(F,axis=1))))
        i=i+1
        print ("Extracting 42 features of " + str(i) + "/"+ str(num_samples) + " data and saving it to the file.....")
        joblib.dump(F_global,open('Metadata/features.pkl','wb'))
    
    return F_global

