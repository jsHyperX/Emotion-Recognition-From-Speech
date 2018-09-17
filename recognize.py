from sklearn.externals import joblib
import sys
import numpy as np
import Algorithms.extraction as Extraction
import time 
import wave 
import classifier

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def clf(audio):
    from keras.models import load_model
    win_size = 0.04
    step = 0.01
    F=[]
    # print('reading the audio file....')
    # time.sleep(2)
    data = wave.open(audio,'rb')
    rate = data.getframerate()
    sig = np.fromstring(data.readframes(data.getnframes()),dtype=np.int16)
    # print('done....')
    # print('extracting features from the audio.....')
    features = Extraction.featureExtraction(sig,rate,win_size*rate,step*rate)
    tmp = np.concatenate((np.mean(features, axis=1),np.std(features,axis=1)))
    # time.sleep(2)
    # print('done.....')
    F.append(tmp)
    F=np.array(F)

    model_path = 'models2/model2.h5'
    model_weights_path = 'models2/weights2.h5'
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    sc = joblib.load(open('models2/scaling.pkl','rb'))
    F = sc.transform(F)
    arr = model.predict(F)
    ans = np.argmax(arr)
    return ans+1


def classify(audioFile):
    win_size = 0.04
    step = 0.01
    F=[]
    # print('reading the audio file....')
    # time.sleep(2)
    data = wave.open(audioFile,'rb')
    rate = data.getframerate()
    sig = np.fromstring(data.readframes(data.getnframes()),dtype=np.int16)
    # print('done....')
    # print('extracting features from the audio.....')
    features = Extraction.featureExtraction(sig,rate,win_size*rate,step*rate)
    tmp = np.concatenate((np.mean(features, axis=1),np.std(features,axis=1)))
    # time.sleep(2)
    # print('done.....')
    F.append(tmp)
    F=np.array(F)
    # print('loading the SVM.....')
    classifier = joblib.load('Metadata/classifier.pkl')
    # time.sleep(2)
    pp = joblib.load('Metadata/transformation_module.pkl')
    db = joblib.load('Metadata/database.pkl')
    # print('scaling the data and applying PCA on it....')
    F = pp.standardize_single(F)
    F = pp.project_on_pc_single(F)
    # time.sleep(2)
    # print('the class that the emotion in the audio file belongs to is:')
    ans = classifier.predict(F)
    db.classes = {v: k for k, v in iter(db.classes.items())}
    return int(ans[0])
