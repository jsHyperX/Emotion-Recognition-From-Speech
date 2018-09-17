from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from scipy import spatial
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from sklearn.externals import joblib
import sys
from Algorithms.preprocessor import Preprocessor
import Algorithms.extraction as ex
import extractor

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def classification_with_SVM(Fglobal,y):

    database = joblib.load(open('Metadata/database.pkl','rb'))
    print("transforming the model with Principle Component Analysis and evaluating the model with k-folds cross-validations...")
    
    k_folds = 10
    sss = StratifiedShuffleSplit(n_splits=k_folds,test_size=0.3,random_state=1)
    splits = sss.split(Fglobal,y)
    
    pp = Preprocessor('standard',n_components=60)
    n_classes = len(database.classes)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf',C=10, gamma=0.01))
    prfs = []; scores = []; acc = np.zeros(n_classes)
    for (train,test) in splits:
        Ftrain = Fglobal[train]; Ftest = Fglobal[test]
        (Ftrain,Ftest) = pp.standardize(Ftrain,Ftest)
        (Ftrain,Ftest) = pp.project_on_pc(Ftrain,Ftest)
        clf.fit(Ftrain, y[train])
        ypred = clf.predict(Ftest)
        scores.append(clf.score(Ftest, y[test]))
        prfs.append(precision_recall_fscore_support(y[test], ypred))
    
    print("\nAccuracy =  %0.2f (%0.2f)\n" % (np.mean(scores), np.std(scores)))

    joblib.dump(pp,open('Metadata/transformation_module.pkl','wb'))
    joblib.dump(clf,open('Metadata/classifier.pkl','wb'))

def classification_with_NeuralNet(Fglobal,y):

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import train_test_split
    from keras.models import model_from_yaml
    from sklearn.preprocessing import StandardScaler

    # best configuration till now: 28-hidden layers,epoch=1000,batch_size=20,accuracy=51.98%
    database = joblib.load(open('Metadata/database.pkl','rb'))
    seed = 7
    classes = {v: k for k, v in iter(database.classes.items())}
    targets = []
    for i in y:
        targets.append(classes[int(i)])
    targets = np.array(targets)

    encoder = LabelEncoder()
    encoder.fit(targets)
    encoded_tar = encoder.transform(targets)
    dummy_tar = np_utils.to_categorical(encoded_tar)

    X_train, X_test, Y_train, Y_test = train_test_split(Fglobal, dummy_tar, test_size=0.20, random_state=seed)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    print('building network..')

    model = Sequential()
    model.add(Dense(70,input_dim=Fglobal.shape[1],activation='relu'))
    model.add(Dense(70,activation='relu'))
    model.add(Dense(70,activation='relu'))
    model.add(Dense(8,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train,Y_train,nb_epoch=250,batch_size=10)
    print("evaluating on testing set...")
    (loss, accuracy) = model.evaluate(X_test,Y_test,batch_size=15, verbose=1)
    print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

    target_dir = 'models2'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save('models2/model2.h5')
    model.save_weights('models2/weights2.h5')
    joblib.dump(sc,open('models2/scaling.pkl','wb'))



