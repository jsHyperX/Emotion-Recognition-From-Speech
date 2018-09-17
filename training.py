from optparse import OptionParser
import numpy as np
from sklearn.externals import joblib
from time import sleep
import sys
from database import Database
from Algorithms.preprocessor import Preprocessor
import Algorithms.extraction as ex
import extractor,classifier 

def init():
    parser = OptionParser()
    parser.add_option("-p","--database_path",dest="a",default="")
    parser.add_option("-l","--load_data",action="store_true",dest="b")
    parser.add_option("-e","--extract_features",action="store_true",dest="c")
    (options,args) = parser.parse_args(sys.argv)
    return options.a,options.b,options.c

if __name__ == '__main__':

    path,load_data,extract_features = init()

    if load_data:
        print("loading data from the dataset....")
        db = Database(path,decode=False)
        print("saving the dataset info to the file...")
        joblib.dump(db,open('Metadata/database.pkl','wb'))
    else:
        print("getting data from the dataset....")
        db = joblib.load(open('Metadata/database.pkl','rb'))

    num_samples = len(db.targets)
    print("number of dataset samples:" + str(num_samples))

    if extract_features:
        Fglobal = extractor.batch_extraction()
    else:
        print("getting features from files...")
        Fglobal = joblib.load(open('Metadata/features.pkl','rb'))

    print('starting training...')
    Fglobal=np.array(Fglobal)
    y=np.array(db.targets)
    print(Fglobal[1].shape)
    classifier.classification_with_NeuralNet(Fglobal,y)