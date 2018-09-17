import Utilities.recorder as Recorder
import recognize
import time
import wave
import os
import Utilities.analyser as Analyser
import Utilities.noiseReduction as NR
import os
from sklearn.externals import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

def load():
    print('please wait we are loading the database.....')
    db = joblib.load(open('Metadata/database.pkl','rb'))
    classes = {v:k for k,v in iter(db.classes.items())}
    print('done')
    return classes

def batch_test(path):
    print('starting..')
    sz=0;cnt=0
    for audio in os.listdir(path):
        sz+=1
        audio_path = os.path.join(path,audio)
        ans = recognize.classify(audio_path)
        if int(audio[7]) is ans:
            cnt+=1
        
        print('actual: %s                       predicted: %s' %(classes[int(audio[7])],classes[ans]))
    print('done..')

    print('accuracy: %.3f' %((cnt/sz)*100))

if __name__ == '__main__':
    i=0
    n=100
    classes = load()
    while n:
        print('please enter 1 to record,2 to run batch test and 0 to exit...')
        n = int(input())
        if n is 1:
            print('please say something in the microphone')
            filename = 'Outputs/output'+str(i)+'.wav'
            i+=1
            Recorder.record(filename,0)
            print('cleaning the audio for you...')
            res = Analyser.clean(filename)
            print('almost done...')
            print('please wait while we fetch the inherent emotion of your audio file....')
            emotion = recognize.clf(filename)
            print('the emotion is : ' + str(classes[emotion]))
        elif n is 2:
            batch_test('testFiles')
        else:
            exit
