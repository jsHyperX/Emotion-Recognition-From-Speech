import librosa
from pysndfx import AudioEffectsChain
import numpy as np
import math
import python_speech_features
import scipy as sp
from scipy import signal
import os
import wave

def read(file):
    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(file)
    return y, sr

def reduce_noise_power(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1
    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    # print(less_noise)
    y_clean = less_noise(y)

    return y_clean

def reduce_noise_centroid_s(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = np.max(cent)
    threshold_l = np.min(cent)
    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5).limiter(gain=6.0)
    y_cleaned = less_noise(y)

    return y_cleaned

def reduce_noise_centroid_mb(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
    # less_noise = AudioEffectsChain().lowpass(frequency=threshold_h).highpass(frequency=threshold_l)
    y_cleaned = less_noise(y)

    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)
    boost_l = math.floor(rows/6)
    boost = math.floor(rows/3)

    # boost_bass = AudioEffectsChain().lowshelf(gain=20.0, frequency=boost, slope=0.8)
    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)#.lowshelf(gain=-20.0, frequency=boost_l, slope=0.8)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted

def output_file(destination ,filename, y, sr, ext=""):
    destination = destination + filename[:-4] + ext + '.wav'
    librosa.output.write_wav(destination, y, sr)
    
def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)

    return y_trimmed, trimmed_length

def main(path):
    for audio in os.listdir(path):
        audio_path = os.path.join(path,audio)
        # print(audio_path)
        data,rate = read(audio_path)
        # print(data,rate)
        y_reduced_power = reduce_noise_power(data,rate)
        y_reduced_centroid_s = reduce_noise_centroid_s(data,rate)
        y_reduced_centroid_mb = reduce_noise_centroid_mb(data,rate)

        y_reduced_power, time_trimmed = trim_silence(y_reduced_power)
        # print (time_trimmed)

        y_reduced_centroid_s, time_trimmed = trim_silence(y_reduced_centroid_s)
        # print (time_trimmed)

        y_reduced_power, time_trimmed = trim_silence(y_reduced_power)
        # print (time_trimmed)
        y_reduced_centroid_mb, time_trimmed = trim_silence(y_reduced_centroid_mb)

        output_file('01_samples_trimmed_noise_reduced/' ,audio_path, y_reduced_power, sr, '_pwr')
        output_file('01_samples_trimmed_noise_reduced/' ,audio_path, y_reduced_centroid_s, sr, '_ctr_s')
        output_file('01_samples_trimmed_noise_reduced/' ,audio_path, y_reduced_centroid_mb, sr, '_ctr_mb')

if __name__ == '__main__':
    main('testFiles')