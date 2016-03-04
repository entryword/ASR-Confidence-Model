#!/usr/bin/env python

import os
import librosa
import numpy
import scipy


CHUNK_SIZE = 10

def split_matrix(l, size):
    n = len(l[0]) // size
    r = len(l[0]) % size
    b, e = 0, n + min(1, r)
    for i in range(size):
        yield l[:, b:e]
        r = max(0, r-1)
        b, e = e, e + n + min(1, r)

def calculate_segmented_mean_std(l):
    result = []
    for col in split_matrix(l, CHUNK_SIZE):
        result += numpy.mean(col, axis=1).tolist()
        result += numpy.std(col, axis=1).tolist()
    return result


def generate_zero_crossing_rate(y):
    zxr = librosa.feature.zero_crossing_rate(y)
    zxr_segment = calculate_segmented_mean_std(zxr)

    return zxr_segment

def generate_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_segment = calculate_segmented_mean_std(mfcc)
    mfcc_delta_segment = calculate_segmented_mean_std(mfcc_delta)

    return mfcc_segment, mfcc_delta_segment

def generate_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(y, sr)
    chroma_segment = calculate_segmented_mean_std(chroma)

    return chroma_segment

def generate_kurtosis(y, frame_length=2048, hop_length=512, center=True):
    if center:
        y = numpy.pad(y, int(frame_length // 2), mode='edge')

    y_framed = librosa.util.frame(y)
    kurtosis = scipy.stats.kurtosis(y_framed).reshape(1, len(y_framed[0]))
    kurtosis_segment = calculate_segmented_mean_std(kurtosis)

    return kurtosis_segment

def generate_features(filename):
    y, sr = librosa.load(filename)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    length = [len(y) / float(sr)]
    zxr = generate_zero_crossing_rate(y)
    mfcc, mfcc_delta = generate_mfcc(y, sr)

    chroma = generate_chroma(y_harmonic, sr)
    kurtosis = generate_kurtosis(y)

    return length + zxr + mfcc + mfcc_delta + chroma + kurtosis

def feaname(prefix, Fout):
    i = 1

    while (True):
        wav = prefix + str(i) + '.wav'
        print(wav)

        # you can assume anything is in the working directory
        if not os.path.isfile(wav):
            break

        features = generate_features(wav)
        Fout.write(', '.join(str(x) for x in features) + '\n')

        i += 1



if __name__ == '__main__':
    PREFIX = '/tmp2/Data/Source/ted'
    Fout = open('test.txt', 'w')

    feaname(PREFIX, Fout)

