from __future__ import print_function, absolute_import

import pyedflib
import mne
import numpy as np

from stacklineplot import stackplot
import sys
import re
import random

from tralie.TDA import *
from tralie.SlidingWindow import *
from sklearn.decomposition import PCA
from sklearn import svm
from zerodpersistence import *


def getFeature(pd):
    featureSize = 40
    persistences = map(lambda x: x[1]-x[0], pd)
    persistences.sort()
    if len(persistences) >= featureSize:
        return persistences[:featureSize]
    else:
        return persistences + [0]*(featureSize - len(persistences))

X = []
y = []

for i in range(2,7):
    locations = {}
    firstFile = {}

    allPositions = []
    allDurations = []

    confirmed = {}
    channel = ''
    l = 0
    for line in open("DREAMS/Visual_scoring1_excerpt"+str(i)+".txt"):
        if l == 0:
            channel = re.findall(r'/(.*)]', line)[0]
            l = 1
            continue
        [pos, dur] = [float(x) for x in line.split()]
        firstFile[pos] = dur
        allPositions.append(pos)
        allDurations.append(dur)
    l = 0
    for line in open("DREAMS/Visual_scoring2_excerpt"+str(i)+".txt"):
        if l == 0:
            l = 1
            continue
        [pos, dur] = [float(x) for x in line.split()]
        allPositions.append(pos)
        allDurations.append(dur)
        closest = min(firstFile.keys(), key=lambda x: abs(x-pos))
        if abs(closest - pos) < 5:
            confirmed[(closest + pos)/2] = (firstFile[closest] + dur) / 2

    allPositions.sort()
    negatives = {}
    for k in 2*range(len(confirmed.keys())):
        guess = 0
        while guess in negatives or len(filter(lambda x: abs(x - guess) < 5, allPositions)) > 0:
            guess = random.uniform(0.0, allPositions[-1])
        negatives[guess] = random.choice(allDurations)

    # At this point, negatives contains negative examples and confirmed is positives

    trashed_channels = set(['event_pneumo_aut', 'VTH', 'NAF1', 'PCPAP', 'VAB', 'NAF2P-A1', 'POS', 'FP2-A1', 'O2-A1', 'CZ2-A1', 'event_pneumo'])
    trashed_channels.discard(channel)
    trashed_channels = list(trashed_channels)
    raw = mne.io.read_raw_edf('DREAMS/excerpt' + str(i) + '.edf', preload=True, verbose=False, exclude=trashed_channels)
    chan = raw.copy().pick_channels([channel])
    sigbufs = chan
    sr = raw.info['sfreq'];
    del raw

    for start, v in confirmed.items():
        end = start + v
        [eegSamples], times = sigbufs[0, (start*sr):(end*sr)]
        data = np.array(map(lambda x,y:np.array([x, y]), times, eegSamples))
        PD = getFunc0DPersistence(data);
        feature = getFeature(PD)
        X.append(feature)
        print(feature)
        y.append(1)
    for start, v in negatives.items():
        end = start + v
        [eegSamples], times = sigbufs[0, (start*sr):(end*sr)]
        data = np.array(map(lambda x,y:np.array([x, y]), times, eegSamples))
        PD = getFunc0DPersistence(data);
        feature = getFeature(PD)
        X.append(feature)
        y.append(0)

combined = list(zip(X, y))
random.shuffle(combined)
X[:], y[:] = zip(*combined)

testX = X[:len(X)/4]
testY = y[:len(X)/4]
trainX = X[len(X)/4:]
trainY = y[len(X)/4:]

clf = svm.SVC(decision_function_shape='ovr')
clf.fit(trainX, trainY)

tp = 0.0
fp = 0.0
tn = 0.0
fn = 0.0
for i in range(len(testY)):
    pred = clf.predict(testX[i])
    actual = testY[i]
    if pred == 1 and pred == actual:
        tp += 1.0
    elif pred == 1 and pred != actual:
        fp += 1.0
    elif pred == 0 and pred == actual:
        tn += 1.0
    elif pred == 0 and pred != actual:
        fn += 1.0

print("Accuracy: " + str((tp + tn) / (tp + tn + fp + fn)))
print("Precision: " + str((tp) / (tp + fp)))
print("Recall: " + str((tp) / (tp + fn)))
