from __future__ import print_function, absolute_import

import numpy as np

import sys

from tralie.TDA import *
from tralie.SlidingWindow import *

import mne
import random
from sklearn import tree, svm

sampleSize = 150
samplesPerEvent = 5

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
f = 0
with open('Sleep-EDF-DB/RECORDS') as edfs, open('Sleep-EDF-DB/HYPNOGRAMS') as annots:
    for edf, ann in zip(edfs, annots):
        f += 1
        if f > 2:
            break
        edf = edf.strip()
        ann = ann.strip()
        if edf[:7] != ann[:7]:
            raise Exception('EDF-Annotation Mismatch')
        raw = mne.io.read_raw_edf('Sleep-EDF-DB/' + edf, annotmap='Sleep-EDF-DB/annotmap', annot='Sleep-EDF-DB/' + ann, preload=True)
        fpz = raw.copy().pick_channels(['EEG Fpz-Cz'])
        pz = raw.copy().pick_channels(['EEG Pz-Oz'])
        events = mne.find_events(raw)
        for i in range(len(events) - 1):
            start, _, event = events[i]
            end, _, _ = events[i+1]
            end = end - sampleSize - 1
            if end - start < 100:
                continue
            for _ in range(samplesPerEvent):
                sampleStart = random.randint(start, end)
                [eegfpz], timesfpz = fpz[0, start:(start+sampleSize)]
                [eegpz], timespz = pz[0, start:(start+sampleSize)]

                fpzX = getSlidingWindowInteger(eegfpz, 30, 2, 2)
                fpzX = fpzX - np.mean(fpzX, 1)[:, None]
                fpzX = fpzX/np.sqrt(np.sum(fpzX**2, 1))[:, None]

                pzX = getSlidingWindowInteger(eegpz, 30, 2, 2)
                pzX = pzX - np.mean(pzX, 1)[:, None]
                pzX = pzX/np.sqrt(np.sum(pzX**2, 1))[:, None]


                [_, fpzPD] = doRipsFiltration(fpzX, 1)
                [_, pzPD] = doRipsFiltration(pzX, 1)

                X.append(getFeature(fpzPD) + getFeature(pzPD))
                y.append(event)
            print("Finished event " + str(i))

combined = list(zip(X, y))
random.shuffle(combined)
X[:], y[:] = zip(*combined)

testX = X[:len(X)/4]
testY = y[:len(X)/4]
trainX = X[len(X)/4:]
trainY = y[len(X)/4:]

print("Training Classifier")

#clf = svm.SVC(decision_function_shape='ovo')
clf = tree.DecisionTreeClassifier()
clf.fit(trainX, trainY)

accuracy = 0.0
accuracyPerStage = [0.0]*9
countPerStage = [0.0]*9

for i in range(len(testY)):
    pred = clf.predict(testX[i])
    actual = testY[i]
    countPerStage[actual] += 1.0
    if pred == actual:
        accuracy += 1.0
        accuracyPerStage[actual] += 1.0

accuracyPerStage = map(lambda (a,b): a/b if b > 0 else 0, zip(accuracyPerStage, countPerStage))

accuracy = accuracy / len(testY)
print("Accuracy: " + str(accuracy))
print("Accuracy per stage: " + str(accuracyPerStage))
print("Count per stage: " + str(countPerStage))
