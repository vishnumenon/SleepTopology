from __future__ import print_function, absolute_import

import numpy as np

import sys

from tralie.TDA import *
from tralie.SlidingWindow import *

import mne
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Poly SVM", "sigmoid SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    SVC(C=1, kernel="poly", decision_function_shape='ovr'),
    SVC(C=1, kernel="sigmoid", decision_function_shape='ovr'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sampleSize = 6000
samplesPerEvent = 2

def getFeatureAlt(signal, pd):
    minMax = [min(signal), max(signal)]
    cellSize = 0.05
    cellCount = (2 - 0.8) / cellSize
    feature = [0] * int(cellCount * cellCount)
    for [x, y] in pd:
        position = int(cellCount * ((x - 0.8) / cellSize) + ((y - 0.8) / cellSize))
        if(position >= len(feature)):
            raise ValueError('Error binning: ' + str(x) + ", " + str(y))
        feature[position] += 1
    return minMax + feature

def getFeature(signal, pd):
    minMax = [min(signal), max(signal)]
    featureSize = 30
    persistences = map(lambda x: x[1]-x[0], pd)
    persistences.sort(reverse=True)
    if len(persistences) >= featureSize:
        return minMax + persistences[:featureSize]
    else:
        return minMax + persistences + [0]*(featureSize - len(persistences))

X = []
y = []

filesToRead = 4
newFileTestX = []
newFileTestY = []

f = 0
with open('Sleep-EDF-DB/RECORDS') as edfs, open('Sleep-EDF-DB/HYPNOGRAMS') as annots:
    for edf, ann in zip(edfs, annots):
        f += 1
        if f > filesToRead:
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
            if event in [6,7,8]:
                print("Skipped event " + str(i))
                continue
            end, _, _ = events[i+1]
            end = end - sampleSize - 1
            if end - start < 100:
                continue
            for _ in range(samplesPerEvent):
                sampleStart = random.randint(start, end)
                [eegfpz], timesfpz = fpz[0, start:(start+sampleSize)]
                [eegpz], timespz = pz[0, start:(start+sampleSize)]

                dim = 100
                tau = 2
                dt = 15

                fpzX = getSlidingWindowInteger(eegfpz, dim, tau, dt)
                fpzX = fpzX - np.mean(fpzX, 1)[:, None]
                fpzX = fpzX/np.sqrt(np.sum(fpzX**2, 1))[:, None]

                pzX = getSlidingWindowInteger(eegpz, dim, tau, dt)
                pzX = pzX - np.mean(pzX, 1)[:, None]
                pzX = pzX/np.sqrt(np.sum(pzX**2, 1))[:, None]


                [_, fpzPD] = doRipsFiltration(fpzX, 1)
                [_, pzPD] = doRipsFiltration(pzX, 1)

                if f == filesToRead:
                    newFileTestX.append(getFeatureAlt(eegfpz, fpzPD) + getFeatureAlt(eegpz, pzPD));
                    newFileTestY.append(event)
                else:
                    X.append(getFeatureAlt(eegfpz, fpzPD) + getFeatureAlt(eegpz, pzPD))
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

def testClassifier(clf, clfName):
    print("Testing classfier: " + clfName)
    clf.fit(trainX, trainY)

    print("Testing on data from training files: ")

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

    print("Testing on data from unseen file: ")

    accuracy = 0.0
    accuracyPerStage = [0.0]*9
    countPerStage = [0.0]*9

    for i in range(len(newFileTestY)):
        pred = clf.predict(newFileTestX[i])
        actual = newFileTestY[i]
        countPerStage[actual] += 1.0
        if pred == actual:
            accuracy += 1.0
            accuracyPerStage[actual] += 1.0

    accuracyPerStage = map(lambda (a,b): a/b if b > 0 else 0, zip(accuracyPerStage, countPerStage))

    accuracy = accuracy / len(newFileTestY)
    print("Accuracy: " + str(accuracy))
    print("Accuracy per stage: " + str(accuracyPerStage))
    print("Count per stage: " + str(countPerStage))

for name, clf in zip(names, classifiers):
    try:
        testClassifier(clf, name)
    except:
        print("Unexpected error:" + str(sys.exc_info()[0]))
