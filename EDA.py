from __future__ import division, print_function, absolute_import

import pyedflib
import mne
import numpy as np

from stacklineplot import stackplot
import sys

from tralie.TDA import *
from tralie.SlidingWindow import *
from sklearn.decomposition import PCA
from zerodpersistence import *

# f = pyedflib.EdfReader("e3.edf");
# n = f.signals_in_file
# signal_labels = [f.getSignalLabels()[0]]
# sigbufs = np.zeros((1, f.getNSamples()[0]))
# sigbufs[0, :] = f.readSignal(0)
# f._close()
# del f

raw = mne.io.read_raw_edf('DREAMS/excerpt5.edf', preload=True, exclude=['event_pneumo_aut', 'PCPAP'])
print(raw.ch_names)
chan = raw.copy().pick_channels(['CZ-A1'])
sigbufs = chan
sr = raw.info['sfreq'];

del raw


def showEEG(start, end):
    stackplot(sigbufs[:, (start*sr):(end*sr)], ylabels=signal_labels)

def getEEG(start, end):
    return sigbufs[0, (start*sr):(end*sr)]

#95-100
#166-176
#225-235
#47723 - 47765
#47660 - 47675
#
# start = 22.02
# end = start+1.15
#
start = 280.5500
end = start+1.1050

[eegSamples], times = getEEG(start, end)

X = getSlidingWindowInteger(eegSamples, 30, 1, 1)
X = X - np.mean(X, 1)[:, None]
X = X/np.sqrt(np.sum(X**2, 1))[:, None]


X = np.array(map(lambda x,y:np.array([x, y]), times, eegSamples))

pca = PCA(n_components = 2)
Y = pca.fit_transform(X)
eigs = pca.explained_variance_

# PDs = doRipsFiltration(X, 1)

fig = plt.figure(figsize=(18, 5))

plt.subplot(131)
plt.plot(times, eegSamples)
plt.title("EEG Signal " + str(start) + "s to " + str(end) + "s")
plt.xlabel("time")

plt.subplot(132)
# plt.title("PCA of Sliding Window Embedding")
# plt.scatter(Y[:, 0], Y[:, 1], edgecolors='none')
# plt.axis('equal')
#
# plt.subplot(133)
PD = getFunc0DPersistence(X);
print(PD)
H1 = plotDGM(PD, color = np.array([1.0, 0.0, 0.2]), label = 'H0', sz = 50, axcolor = np.array([0.8]*3))

#H1 = plotDGM(PDs[0], color = np.array([1.0, 0.0, 0.2]), label = 'H0', sz = 50, axcolor = np.array([0.8]*3))
# plt.hold(True)
# H2 = plotDGM(PDs[1], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H1', axcolor = np.array([0.8]*3))
# plt.legend(handles=[H1, H2])

plt.title('0D Persistence Diagram')

plt.show()
