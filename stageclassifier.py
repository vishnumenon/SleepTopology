import mne

with open('Sleep-EDF-DB/RECORDS') as edfs, open('Sleep-EDF-DB/HYPNOGRAMS') as annots:
    for edf, ann in zip(edfs, annots):
        edf = edf.strip()
        ann = ann.strip()
        if edf[:7] != ann[:7]:
            raise Exception('EDF-Annotation Mismatch')
        raw = mne.io.read_raw_edf('Sleep-EDF-DB/' + edf, annotmap='Sleep-EDF-DB/annotmap', annot='Sleep-EDF-DB/' + ann, preload=True)
        print(raw.ch_names)
        chan = raw.copy().pick_channels(['EEG Fpz-Cz', 'STI 014'])
        events = mne.find_events(raw)
