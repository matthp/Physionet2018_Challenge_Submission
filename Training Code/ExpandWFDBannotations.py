from scipy.io import loadmat
import numpy as np
import os
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
from wfdb import rdann

"""Expands the sleep stage and apnea/hypopnea annotation files from Physionet into numpy memmaps with one annotation per sample"""


########################################################################################################################
# SET THESE BASED ON YOUR ENVIRONMENT
arousalDataPath = 'path/to/arousal/.mat/annotations'
dataPath = 'path/to/WFDB/annotation/files/from/physionet'

# Paths to save the expanded annotations
sleepStageFilePath = 'path/to/save/sleepStage/annotations/'
sleepWakeFilePath = 'path/to/save/sleeWake/annotations/'
apneaHypopneaFilePath = 'path/to/save/apneaHypopnea/annotations/'
obstructiveApneaHypopneaFilePath = 'path/to/save/obstructiveApneaHypopnea/annotations/'
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load records list and define data extraction mode
recordList = filter(lambda x: os.path.isdir(dataPath + x), os.listdir(dataPath))

# apneaHypopneaMode = 'apneaHypopnea' was used for the final submitted models, 'obstructiveApneaHypopnea' was investigated but not used in the final work
apneaHypopneaMode = 'apneaHypopnea' # apneaHypopnea is referred to all types of Apnea and Hypopnea
#apneaHypopneaMode = 'obstructiveApneaHypopnea' # obstructiveApneaHypopnea is referred only obstructive Apnea and Hypopnea

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convenience function to load annotations
def loadAnnotations(recordName):

    arousalAnnotations = loadmat(arousalDataPath + recordName + '-arousal.mat')['data']['arousals'][0][0]
    arousalAnnotations = np.squeeze(arousalAnnotations.astype(np.int32))

    sleepZoneAnnotations = rdann(dataPath + recordName + '/' + recordName, 'arousal')
    return arousalAnnotations, sleepZoneAnnotations



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class classificationDataset(Dataset):
    def __init__(self):
        super(classificationDataset, self).__init__()

    def __len__(self):
        return 994

    def __getitem__(self, item):

        ind = recordList[item]
        ind = str(ind)
        arousalAnnotations, sleepZoneAnnotations = loadAnnotations(ind)


        numSamples = len(sleepZoneAnnotations.sample)

        apneaHypopneaAnn = np.zeros(shape=(len(arousalAnnotations)))
        sleepWakeAnnotation = np.zeros(shape=(len(arousalAnnotations)))
        sleepStageAnnotation = np.zeros(shape=(len(arousalAnnotations)))
        startObsApnea = np.zeros(shape=numSamples)
        endObsApnea = np.zeros(shape=numSamples)
        startCntApnea = np.zeros(shape=numSamples)
        endCntApnea = np.zeros(shape=numSamples)
        startMixApnea = np.zeros(shape=numSamples)
        endMixApnea = np.zeros(shape=numSamples)
        startHypopnea = np.zeros(shape=numSamples)
        endHypopnea = np.zeros(shape=numSamples)
        startApnea = np.zeros(shape=numSamples)
        endApnea = np.zeros(shape=numSamples)



        counter = 0
        for dataPoints in range(0, numSamples, 1):

            # Extracting Hypopnea annotations
            if sleepZoneAnnotations.aux_note[dataPoints] == '(resp_hypopnea':
                startHypopnea[counter] = sleepZoneAnnotations.sample[dataPoints]
            elif sleepZoneAnnotations.aux_note[dataPoints] == 'resp_hypopnea)':
                endHypopnea[counter] = sleepZoneAnnotations.sample[dataPoints]

            # Extracting mode-based Apnea annotations
            if apneaHypopneaMode == 'apneaHypopnea':
                if (sleepZoneAnnotations.aux_note[dataPoints] == '(resp_obstructiveapnea') or (sleepZoneAnnotations.aux_note[dataPoints] == '(resp_centralapnea')\
                        or (sleepZoneAnnotations.aux_note[dataPoints] == '(resp_mixedapnea'):
                        startApnea[counter] = sleepZoneAnnotations.sample[dataPoints]
                elif (sleepZoneAnnotations.aux_note[dataPoints] == 'resp_obstructiveapnea)') or (sleepZoneAnnotations.aux_note[dataPoints] == 'resp_centralapnea)')\
                        or (sleepZoneAnnotations.aux_note[dataPoints] == 'resp_mixedapnea)'):
                        endApnea[counter] = sleepZoneAnnotations.sample[dataPoints]

            elif apneaHypopneaMode == 'obstructiveApneaHypopnea':
                if sleepZoneAnnotations.aux_note[dataPoints] == '(resp_obstructiveapnea':
                    startObsApnea[counter] = sleepZoneAnnotations.sample[dataPoints]
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'resp_obstructiveapnea)':
                    endObsApnea[counter] = sleepZoneAnnotations.sample[dataPoints]
                elif sleepZoneAnnotations.aux_note[dataPoints] == '(resp_centralapnea':
                    startCntApnea[counter] = sleepZoneAnnotations.sample[dataPoints]
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'resp_centralapnea)':
                    endCntApnea[counter] = sleepZoneAnnotations.sample[dataPoints]
                elif sleepZoneAnnotations.aux_note[dataPoints] == '(resp_mixedapnea':
                    startMixApnea[counter] = sleepZoneAnnotations.sample[dataPoints]
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'resp_mixedapnea)':
                    endMixApnea[counter] = sleepZoneAnnotations.sample[dataPoints]



            # Extracting sleep stage annotation
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'W':  # Wake-up stage
                    W = sleepZoneAnnotations.sample[dataPoints]
                    sleepStageAnnotation[W] = 1
                    sleepWakeAnnotation[W] = 2
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'R':    # REM stage
                    R = sleepZoneAnnotations.sample[dataPoints]
                    sleepStageAnnotation[R] = 2
                    sleepWakeAnnotation[R] = 1
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'N1':   # NREM stage 1
                    N1 = sleepZoneAnnotations.sample[dataPoints]
                    sleepStageAnnotation[N1] = 3
                    sleepWakeAnnotation[N1] = 1
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'N2':   # NREM stage 2
                    N2 = sleepZoneAnnotations.sample[dataPoints]
                    sleepStageAnnotation[N2] = 4
                    sleepWakeAnnotation[N2] = 1
                elif sleepZoneAnnotations.aux_note[dataPoints] == 'N3':   # NREM stage 3
                    N3 = sleepZoneAnnotations.sample[dataPoints]
                    sleepStageAnnotation[N3] = 5
                    sleepWakeAnnotation[N3] = 1

            counter += 1


        # Preparing model-based Apnea and Hypopnea annotation array
        if apneaHypopneaMode == 'apneaHypopnea':
            startApnea = startApnea[startApnea > 0]
            endApnea = endApnea[endApnea > 0]

            startApnea = np.squeeze(startApnea.astype(np.int32))
            endApnea = np.squeeze(endApnea.astype(np.int32))

            if np.size(startApnea) > 1:
                for numApnea in range(len(startApnea)):
                    apneaHypopneaAnn[startApnea[numApnea]:endApnea[numApnea]] = 1  # Apnea region
            elif np.size(startApnea) == 1:
                apneaHypopneaAnn[startApnea:endApnea] = 1

        elif apneaHypopneaMode == 'obstructiveApneaHypopnea':
            startObsApnea = startObsApnea[startObsApnea > 0]
            endObsApnea = endObsApnea[endObsApnea > 0]
            startCntApnea = startCntApnea[startCntApnea > 0]
            endCntApnea = endCntApnea[endCntApnea > 0]
            startMixApnea = startMixApnea[startMixApnea > 0]
            endMixApnea = endMixApnea[endMixApnea > 0]


            startObsApnea = np.squeeze(startObsApnea.astype(np.int32))
            endObsApnea = np.squeeze(endObsApnea.astype(np.int32))
            startCntApnea = np.squeeze(startCntApnea.astype(np.int32))
            endCntApnea = np.squeeze(endCntApnea.astype(np.int32))
            startMixApnea = np.squeeze(startMixApnea.astype(np.int32))
            endMixApnea = np.squeeze(endMixApnea.astype(np.int32))

            if np.size(startObsApnea) > 1:
                for numApnea in range(len(startObsApnea)):
                    apneaHypopneaAnn[startObsApnea[numApnea]:endObsApnea[numApnea]] = 1  # Obstructive Apnea region
            elif np.size(startObsApnea) == 1:
                apneaHypopneaAnn[startObsApnea:endObsApnea] = 1

            if np.size(startCntApnea) > 1:
                for numApnea in range(len(startCntApnea)):
                    apneaHypopneaAnn[startCntApnea[numApnea]:endCntApnea[numApnea]] = 3  # Central Apnea region
            elif np.size(startCntApnea) == 1:
                apneaHypopneaAnn[startCntApnea:endCntApnea] = 3

            if np.size(startMixApnea) > 1:
                for numApnea in range(len(startMixApnea)):
                    apneaHypopneaAnn[startMixApnea[numApnea]:endMixApnea[numApnea]] = 4  # Mixed Apnea region
            elif np.size(startMixApnea) == 1:
                apneaHypopneaAnn[startMixApnea:endMixApnea] = 4


        startHypopnea = startHypopnea[startHypopnea > 0]
        endHypopnea = endHypopnea[endHypopnea > 0]

        startHypopnea = np.squeeze(startHypopnea.astype(np.int32))
        endHypopnea = np.squeeze(endHypopnea.astype(np.int32))

        if np.size(startHypopnea) > 1:
            for numHypopnea in range(len(startHypopnea)):
                apneaHypopneaAnn[startHypopnea[numHypopnea]:endHypopnea[numHypopnea]] = 2  # Hypopnea region
        elif np.size(startHypopnea) == 1:
            apneaHypopneaAnn[startHypopnea:endHypopnea] = 2

        # Preparing sleep-stage and sleep-wake mode annotation arrays
        for index in range(len(sleepWakeAnnotation)):
            if not sleepWakeAnnotation[index]:
                sleepWakeAnnotation[index] = sleepWakeAnnotation[index-1]

        for index in range(len(sleepStageAnnotation)):
            if not sleepStageAnnotation[index]:
                sleepStageAnnotation[index] = sleepStageAnnotation[index - 1]

        # Usually the initial sleep stage is not labelled, this interval is considered an undefined stage
        sleepWakeAnnotation[sleepWakeAnnotation == 0] = -1 # undefined
        sleepWakeAnnotation[sleepWakeAnnotation == 2] = 0  # wake

        sleepStageAnnotation[sleepStageAnnotation == 0] = 6



        return apneaHypopneaAnn, sleepWakeAnnotation, sleepStageAnnotation



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ds = classificationDataset()

# Loop over each annotation file and save its expanded sleep/wake, sleep stage and apnea/hypopnea annotations
for n in range(len(ds)):
    apneaHypopneaAnn, sleepWakeAnnotation, sleepStageAnnotation = ds[n]


    if apneaHypopneaMode == 'obstructiveApneaHypopnea':
        fp = np.memmap(obstructiveApneaHypopneaFilePath + 'obstructiveApneaHypopneaAnnotation_' + str(recordList[n]) + '.dat', dtype='int32', mode='w+',
                       shape=apneaHypopneaAnn.shape)
        fp[:] = apneaHypopneaAnn[:]
        del fp
    elif apneaHypopneaMode == 'apneaHypopnea':
        fp = np.memmap(apneaHypopneaFilePath + 'apneaHypopneaAnnotation_' + str(recordList[n]) + '.dat', dtype='int32', mode='w+',
                       shape=apneaHypopneaAnn.shape)
        fp[:] = apneaHypopneaAnn[:]
        del fp

    fp = np.memmap(sleepWakeFilePath + 'sleepWakeAnnotation_' + str(recordList[n]) + '.dat', dtype='int32', mode='w+',
                   shape=sleepWakeAnnotation.shape)
    fp[:] = sleepWakeAnnotation[:]
    del fp

    fp = np.memmap(sleepStageFilePath + 'sleepStageAnnotation_' + str(recordList[n]) + '.dat', dtype='int32', mode='w+',
                   shape=sleepStageAnnotation.shape)
    fp[:] = sleepStageAnnotation[:]
    del fp

    print('Annotations are extracted for record ' + str(n))



























