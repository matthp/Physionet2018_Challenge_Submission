import numpy as np
import torch
import multiprocessing as mltp
import gc as garbageCollector
from random import Random

########################################################################################################################
#

def remapLabels(batchArousalTargs, batchApneaHypopneaTargs, batchSleepStageTargs):

    # Remap bin 3, 6, 12, 21 to bin 0
    remapIndices = batchSleepStageTargs == -1
    batchArousalTargs[remapIndices] = -1
    batchApneaHypopneaTargs[remapIndices] = -1

    # Remap bin 1 to 2
    remapIndices = (batchArousalTargs == -1) & (batchApneaHypopneaTargs == -1)
    batchSleepStageTargs[remapIndices] = 1

    # Remap bin 5 to 4
    remapIndices = (batchArousalTargs == -1) & (batchApneaHypopneaTargs == 0)
    batchSleepStageTargs[remapIndices] = 0

    # Remap bin 7 to 8
    remapIndices = (batchArousalTargs == -1) & (batchApneaHypopneaTargs == 1)
    batchSleepStageTargs[remapIndices] = 1

    return batchArousalTargs, batchApneaHypopneaTargs, batchSleepStageTargs

def loadRecords(extractedDataPath, recordIndices, reductionFactor, numRecords, recordLength, numChannels):
    fp = np.memmap(extractedDataPath + '/Training_data.dat', dtype='float32', mode='r', shape=(numRecords, recordLength, numChannels+3))

    feats = torch.FloatTensor(np.array(fp[recordIndices, ::, 0:numChannels])).permute(0, 2, 1).detach().contiguous()
    arousalTargs = torch.LongTensor(np.array(fp[recordIndices, 0::reductionFactor, numChannels]).astype(np.int32)).contiguous()
    apneaTargs = torch.LongTensor(np.array(fp[recordIndices, 0::reductionFactor, numChannels+1]).astype(np.int32)).contiguous()
    sleepTargs = torch.LongTensor(np.array(fp[recordIndices, 0::reductionFactor, numChannels+2]).astype(np.int32)).contiguous()

    del fp

    # if np.any(np.isnan(feats) | np.isinf(feats)):
    #     print('Invalid Values For Training Record: ' + str(index))
    #     return None, None, None, None

    arousalTargs, apneaTargs, sleepTargs = remapLabels(arousalTargs, apneaTargs, sleepTargs)

    return feats, arousalTargs, apneaTargs, sleepTargs

def fillBatchQueue(resultQueue, extractedDataPath, reductionFactor, numRecords, recordLength, numChannels, seed):
    rng = Random()
    rng.seed(seed)

    recordList = list(range(numRecords))

    while True:
        rng.shuffle(recordList)
        for n in range(numRecords//30):
            feats, arousalTargs, apneaTargs, sleepTargs = loadRecords(extractedDataPath, recordList[(n*30):((n+1)*30)], reductionFactor, numRecords, recordLength, numChannels)
            resultQueue.put((feats, arousalTargs, apneaTargs, sleepTargs))

class asyncTrainingDataLoader:
    def __init__(self, extractedDataPath, numRecords=694, reductionFactor=100, numChannels=12):

        self.extractedDataPath = extractedDataPath
        self.numRecords = numRecords
        self.reductionFactor = reductionFactor
        self.recordLength = 3600*7*reductionFactor
        self.numChannels = numChannels

        self.seed = 51

        self.CurrentFeats = None
        self.CurrentArousalTargs = None
        self.CurrentApneaTargs = None
        self.CurrentSleepTargs = None
        self.CurrentItemIndex = 0

        # Multiprocessing properties for managing the batch queue filling
        self.resultManager = mltp.Manager()
        self.resultQueue = self.resultManager.Queue(maxsize=1)

        self.batchFillProcess = mltp.Process(target=fillBatchQueue, args=(self.resultQueue, self.extractedDataPath, self.reductionFactor, self.numRecords, self.recordLength, self.numChannels, self.seed))
        self.batchFillProcess.start()

    def loadNextBatch(self):
        result = None

        while result is None:
            try:
                result = self.resultQueue.get(timeout=45)
            except Exception as e:
                # Batch filling is hung, kill the process
                print('Batch Filling Hung')
                self.batchFillProcess.terminate()

                # Restart everything
                self.resultManager = mltp.Manager()
                self.resultQueue = self.resultManager.Queue(maxsize=1)

                self.seed += 7

                self.batchFillProcess = mltp.Process(target=fillBatchQueue, args=(self.resultQueue, self.extractedDataPath, self.reductionFactor, self.numRecords, self.recordLength, self.numChannels, self.seed))
                garbageCollector.collect()
                self.batchFillProcess.start()

                result = None

        self.CurrentFeats = result[0]
        self.CurrentArousalTargs = result[1]
        self.CurrentApneaTargs = result[2]
        self.CurrentSleepTargs = result[3]

    def getNextItem(self):
        if (self.CurrentItemIndex >= 30) or (self.CurrentFeats is None):
            self.loadNextBatch()
            self.CurrentItemIndex = 0

        feats = self.CurrentFeats[self.CurrentItemIndex, ::, ::].view(1, self.numChannels, self.recordLength)
        arousalTargs = self.CurrentArousalTargs[self.CurrentItemIndex, ::].view(-1)
        apneaTargs = self.CurrentApneaTargs[self.CurrentItemIndex, ::].view(-1)
        sleepTargs = self.CurrentSleepTargs[self.CurrentItemIndex, ::].view(-1)

        self.CurrentItemIndex += 1

        return (feats, arousalTargs, apneaTargs, sleepTargs)

