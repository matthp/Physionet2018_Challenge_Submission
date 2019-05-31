# Imports
from scipy.io import loadmat
from scipy.signal import fftconvolve
import numpy as np
import os
import random
import gc as garbageCollector
from multiprocessing import Pool

'''Script to extract preprocessed signals and save to temporary numpy memap files to be used for model training'''

########################################################################################################################
# Function to compute consistent train/validation/test split by a constant random seed

def computeTrainValidationTestRecords(dataPath, foldName='Default'):
    print('////////////////////')
    print('Computing train/validation/test data split, Test results are:')

    # Define consistent mappings for the index from training data to the actual returned value to ensure shuffling
    r = random.Random()
    r.seed(29)
    requestRecordMap = list(range(994))
    r.shuffle(requestRecordMap)

    #########
    if foldName == 'Default':
        testIndices = requestRecordMap[0:200]
        validationIndices = requestRecordMap[894::]
        trainIndices = requestRecordMap[200:894]
    elif foldName == 'Auxiliary1':
        testIndices = requestRecordMap[0:100]
        validationIndices = requestRecordMap[100:200]
        trainIndices = requestRecordMap[200::]
    elif foldName == 'Auxiliary2':
        testIndices = requestRecordMap[0:100]
        validationIndices = requestRecordMap[300:400]
        trainIndices = requestRecordMap[100:300]
        trainIndices.extend(requestRecordMap[400::])
    elif foldName == 'Auxiliary3':
        testIndices = requestRecordMap[0:100]
        validationIndices = requestRecordMap[600:700]
        trainIndices = requestRecordMap[100:600]
        trainIndices.extend(requestRecordMap[700::])
    elif foldName == 'Auxiliary4':
        testIndices = requestRecordMap[0:100]
        validationIndices = requestRecordMap[894::]
        trainIndices = requestRecordMap[100:894]

    #########

    recordList = list(filter(lambda x: os.path.isdir(dataPath + x),
                        os.listdir(dataPath)))
    trainRecordList = [recordList[ind] for ind in trainIndices]
    validationRecordList = [recordList[ind] for ind in validationIndices]
    testRecordList = [recordList[ind] for ind in testIndices]

    # Small test to make sure the sets are non overlapping and use all of the data
    areUnique = len(list(set(trainRecordList).intersection(set(validationRecordList).intersection(testRecordList)))) == 0
    isComplete = len(list(set(trainRecordList).union(set(validationRecordList).union(testRecordList)).union(set(recordList)))) == len(recordList)

    print('Uniqueness Test: ' + str(areUnique) + '  Completeness Test: ' + str(isComplete))
    print('////////////////////\n')

    return trainRecordList, validationRecordList, testRecordList

def computeUnseenRecords(dataPath):
    recordList = filter(lambda x: os.path.isdir(dataPath + x),
                        os.listdir(dataPath))

    return recordList


########################################################################################################################
# Load signals or annotations from a specific file in the source files

# Convenience function to load signals
def loadSignals(recordName, dataPath):
    signals = loadmat(dataPath + recordName + '/' + recordName + '.mat')
    signals = signals['val']
    garbageCollector.collect()

    return signals

# Convenience function to load annotations
def loadAnnotations(recordName, arousalAnnotationPath, apneaHypopneaAnnotationPath, sleepWakeAnnotationPath):

    arousalAnnotations = loadmat(arousalAnnotationPath + recordName + '-arousal.mat')['data']['arousals'][0][0]
    arousalAnnotations = np.squeeze(arousalAnnotations.astype(np.int32))
    garbageCollector.collect()

    fp = np.memmap(apneaHypopneaAnnotationPath + 'apneaHypopneaAnnotation_' + str(recordName) + '.dat', dtype='int32', mode='r')
    apneaHypopneaAnnotations = np.zeros(shape=fp.shape)
    apneaHypopneaAnnotations[:] = fp[:]

    fp = np.memmap(sleepWakeAnnotationPath + 'sleepWakeAnnotation_' + str(recordName) + '.dat', dtype='int32',  mode='r')
    sleepStageAnnotations = np.zeros(shape=fp.shape)
    sleepStageAnnotations[:] = fp[:]

    return arousalAnnotations, apneaHypopneaAnnotations, sleepStageAnnotations

########################################################################################################################
# Load signals or annotations from a specific file in the source files

def extractWholeRecord(recordName,
                       dataPath,
                       arousalAnnotationPath,
                       apneaHypopneaAnnotationPath,
                       sleepWakeAnnotationPath,
                       extractAnnotations=True):
    # Keep all channels except ECG
    keepChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    if extractAnnotations:
        arousalAnnotations, apneaHypopneaAnnotations, sleepStageAnnotations = loadAnnotations(recordName, arousalAnnotationPath, apneaHypopneaAnnotationPath, sleepWakeAnnotationPath)

    signals = loadSignals(recordName, dataPath)
    signals = np.transpose(signals).astype(np.float64)

    # Apply antialiasing FIR filter to each channel and downsample to 50Hz
    filtCoeff = np.array([0.00637849379422531, 0.00543091599801427, -0.00255136650039784, -0.0123109503066702,
                          -0.0137267267561505, -0.000943230632358082, 0.0191919895027550, 0.0287148886882440,
                          0.0123598773891149, -0.0256928886371578, -0.0570987715759348, -0.0446385294777459,
                          0.0303553522906817, 0.148402006671856, 0.257171285176269, 0.301282456398562,
                          0.257171285176269, 0.148402006671856, 0.0303553522906817, -0.0446385294777459,
                          -0.0570987715759348, -0.0256928886371578, 0.0123598773891149, 0.0287148886882440,
                          0.0191919895027550, -0.000943230632358082, -0.0137267267561505, -0.0123109503066702,
                          -0.00255136650039784, 0.00543091599801427, 0.00637849379422531])

    for n in range(signals.shape[1]):
        signals[::, n] = np.convolve(signals[::, n], filtCoeff, mode='same')

    signals = signals[0::4, keepChannels]
    if extractAnnotations:
        arousalAnnotations = arousalAnnotations[0::4]
        apneaHypopneaAnnotation = apneaHypopneaAnnotations[0::4]
        sleepStageAnnotation = sleepStageAnnotations[0::4]

    garbageCollector.collect()

    # Scale SaO2 to sit between -0.5 and 0.5, a good range for input to neural network
    signals[::, 11] += -32768.0
    signals[::, 11] /= 65535.0
    signals[::, 11] -= 0.5

    # Normalize all the other channels by removing the mean and the rms in an 18 minute rolling window, using fftconvolve for computational efficiency
    # 18 minute window is used because because baseline breathing is established in 2 minute window according to AASM standards.
    # Normalizing over 18 minutes ensure a 90% overlap between the beginning and end of the baseline window
    kernel_size = (50*18*60)+1

    # Remove DC bias and scale for FFT convolution
    center = np.mean(signals, axis=0)
    scale = np.std(signals, axis=0)
    scale[scale == 0] = 1.0
    signals = (signals - center) / scale

    # Compute and remove moving average with FFT convolution
    center = np.zeros(signals.shape)
    for n in range(signals.shape[1]):
        center[::, n] = fftconvolve(signals[::, n], np.ones(shape=(kernel_size,))/kernel_size, mode='same')

    # Exclude SAO2
    center[::, 11] = 0.0
    center[np.isnan(center) | np.isinf(center)] = 0.0
    signals = signals - center

    # Compute and remove the rms with FFT convolution of squared signal
    scale = np.ones(signals.shape)
    for n in range(signals.shape[1]):
        temp = fftconvolve(np.square(signals[::, n]), np.ones(shape=(kernel_size,))/kernel_size, mode='same')

        # Deal with negative values (mathematically, it should never be negative, but fft artifacts can cause this)
        temp[temp < 0] = 0.0

        # Deal with invalid values
        invalidIndices = np.isnan(temp) | np.isinf(temp)
        temp[invalidIndices] = 0.0
        maxTemp = np.max(temp)
        temp[invalidIndices] = maxTemp

        # Finish rms calculation
        scale[::, n] = np.sqrt(temp)

    # Exclude SAO2
    scale[::, 11] = 1.0

    scale[(scale == 0) | np.isinf(scale) | np.isnan(scale)] = 1.0  # To correct for record 12 that has a zero amplitude chest signal
    signals = signals / scale

    garbageCollector.collect()

    # Convert to 32 bits
    signals = signals.astype(np.float32)

    if extractAnnotations:
        arousalAnnotations = np.expand_dims(arousalAnnotations, axis=1).astype(np.float32)
        apneaHypopneaAnnotation = np.expand_dims(apneaHypopneaAnnotation, axis=1).astype(np.float32)
        sleepStageAnnotation = np.expand_dims(sleepStageAnnotation, axis=1).astype(np.float32)
        return np.concatenate([signals, arousalAnnotations, apneaHypopneaAnnotation, sleepStageAnnotation], axis=1)
    else:
        return signals

########################################################################################################################
# Functions to extract all three datasets

def extractBatchedData(datasetName,
                       dataPath,
                       savePath,
                       foldName,
                       dataLimitInHours=7,
                       useMultiprocessing=False):

    # Get list of records and divide into train, validation and testing based on pre defined randomly divided folds
    trainRecordList, validationRecordList, testRecordList = computeTrainValidationTestRecords(dataPath, foldName=foldName)
    testRecordList = None

    if datasetName == 'Training':
        recordList = trainRecordList
        validationRecordList = None
    elif datasetName == 'Validation':
        recordList = validationRecordList
        trainRecordList = None
    else:
        raise Exception('Invalid data set name for batched data extraction!')

    # Set up
    numRecordsPerBatch = 11
    batchCount = 0
    sampleDataLimit = dataLimitInHours*3600*50

    extractDatasetFilePath = savePath + datasetName + '_'

    if useMultiprocessing:
        pool = Pool(processes=numRecordsPerBatch)

    fp = np.memmap(extractDatasetFilePath + 'data.dat', dtype='float32', mode='w+', shape=(len(recordList), sampleDataLimit, 15))

    # Extract data in batches of numRecordsPerBatch
    for n in range(0, len(recordList), numRecordsPerBatch):

        print('Extracting Batch: ' + str(batchCount))

        # Compute upper limit for this batch of records to extract
        limit = n+numRecordsPerBatch
        if (n+numRecordsPerBatch) > len(recordList):
            limit = len(recordList)

        # Process and extract batch of records
        if useMultiprocessing:
            data = pool.map(extractWholeRecord, recordList[n:limit])
        else:
            data = list(map(extractWholeRecord, recordList[n:limit]))

        # Enforce dataLimitInHours hour length with chopping / zero padding for memory usage stability and effficiency in cuDNN
        for n in range(len(data)):
            originalLength = data[n].shape[0]/(3600*50)
            if data[n].shape[0] < sampleDataLimit:
                # Zero Pad
                neededLength = sampleDataLimit - data[n].shape[0]
                extension = np.zeros(shape=(neededLength, data[n].shape[1]))
                extension[::, -3::] = -1.0
                data[n] = np.concatenate([data[n], extension], axis=0)

            elif data[n].shape[0] > sampleDataLimit:
                # Chop
                data[n] = data[n][0:sampleDataLimit, ::]

            print('Original Length: ' + str(originalLength) + '  New Length: ' + str(data[n].shape[0]/(3600*50)))

        garbageCollector.collect()

        print('Saving Batch: ' + str(batchCount))

        # Save batch of extracted records to extracted data path
        for m in range(len(data)):
            fp[(batchCount*numRecordsPerBatch)+m, ::, ::] = data[m][::, ::]
            assert isinstance(fp, np.memmap)
            fp.flush()

        garbageCollector.collect()

        batchCount += 1

    del fp

# Extract training and validation sets from backing store, YOU NEED TO COMPLETE THE REQURIED INPUTS BASED ON YOUR ENVIRONMENT
extractBatchedData(useMultiprocessing=True, datasetName='Training', foldName='Auxiliary4', dataPath, savePath)
extractBatchedData(useMultiprocessing=True, datasetName='Validation', foldName='Auxiliary4', dataPath, savePath)







