# Imports
from scipy.io import loadmat
from scipy.signal import fftconvolve
import numpy as np
import os
import random
import gc as garbageCollector
from multiprocessing import Pool
import subprocess

########################################################################################################################
# Function to compute consistent train/validation/test split by a constant random seed

def computeTrainValidationTestRecords(dataPath, foldName='Default'):
    """
    Function to compute consistent train/validation/test split by a constant random seed on the Physionet 2018 challenge data
    :return: Names of records for training, validation, testing data sets
    """
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
    """
    """
    recordList = filter(lambda x: os.path.isdir(dataPath + x),
                        os.listdir(dataPath))

    return list(recordList)

#dataPath='/mnt/HD/Physionet2018_SourceFiles/training/'
#trainRecordList, validationRecordList, testRecordList = computeTrainValidationTestRecords(dataPath, foldName='Auxiliary1')

dataPath='/mnt/HD/Physionet2018_SourceFiles/unseen/'
testRecordList = computeUnseenRecords(dataPath)

for record in testRecordList:
    print('Computing Record: ' + str(record))
    sourcePath = dataPath + record + '/' + record + '.mat'
    destinationPath = './' + record + '.mat'

    cmdString = 'cp ' + str(sourcePath) + ' ' + str(destinationPath)
    subprocess.call(cmdString, shell=True)

    cmdString = 'export PATH="/home/matthp/anaconda3/bin/:$PATH"; source activate condaVirtualEnv3; bash next.sh ' + record
    subprocess.call(cmdString, shell=True)

    os.remove(destinationPath)







