import torch
import torch.optim as optim
import os
import gc as garbageCollector

from ModelDefinition import Sleep_model_MultiTarget

from TrainingDataManager import asyncTrainingDataLoader
from ValidationDataManager import asyncValidationDataLoader, evalValidationAccuracy

import timeit

torch.backends.cudnn.benchmark = True

'''
Script which trains the model and saves it based on validation performance
'''

########################################################################################################################
# Settings

# Control parameters
gpuSetting = 1     # Which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuSetting)

foldName = 'Auxiliary1'

numChannels = 12

########################################################################################################################
# Training and validation data asynchronous managers

extractedDataPath = UPDATE BASED ON YOUR ENVIRONMENT
trainingData = asyncTrainingDataLoader(extractedDataPath=extractedDataPath, reductionFactor=50, numChannels=numChannels)
validationData = asyncValidationDataLoader(extractedDataPath=extractedDataPath, numRecords=100, reductionFactor=50, numChannels=numChannels)
garbageCollector.collect()

########################################################################################################################

# Trains the model
def trainModel(model):

    optimizer = optim.Adam(model.parameters())

    arousalCriterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=True).cuda()
    apneaCriterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=True).cuda()
    wakeCriterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=True).cuda()

    bestArousalAUC = 0.5
    bestArousalAP = 0.0

    bestApneaAUC = 0.5
    bestApneaAP = 0.0

    bestWakeAUC = 0.5
    bestWakeAP = 0.0

    i_epoch = 0

    numBatchesPerEpoch = 100

    while True:
        # Put in train mode
        trainingStartTime = timeit.default_timer()

        model.train()
        runningLoss = 0.0

        for n in range(numBatchesPerEpoch):

            # Fetch pre-loaded batch from asynchronous data loader
            batchFeats, batchArousalTargs, batchApneaTargs, batchWakeTargs = trainingData.getNextItem()

            # Send batch to GPU
            batchFeats = batchFeats.cuda()
            batchArousalTargs = batchArousalTargs.cuda()
            batchApneaTargs = batchApneaTargs.cuda()
            batchWakeTargs = batchWakeTargs.cuda()

            # Compute the network outputs on the batch
            arousalOutputs, apneaHypopneaOutputs, sleepStageOutputs = model(batchFeats)

            # Compute the losses
            arousalOutputs = arousalOutputs.permute(0, 2, 1).contiguous().view(-1, 2)
            apneaHypopneaOutputs = apneaHypopneaOutputs.permute(0, 2, 1).contiguous().view(-1, 2)
            sleepStageOutputs = sleepStageOutputs.permute(0, 2, 1).contiguous().view(-1, 2)

            arousalLoss = arousalCriterion(arousalOutputs, batchArousalTargs)
            apneaHypopneaLoss = apneaCriterion(apneaHypopneaOutputs, batchApneaTargs)
            sleepStageLoss = wakeCriterion(sleepStageOutputs, batchWakeTargs)

            loss = ((2*arousalLoss) + apneaHypopneaLoss + sleepStageLoss) / 4.0

            # Backpropagation
            loss.backward()

            # Perform one optimization step
            currentBatchLoss = loss.data.cpu().numpy()
            runningLoss += currentBatchLoss

            optimizer.step()
            optimizer.zero_grad()

        i_epoch += 1

        trainingEndTime = timeit.default_timer()

        # Get validation accuracy
        arousalAUC, arousalAP, apneaAUC, apneaAP, wakeAUC, wakeAP = evalValidationAccuracy(model, validationData=validationData)

        validationEndTime = timeit.default_timer()

        print('////////////////////')
        print('Epoch Number: ' + str(i_epoch) + '  Training Time: ' + str(trainingEndTime - trainingStartTime) + '  Validation Time: ' + str(validationEndTime - trainingEndTime))
        print('Average Training Loss: ' + str(runningLoss / float(numBatchesPerEpoch)))
        print('arousalAUC: ' + "{:.3f}".format(arousalAUC) + '  arousalAP: ' + "{:.3f}".format(arousalAP) + \
              '  apneaAUC: ' + "{:.3f}".format(apneaAUC) + '  apneaAP: ' + "{:.3f}".format(apneaAP) +
              '  wakeAUC: ' + "{:.3f}".format(wakeAUC), '  wakeAP: ' + "{:.3f}".format(wakeAP))

        # If the validation accuracy improves, print out training and validation accuracy values and checkpoint the model
        if (arousalAUC > bestArousalAUC) \
                or (arousalAP > bestArousalAP) \
                or (apneaAUC > bestApneaAUC) \
                or (apneaAP > bestApneaAP) \
                or (wakeAUC > bestWakeAUC) \
                or (wakeAP > bestWakeAP):
            print('Model Improved: '
                  + ' arousalAUC: ' + "{:.3f}".format(arousalAUC - bestArousalAUC)
                  + '  arousalAP: ' + "{:.3f}".format(arousalAP - bestArousalAP)
                  + '  apneaAUC: ' + "{:.3f}".format(apneaAUC - bestApneaAUC)
                  + '  apneaAP: ' + "{:.3f}".format(apneaAP - bestApneaAP)
                  + '  wakeAUC: ' + "{:.3f}".format(wakeAUC - bestWakeAUC)
                  + '  wakeAP: ' + "{:.3f}".format(wakeAP - bestWakeAP))

            if (arousalAUC > bestArousalAUC):
                bestArousalAUC = arousalAUC
            if (arousalAP > bestArousalAP):
                bestArousalAP = arousalAP
            if (apneaAUC > bestApneaAUC):
                bestApneaAUC = apneaAUC
            if (apneaAP > bestApneaAP):
                bestApneaAP = apneaAP
            if (wakeAUC > bestWakeAUC):
                bestWakeAUC = wakeAUC
            if (wakeAP > bestWakeAP):
                bestWakeAP = wakeAP

            f = open('./Models/Auxiliary1' + '/checkpointModel_' + str(i_epoch)
                     + '_' + "{:.3f}".format(arousalAUC)
                     + '_' + "{:.3f}".format(arousalAP)
                     + '_' + "{:.3f}".format(apneaAUC)
                     + '_' + "{:.3f}".format(apneaAP)
                     + '_' + "{:.3f}".format(wakeAUC)
                     + '_' + "{:.3f}".format(wakeAP) + '.pkl', 'wb')
            torch.save(model, f)
            f.close()

########################################################################################################################
# Create and train model

# Create new model and optimizer
model = Sleep_model_MultiTarget(numSignals=numChannels)
model.cuda()
model.train()

# Train and evaluate the model
model = trainModel(model=model)







