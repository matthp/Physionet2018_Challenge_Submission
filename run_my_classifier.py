import torch
import numpy as np
from ExtractAllData import extractWholeRecord
import sys
from ModelDefinition import Sleep_model_MultiTarget
import os
import gc

modelPaths = ['./TrainedModels/Model_Auxiliary1_stateDict.pkl', './TrainedModels/Model_Auxiliary2_stateDict.pkl',
              './TrainedModels/Model_Auxiliary3_stateDict.pkl', './TrainedModels/Model_Auxiliary4_stateDict.pkl']

testInPython = False
if testInPython:
    recordList = ['tr03-1371']
    cudaAvailable = True
else:
    recordList = sys.argv[1:]
    cudaAvailable = torch.cuda.is_available()

if __name__ == '__main__':
    for record in recordList:
        processedSignal = extractWholeRecord(recordName=str(record),
                                             extractAnnotations=False,
                                             dataPath='./',
                                             arousalAnnotationPath=None,
                                             apneaHypopneaAnnotationPath=None,
                                             sleepWakeAnnotationPath=None,
                                             dataInDirectory=False)
        arousalPredictions = np.zeros(processedSignal.shape[0]*4,)
        processedSignal = torch.Tensor(processedSignal).unsqueeze(0).permute(0, 2, 1)

        for foldIndex in range(len(modelPaths)):
            model = Sleep_model_MultiTarget()
            model.load_state_dict(torch.load(modelPaths[foldIndex]))
            model = model.eval()

            if cudaAvailable:
                model = model.cuda()
                processedSignal = processedSignal.cuda()

            with torch.no_grad():
                tempArousalPredictions, tempApneaHypopneaPredictions, tempSleepWakePredictions = model(processedSignal)
                tempArousalPredictions = torch.nn.functional.upsample(tempArousalPredictions, scale_factor=200, mode='linear')
                tempArousalPredictions = tempArousalPredictions[0, 1, ::].detach().cpu().numpy()
                arousalPredictions += tempArousalPredictions

            # Clean up
            del tempArousalPredictions
            del tempApneaHypopneaPredictions
            del tempSleepWakePredictions
            del model
            torch.cuda.empty_cache()
            gc.collect()

        arousalPredictions /= float(len(modelPaths))

        output_file = os.path.basename(record) + '.vec'
        np.savetxt(output_file, arousalPredictions, fmt='%.3f')





