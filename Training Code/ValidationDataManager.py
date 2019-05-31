import numpy as np
import gc as garbageCollector
import torch

########################################################################################################################
# Scoring code from physionet

class Challenge2018Score:
    """Class used to compute scores for the 2018 PhysioNet/CinC Challenge.

    A Challenge2018Score object aggregates the outputs of a proposed
    classification algorithm, and calculates the area under the
    precision-recall curve, as well as the area under the receiver
    operating characteristic curve.

    After creating an instance of this class, call score_record() for
    each record being tested.  To calculate scores for a particular
    record, call record_auprc() and record_auroc().  After scoring all
    records, call gross_auprc() and gross_auroc() to obtain the scores
    for the database as a whole.
    """

    def __init__(self, input_digits=None):
        """Initialize a new scoring buffer.

        If 'input_digits' is given, it is the number of decimal digits
        of precision used in input probability values.
        """
        if input_digits is None:
            input_digits = 3
        self._scale = 10**input_digits
        self._pos_values = np.zeros(self._scale + 1, dtype=np.int64)
        self._neg_values = np.zeros(self._scale + 1, dtype=np.int64)
        self._record_auc = {}

    def score_record(self, truth, predictions, record_name=None):
        """Add results for a given record to the buffer.

        'truth' is a vector of arousal values: zero for non-arousal
        regions, positive for target arousal regions, and negative for
        unscored regions.

        'predictions' is a vector of probabilities produced by the
        classification algorithm being tested.  This vector must be
        the same length as 'truth', and each value must be between 0
        and 1.

        If 'record_name' is specified, it can be used to obtain
        per-record scores afterwards, by calling record_auroc() and
        record_auprc().
        """
        # Check if length is correct
        if len(predictions) != len(truth):
            raise ValueError("length of 'predictions' does not match 'truth'")

        # Compute the histogram of all input probabilities
        b = self._scale + 1
        r = (-0.5 / self._scale, 1.0 + 0.5 / self._scale)
        all_values = np.histogram(predictions, bins=b, range=r)[0]

        # Check if input contains any out-of-bounds or NaN values
        # (which are ignored by numpy.histogram)
        if np.sum(all_values) != len(predictions):
            raise ValueError("invalid values in 'predictions'")

        # Compute the histogram of probabilities within arousal regions
        pred_pos = predictions[truth > 0]
        pos_values = np.histogram(pred_pos, bins=b, range=r)[0]

        # Compute the histogram of probabilities within unscored regions
        pred_ign = predictions[truth < 0]
        ign_values = np.histogram(pred_ign, bins=b, range=r)[0]

        # Compute the histogram of probabilities in non-arousal regions,
        # given the above
        neg_values = all_values - pos_values - ign_values

        self._pos_values += pos_values
        self._neg_values += neg_values

        if record_name is not None:
            self._record_auc[record_name] = self._auc(pos_values, neg_values)

    def _auc(self, pos_values, neg_values):
        # Calculate areas under the ROC and PR curves by iterating
        # over the possible threshold values.

        # At the minimum threshold value, all samples are classified as
        # positive, and thus TPR = 1 and TNR = 0.
        tp = np.sum(pos_values)
        fp = np.sum(neg_values)
        tn = fn = 0
        tpr = 1
        tnr = 0
        if tp == 0 or fp == 0:
            # If either class is empty, scores are undefined.
            return (float('nan'), float('nan'))
        ppv = float(tp) / (tp + fp)
        auroc = 0
        auprc = 0

        # As the threshold increases, TP decreases (and FN increases)
        # by pos_values[i], while TN increases (and FP decreases) by
        # neg_values[i].
        for (n_pos, n_neg) in zip(pos_values, neg_values):
            tp -= n_pos
            fn += n_pos
            fp -= n_neg
            tn += n_neg
            tpr_prev = tpr
            tnr_prev = tnr
            ppv_prev = ppv
            tpr = float(tp) / (tp + fn)
            tnr = float(tn) / (tn + fp)
            if tp + fp > 0:
                ppv = float(tp) / (tp + fp)
            else:
                ppv = ppv_prev
            auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5
            auprc += (tpr_prev - tpr) * ppv_prev
        return (auroc, auprc)

########################################################################################################################
# Data loading code
class asyncValidationDataLoader:
    def __init__(self, extractedDataPath, numRecords=100, reductionFactor=100, numChannels=12):

        self.extractedDataPath = extractedDataPath
        self.numRecords = numRecords
        self.reductionFactor = reductionFactor
        self.recordLength = 3600*7*reductionFactor
        self.numChannels = numChannels

        # Load the data
        self._load()

    def _load(self):
        print('Loading Validation Data')

        # Load data
        fp = np.memmap(self.extractedDataPath + '/Validation_data.dat', dtype='float32', mode='r', shape=(self.numRecords, self.recordLength, self.numChannels+3))
        self.feats = torch.FloatTensor(np.array(fp[::, ::, 0:self.numChannels])).permute(0, 2, 1).detach().contiguous()
        self.arousalTargs = np.array(fp[::, ::self.reductionFactor, self.numChannels]).reshape(-1).astype(np.int32)
        self.apneaTargs = np.array(fp[::, ::self.reductionFactor, self.numChannels+1]).reshape(-1).astype(np.int32)
        self.sleepTargs = np.array(fp[::, ::self.reductionFactor, self.numChannels+2]).reshape(-1).astype(np.int32)
        del fp
        garbageCollector.collect()
        print('Done Loading Validation Data')

    def __len__(self):
        return self.numRecords

    def __getitem__(self, item):
        return self.feats[item, ::, ::].clone().view(1, self.numChannels, self.recordLength)

    def getTargets(self):
        return (np.copy(self.arousalTargs), np.copy(self.apneaTargs), np.copy(self.sleepTargs))

########################################################################################################################
# Validation evaluation code

# Computes the overall accuracy on the validation data
def evalValidationAccuracy(model, validationData):
    
    print('Predicting on Validation Data')
    
    assert isinstance(validationData, asyncValidationDataLoader)
    arousalTargs, apneaTargs, sleepTargs = validationData.getTargets()

    model.eval()

    realArousalPredictions = [None]*len(validationData)
    realWakePredictions = [None]*len(validationData)
    realApneaPredictions = [None]*len(validationData)

    # Proceed in batches of batchSize
    for n in range(len(validationData)):
        feats = validationData[n]

        # Run model
        arousalOutputs, apneaHypopneaOutputs, sleepStageOutputs = model(feats.cuda())

        # Reshape and pull back to cpu
        arousalOutputs = arousalOutputs.data.cpu().permute(0, 2, 1).contiguous().view(-1, 2)
        apneaHypopneaOutputs = apneaHypopneaOutputs.data.cpu().permute(0, 2, 1).contiguous().view(-1, 2)
        sleepStageOutputs = sleepStageOutputs.data.cpu().permute(0, 2, 1).contiguous().view(-1, 2)

        # Store the probabilities for each target
        realArousalPredictions[n] = np.squeeze(arousalOutputs[::, 1].numpy())
        realApneaPredictions[n] = np.squeeze(apneaHypopneaOutputs[::, 1].numpy())
        realWakePredictions[n] = np.squeeze(sleepStageOutputs[::, 0].numpy())

        garbageCollector.collect()

    print('Evaluating Validation Performance')
    
    # Compact probability predictions
    realArousalPredictions = np.concatenate(realArousalPredictions)
    realApneaPredictions = np.concatenate(realApneaPredictions)
    realWakePredictions = np.concatenate(realWakePredictions)

    # Remove ignore index from arousal
    realArousalPredictions = realArousalPredictions[arousalTargs >= 0]
    arousalTargs = arousalTargs[arousalTargs >= 0]
    realArousalPredictions[realArousalPredictions > 0.999] = 0.999
    realArousalPredictions[realArousalPredictions < 0.001] = 0.001

    # Remove ignore index from apnea
    realApneaPredictions = realApneaPredictions[apneaTargs >= 0]
    apneaTargs = apneaTargs[apneaTargs >= 0]
    realApneaPredictions[realApneaPredictions > 0.999] = 0.999
    realApneaPredictions[realApneaPredictions < 0.001] = 0.001

    # Remove ignore index from sleep
    realWakePredictions = realWakePredictions[sleepTargs >= 0]
    sleepTargs = sleepTargs[sleepTargs >= 0]
    realWakePredictions[realWakePredictions > 0.999] = 0.999
    realWakePredictions[realWakePredictions < 0.001] = 0.001

    # Compute performance stats
    scorer = Challenge2018Score()
    scorer.score_record(truth=arousalTargs, predictions=realArousalPredictions)
    arousalAUC, arousalAP = scorer._auc(scorer._pos_values, scorer._neg_values)

    scorer = Challenge2018Score()
    scorer.score_record(truth=apneaTargs, predictions=realApneaPredictions)
    apneaAUC, apneaAP = scorer._auc(scorer._pos_values, scorer._neg_values)

    scorer = Challenge2018Score()
    scorer.score_record(truth=1-sleepTargs, predictions=realWakePredictions)
    wakeAUC, wakeAP = scorer._auc(scorer._pos_values, scorer._neg_values)

    garbageCollector.collect()

    return arousalAUC, arousalAP, apneaAUC, apneaAP, wakeAUC, wakeAP



