# Imports
from scipy.io import loadmat
from scipy.signal import fftconvolve
import numpy as np
import gc as garbageCollector

########################################################################################################################
# Load signals from a specific file in the source files

# Convenience function to load signals
def loadSignals(recordName, dataPath, dataInDirectory):

    if dataInDirectory:
        signals = loadmat(dataPath + recordName + '/' + recordName + '.mat')
    else:
        signals = loadmat(dataPath + recordName + '.mat')
    signals = signals['val']
    garbageCollector.collect()

    return signals

########################################################################################################################
# Loads and prepossesses signals from a specific record

def extractWholeRecord(recordName,
                       dataPath='PATH/',
                       dataInDirectory=True):
    # Keep all channels except ECG
    keepChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    signals = loadSignals(recordName, dataPath, dataInDirectory)
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

    return signals






