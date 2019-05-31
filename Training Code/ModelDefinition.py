import torch
import torch.nn as nn
import torch.nn.functional as Functional

########################################################################################################################
# Functional blocks
class Normalizer(nn.Module):
    def __init__(self, numChannels, momentum=0.985, affine=True, channelNorm=True):
        super(Normalizer, self).__init__()

        self.momentum = momentum
        self.numChannels = numChannels
        self.affine = affine
        self.channelNorm = channelNorm

        self.movingAverage = torch.zeros(1, numChannels, 1).cuda()
        self.movingVariance = torch.ones(1, numChannels, 1).cuda()

        if affine:
            self.BatchNormScale = nn.Parameter(torch.ones(1, numChannels, 1)).cuda()
            self.BatchNormBias = nn.Parameter(torch.zeros(1, numChannels, 1)).cuda()

    def forward(self, x):

        # Apply channel wise normalization
        if self.channelNorm:
            x = (x-torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 0.00001)

        # If in training mode, update moving per channel statistics
        if self.training:
            newMean = torch.mean(x, dim=2, keepdim=True)
            self.movingAverage = ((self.momentum * self.movingAverage) + ((1 - self.momentum) * newMean)).detach()
            x = x - self.movingAverage

            newVariance = torch.mean(torch.pow(x, 2), dim=2, keepdim=True)
            self.movingVariance = ((self.momentum * self.movingVariance) + ((1 - self.momentum) * newVariance)).detach()
            x = x / (torch.sqrt(self.movingVariance) + 0.00001)
        else:
            x = (x - self.movingAverage) / (torch.sqrt(self.movingVariance) + 0.00001)

        # Apply batch norm affine transform
        if self.affine:
            x = x * torch.abs(self.BatchNormScale)
            x = x + self.BatchNormBias

        return x

class SeperableDenseNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernelSize,
                 groups=1, dilation=1, channelNorm=True):
        super(SeperableDenseNetUnit, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernelSize = kernelSize
        self.groups = groups
        self.dilation = dilation

        # Convolutional transforms
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernelSize,
                               padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=4*out_channels, groups=1, kernel_size=1,
                               padding=0, dilation=1)

        self.conv3 = nn.Conv1d(in_channels=4*out_channels, out_channels=4*out_channels, groups=4*out_channels, kernel_size=kernelSize,
                               padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) / 2, dilation=dilation)
        self.conv4 = nn.Conv1d(in_channels=4*out_channels, out_channels=out_channels, groups=1, kernel_size=1,
                               padding=0, dilation=1)

        self.conv1 = nn.utils.weight_norm(self.conv1, 'weight')
        self.conv2 = nn.utils.weight_norm(self.conv2, 'weight')
        self.conv3 = nn.utils.weight_norm(self.conv3, 'weight')
        self.conv4 = nn.utils.weight_norm(self.conv4, 'weight')

        self.norm1 = Normalizer(numChannels=4 * out_channels, channelNorm=channelNorm)
        self.norm2 = Normalizer(numChannels=out_channels, channelNorm=channelNorm)

    def forward(self, x):
        # Apply first convolution block
        y = self.conv2(self.conv1(x))
        y = self.norm1(y)
        y = Functional.selu(y)

        # Apply second convolution block
        y = self.conv4(self.conv3(y))
        y = self.norm2(y)
        y = Functional.selu(y)

        # Return densely connected feature map
        return torch.cat((y, x), dim=1)

########################################################################################################################
# Define the Sleep model

class SkipLSTM(nn.Module):
    def __init__(self, in_channels, out_channels=4, hiddenSize=32):
        super(SkipLSTM, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Bidirectional LSTM to apply temporally across input channels
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=hiddenSize, num_layers=1, batch_first=True, dropout=0.0,
                           bidirectional=True).cuda()
        self.rnn = nn.utils.weight_norm(self.rnn, name='weight_ih_l0')
        self.rnn = nn.utils.weight_norm(self.rnn, name='weight_hh_l0')

        # Output convolution to map the LSTM hidden states from forward and backward pass to the output shape
        self.outputConv1 = nn.Conv1d(in_channels=hiddenSize*2, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)
        self.outputConv1 = nn.utils.weight_norm(self.outputConv1, name='weight')

        self.outputConv2 = nn.Conv1d(in_channels=hiddenSize, out_channels=out_channels, groups=1, kernel_size=1, padding=0)
        self.outputConv2 = nn.utils.weight_norm(self.outputConv2, name='weight')

        # Residual mapping
        self.identMap1 = nn.Conv1d(in_channels=in_channels, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y, z = self.rnn(y)
        z = None
        y = y.permute(0, 2, 1)
        y = Functional.tanh((self.outputConv1(y) + self.identMap1(x)) / 1.41421)
        y = self.outputConv2(y)

        return y

def marginalize(x):
    p_joint = Functional.log_softmax(x, dim=1)

    # Compute marginal for arousal predictions
    p_arousal = p_joint[::, 3, ::]
    x1 = torch.cat((torch.log(1 - torch.exp(p_arousal.unsqueeze(1))), p_arousal.unsqueeze(1)), dim=1)

    # Compute marginal for apnea predictions
    p_apnea = p_joint[::, 1, ::]
    x2 = torch.cat((torch.log(1 - torch.exp(p_apnea.unsqueeze(1))), p_apnea.unsqueeze(1)), dim=1)

    # Compute marginal for sleep/wake predictions
    p_wake = p_joint[::, 0, ::]
    x3 = torch.cat((p_wake.unsqueeze(1), torch.log(1 - torch.exp(p_wake.unsqueeze(1)))), dim=1)
    
    return x1, x2, x3

class Sleep_model_MultiTarget(nn.Module):
    def __init__(self, numSignals=12):
        super(Sleep_model_MultiTarget, self).__init__()
        self.channelMultiplier = 2
        self.kernelSize = 25
        self.numSignals = numSignals

        # Set up downsampling densenet blocks
        self.dsMod1 = SeperableDenseNetUnit(in_channels=self.numSignals, out_channels=self.channelMultiplier*self.numSignals,
                                kernelSize=(2*self.kernelSize)+1, groups=1, dilation=1, channelNorm=False)
        self.dsMod2 = SeperableDenseNetUnit(in_channels=(self.channelMultiplier+1)*self.numSignals, out_channels=self.channelMultiplier*self.numSignals,
                                 kernelSize=(2*self.kernelSize)+1, groups=1, dilation=1, channelNorm=False)
        self.dsMod3 = SeperableDenseNetUnit(in_channels=((2*self.channelMultiplier)+1)*self.numSignals, out_channels=self.channelMultiplier*self.numSignals,
                                 kernelSize=(2*self.kernelSize)+1, groups=1, dilation=1, channelNorm=False)

        # Set up densenet modules
        self.denseMod1 = SeperableDenseNetUnit(in_channels=((3 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=1, channelNorm=True)
        self.denseMod2 = SeperableDenseNetUnit(in_channels=((4 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=2, channelNorm=True)
        self.denseMod3 = SeperableDenseNetUnit(in_channels=((5 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=4, channelNorm=True)
        self.denseMod4 = SeperableDenseNetUnit(in_channels=((6 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=8, channelNorm=True)
        self.denseMod5 = SeperableDenseNetUnit(in_channels=((7 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=16, channelNorm=True)
        self.denseMod6 = SeperableDenseNetUnit(in_channels=((8 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=32, channelNorm=True)
        self.denseMod7 = SeperableDenseNetUnit(in_channels=((9 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                 kernelSize=self.kernelSize, groups=1, dilation=16, channelNorm=True)
        self.denseMod8 = SeperableDenseNetUnit(in_channels=((10 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                  kernelSize=self.kernelSize, groups=1, dilation=8, channelNorm=True)
        self.denseMod9 = SeperableDenseNetUnit(in_channels=((11 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                  kernelSize=self.kernelSize, groups=1, dilation=4, channelNorm=True)
        self.denseMod10 = SeperableDenseNetUnit(in_channels=((12 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                  kernelSize=self.kernelSize, groups=1, dilation=2, channelNorm=True)
        self.denseMod11 = SeperableDenseNetUnit(in_channels=((13 * self.channelMultiplier) + 1) * self.numSignals, out_channels=self.channelMultiplier * self.numSignals,
                                  kernelSize=self.kernelSize, groups=1, dilation=1, channelNorm=True)

        self.skipLSTM = SkipLSTM(((14*self.channelMultiplier)+1)*self.numSignals, hiddenSize=self.channelMultiplier*64, out_channels=4)

    def forward(self, x):
        x = x.detach().contiguous()

        # Downsampling to 1 entity per second
        x = self.dsMod1(x)
        x = Functional.max_pool1d(x, kernel_size=2)
        x = self.dsMod2(x)
        x = Functional.max_pool1d(x, kernel_size=5)
        x = self.dsMod3(x)
        x = Functional.max_pool1d(x, kernel_size=5)
        
        # Dilated Densenet
        x = self.denseMod1(x)
        x = self.denseMod2(x)
        x = self.denseMod3(x)
        x = self.denseMod4(x)
        x = self.denseMod5(x)
        x = self.denseMod6(x)
        x = self.denseMod7(x)
        x = self.denseMod8(x)
        x = self.denseMod9(x)
        x = self.denseMod10(x)
        x = self.denseMod11(x)
        
        # Bidirectional skip LSTM and convert joint predictions to marginal predictions
        x = self.skipLSTM(x)
        x1, x2, x3 = marginalize(x)

        if not(self.training):
            x1 = torch.exp(x1)
            x2 = torch.exp(x2)
            x3 = torch.exp(x3)

        return (x1, x2, x3)












