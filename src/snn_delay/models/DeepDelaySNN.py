import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional



class DeepDelaySNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, max_delay):
        super(DeepDelaySNN, self).__init__()

        surrogate_function = surrogate.ATan()

        self.layer1 = DelayLayerLogic(in_features, hidden_features, max_delay)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function)
        self.dropout = nn.Dropout(p=0.25)


        self.layer2 = DelayLayerLogic(hidden_features, out_features, max_delay)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_function)

    def forward(self, x):

        out1 = self.layer1(x)
        spike1 = self.lif1(out1)
        spike1_dropped = self.dropout(spike1)

        out2 = self.layer2(spike1_dropped)
        spike2 = self.lif2(out2)

        return spike2
