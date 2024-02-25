import cupy as cp
from math import exp
import numpy as np

from bats.Layers import InputLayer, LIFLayer
from bats.Losses import SpikeTimeLoss
from bats.Network import Network
from bats.Optimizers import GradientDescentOptimizer


if __name__ == "__main__":
    network = Network()
    input_layer = InputLayer(n_neurons=1, name="Input layer")
    network.add_layer(input_layer, input=True)

    threshold_hat_output = 1.0
    delta_threshold_output = threshold_hat_output
    output_layer = LIFLayer(previous_layer=input_layer, n_neurons=1, tau_s=1.,
                            theta=threshold_hat_output,
                            delta_theta=delta_threshold_output,
                            max_n_spike=1,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeTimeLoss()

    weight = 4.
    input_spike_time = 0.
    SPIKE_TIMES = np.array([[[input_spike_time]]])
    N_SPIKE_TIMES = np.array([[1]])
    LABELS_GPU = cp.array([0], dtype=cp.int32)

    output_layer.weights = np.array([[weight]])
    network.forward(SPIKE_TIMES, N_SPIKE_TIMES, max_simulation=10.)
    out_spikes, n_out_spikes = network.output_spike_trains
    _loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, LABELS_GPU)
    spiketime = out_spikes[0, 0, 0]
    gradient = network.backward(errors)[1][0, 0, 0]

    s_m = exp(-spiketime / 2)
    s_s = exp(-spiketime / 1)
    dtdw = - (s_m - s_s) / (weight * (-s_m / 2 + s_s))
    gradient_gt = spiketime * dtdw
    #print(f"gradient_gt={gradient_gt} ?= {gradient}=gradient")
    assert(abs(gradient_gt - gradient) < 1e-6)
