
import cupy as cp
import numpy as np

from bats.Layers import InputLayer, LIFLayer
from bats.Losses import SpikeTimeLoss
from bats.Network import Network
from bats.Optimizers import GradientDescentOptimizer

# Dataset
SPIKE_TIMES = np.array([[[0]]])
N_SPIKE_TIMES = np.array([[1]])
LABELS = np.array([0])
LABELS_GPU = cp.array(LABELS, dtype=cp.int32)

# Model parameters
N_INPUTS = 1
SIMULATION_TIME = 10.0

# Output_layer
N_OUTPUTS = 1
TAU_S_OUTPUT = 1.0
THRESHOLD_HAT_OUTPUT = 1.0
DELTA_THRESHOLD_OUTPUT = THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 1

if __name__ == "__main__":
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    output_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeTimeLoss()

    # one step
    output_layer.weights = np.array([[4.]])
    print(f"weight    = {output_layer.weights[0, 0]}")
    network.forward(SPIKE_TIMES, N_SPIKE_TIMES, max_simulation=SIMULATION_TIME)
    out_spikes, n_out_spikes = network.output_spike_trains
    print(f"spiketime = {out_spikes[0, 0, 0]}")
    loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, LABELS_GPU)
    print(f"loss      = {loss[0]}")
    gradient = network.backward(errors)
    print(f"gradient  = {gradient[1][0, 0, 0]}")

    #s_m = exp(-spiketime / 2)
    #s_s = exp(- spiketime / 1)
    #dtdw = - (s_m - s_s) / (weight * (-s_m / 2 + s_s))
    #dldt * dtdw * .1
    ##loss = spiketime ^ 2 / 2
    #dldt = spiketime
    #== gradient

