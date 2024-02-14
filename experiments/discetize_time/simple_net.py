
import cupy as cp

# Dataset paths
import numpy as np

from bats import AbstractOptimizer, AbstractLoss, AbstractLayer
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import TTFSSoftmaxCrossEntropy
from bats.Network import Network
from bats.Optimizers import GradientDescentOptimizer

# Dataset
SPIKE_TIMES = np.array([[[0.0],
                         [0.0],
                         [1.0],
                         [2.0]],
                       ])
N_SPIKE_TIMES = np.array([[1, 1, 1, 1],
                         ])
LABELS = np.array([0])
LABELS_GPU = cp.array(LABELS, dtype=cp.int32)

# Model parameters
N_INPUTS = 4
SIMULATION_TIME = 10.0

# Output_layer
N_OUTPUTS = 2
TAU_S_OUTPUT = 1.0
THRESHOLD_HAT_OUTPUT = 1.0
DELTA_THRESHOLD_OUTPUT = THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 1

# Training parameters
N_TRAINING_EPOCHS = 10
TAU_LOSS = 0.1
LEARNING_RATE = 1e-1  # np.full((3,), 1e-2)


def get_predictions(output_spikes: cp.ndarray) -> cp.ndarray:
    return cp.argmin(output_spikes[:, :, 0], axis=1)


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.sum(predictions == labels) / len(labels)


def train(network: Network, output_layer: AbstractLayer, loss_fct: AbstractLoss, optimizer: AbstractOptimizer,
          weight_mean, weight_std):
    weights = np.random.normal(loc=weight_mean, scale=weight_std, size=(N_OUTPUTS, N_INPUTS))
    print(weights)
    output_layer.weights = weights
    best_accuracy = 0.0
    for epoch in range(N_TRAINING_EPOCHS):
        # Training inference
        network.reset()
        network.forward(SPIKE_TIMES, N_SPIKE_TIMES, max_simulation=SIMULATION_TIME)
        out_spikes, n_out_spikes = network.output_spike_trains

        # Metrics
        pred = get_predictions(out_spikes)
        loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, LABELS_GPU)

        #gradient = network.backward(errors, cp.array(LABELS))
        gradient = network.backward(errors)

        avg_gradient = [None if g is None else cp.mean(g, axis=0) for g in gradient]
        deltas = optimizer.step(avg_gradient)
        network.apply_deltas(deltas)

        acc = accuracy(pred.get(), LABELS) * 100
        print(loss)
        if acc > best_accuracy:
            best_accuracy = acc
    return best_accuracy


if __name__ == "__main__":
    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    output_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = TTFSSoftmaxCrossEntropy(TAU_LOSS)
    optimizer = GradientDescentOptimizer(LEARNING_RATE)

    acc = train(network, output_layer, loss_fct, optimizer, 1.0, 0.0)
    print(acc)
