import torch
import torch.nn as nn
import numpy as np
from sns_toolbox.neurons import SpikingNeuron, NonSpikingNeuron
from sns_toolbox.networks import Network
from sns_toolbox.connections import SpikingConnection
from sns_toolbox.compilers import torch


class ThreeFactorModel(nn.Module):
    def __init__(self, num_neurons, dt, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons # per population 
        self.dt = dt
        self.device = device

        # build network
        self.net = Network(name='Three Factor Correlative Model')

        self.min_stimulus = 0.0
        self.max_stimulus = 100.0

        # create centers of each neuron's Gaussian curve
        threshold_range = np.linspace(self.min_stimulus, self.max_stimulus, self.num_neurons)

        # joint angle population
        for i, threshold in enumerate(threshold_range):
            joint_neuron_name = f"SpikingAngle_{i}"
            JointinputTemplate = SpikingNeuron(threshold_time_constant=1.0,
                                            threshold_proportionality_constant=1.0,
                                            threshold_initial_value=threshold,
                                            reset_potential=0.0)
            self.net.add_neuron(JointinputTemplate, joint_neuron_name)
            self.net.add_input(dest=joint_neuron_name, name=f"JointSpikingInput_{i}")
            self.net.add_output(source=joint_neuron_name)

        # concatenated torques
        for j, thresh in enumerate(threshold_range):
            torque_neuron_name = f"SpikingTau_{j}"
            TauinputTemplate = SpikingNeuron(threshold_time_constant=1.0,
                                            threshold_proportionality_constant=1.0,
                                            threshold_initial_value=thresh[j],
                                            reset_potential=0.0)
            self.net.add_neuron(TauinputTemplate,torque_neuron_name)
            self.net.add_input(dest=torque_neuron_name, name=f"TauSpikingInput_{j}")
            self.net.add_output(source=torque_neuron_name)

        self.conductance_param = nn.Parameter(torch.ones(num_neurons, num_neurons)) # parameter to be changed in the factor fuction

        # add synapse connections between inputs and NSI layers
        self.starting_synapses = SpikingConnection(max_conductance=self.conductance_param, reversal_potential=0.0, time_constant=2.0) 
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                NSI_neuron_name = f"NSI_{i}{j}"
                NSI_template = NonSpikingNeuron()
                self.net.add_neuron(NSI_template, NSI_neuron_name)
                self.net.add_connection(connection_type=self.starting_synapses,source=joint_neuron_name, destination=NSI_neuron_name, name=f"AngleSyn{i}{j}")
                self.net.add_connection(connection_type=self.starting_synapses,source=torque_neuron_name, destination=NSI_neuron_name, name=f"TauSyn{i}{j}")
                self.net.add_output (source=NSI_neuron_name, name='Event')

        self.model = self.net.compile(backend='torch', dt=self.dt, debug=False)
        for i in range(num_neurons):
            self.start_angle_idx = self.net.get_neuron_index("SpikingAngle_{0}")
            self.start_torquw_idx = self.net.get_neuron_index("SpikingAngle_{0}")

    def forward(self, torques, angles):
        self.model.g_spike = self.g_spike # track conductances of spikes
        combined_inputs = torch.cat([angles, torques], dim=-1)
        outputs = self.model(combined_inputs) 
        


