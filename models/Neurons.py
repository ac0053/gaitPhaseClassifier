import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from typing import List
import itertools
import scipy as sp

class SpikingInput(nn.Module):

    def __init__(self,
                 num_neurons_per_joint: int,
                 parameters: dict = None,
                 device: torch.device = torch.device('cpu')):

        super().__init__()

        self.device = device

        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)

        c_m = parameters.get('c_m', 1.0) # membrane capacitance
        e_rest = parameters.get('e_rest', 0.0) # resting voltage 
        g_m_leak = parameters.get('g_m_leak', 1.0) # membrane leak conductance
        e_syn = parameters.get('e_syn', 20.0) # synaptic voltage
        i_bias = parameters.get('i_bias', 0.0) # bias current
        e_reset = parameters.get('e_reset', 0.0) # reset voltage
        theta_0 = parameters.get('theta_0', 1.0) # initial threshold voltage

        self.num_neurons = num_neurons_per_joint

        # Neuron parameters
        self.c_m = torch.full((self.num_neurons, 1), c_m, device=self.device)
        self.e_rest = torch.full((self.num_neurons, 1), e_rest, device=self.device)
        self.g_m_leak = torch.full((self.num_neurons, 1), g_m_leak, device=self.device)
        self.e_syn_delta = torch.full((self.num_neurons, 1), e_rest + e_syn, device=self.device)
        self.i_bias = torch.full((self.num_neurons, 1), i_bias, device=self.device)
        self.e_reset = torch.full((self.num_neurons, 1), e_reset, device=self.device)
        self.v_m = self.e_rest.clone().squeeze(1)
        self.theta_0 = torch.full((self.num_neurons, 1), theta_0, device=self.device).squeeze(1)

    def _dv_dt(self, v_post, i_ext):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        return (self.i_bias + i_ext - i_leak) / self.c_m

    def _rk2(self, i_ext_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_ext_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_ext_values).squeeze(1)

        self.v_m += self.dt * k2

        if self.v_m >= self.theta_0:
            self.v_m = self.e_reset.clone()
        else:
            self.v_m = self.v_m.clone()

        return self.v_m

    def forward(self, i_ext_vector: torch.Tensor) -> torch.Tensor:

        num_steps = i_ext_vector.shape[1]

        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        for t in range(num_steps):
            i_ext_t = i_ext_vector[:, t].unsqueeze(1)

            self.v_m = self._rk2(i_ext_values=i_ext_t)
            if self.v_m >= self.theta_0:
                self.v_m = self.e_reset.clone()
            else:
                self.v_m = self.v_m.clone()
            v_m_Post[:, t] = self.v_m

        return v_m_Post

class NonSpikingInterneuron(nn.Module):

    def __init__(self,
                 num_neurons_per_joint: int,
                 num_inputs: int = 1,
                 parameters: dict = None,
                 connection_type: str = 'row_col',
                 device: torch.device = torch.device('cpu'),
                 k_syn_init: torch.Tensor = None,
                 ode_method: str = 'rk2'):

        super().__init__()

        self.device = device
        self.connection_type = connection_type
        self.ode_method = ode_method

        # Default parameters
        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)
        c_m = parameters.get('c_m', 1.0)
        e_rest = parameters.get('e_rest', 0.0)
        g_m_leak = parameters.get('g_m_leak', 1.0)
        e_syn = parameters.get('e_syn', 20.0)
        R_op_range = parameters.get('R_op_range', 1.0)
        i_bias = parameters.get('i_bias', 0.0)

        self.num_inputs = num_inputs  # number of joints
        self.N = num_neurons_per_joint  # N neurons per joint

        self.num_neurons = self.N * self.N  # Total  NSI neurons in the N x N grid

        # Neuron parameters
        self.c_m = torch.full((self.num_neurons, 1), c_m, device=self.device)
        self.e_rest = torch.full((self.num_neurons, 1), e_rest, device=self.device)
        self.g_m_leak = torch.full((self.num_neurons, 1), g_m_leak, device=self.device)
        self.e_syn_delta = torch.full((self.num_neurons, 1), e_rest + e_syn, device=self.device)
        self.R_op_range = torch.full((self.num_neurons, 1), R_op_range, device=self.device)
        self.i_bias = torch.full((self.num_neurons, 1), i_bias, device=self.device)
        self.v_m = self.e_rest.clone().squeeze(1)

        expected_shape = (self.num_inputs, self.N)

        if k_syn_init is None:
            # initialized default k_syn
            initial_k_syn = torch.ones(expected_shape, device=self.device)
        else:
            initial_k_syn = k_syn_init.to(device=self.device)

        self.k_syn = initial_k_syn

    def g_max(self, k_syn_gain_t: torch.Tensor, num_active_inputs: int = 2) -> torch.Tensor:
        g_max_factor_num = k_syn_gain_t * self.R_op_range.T
        g_max_factor_den = self.e_syn_delta.T - g_max_factor_num * num_active_inputs

        if torch.any(g_max_factor_den <= 1e-9):
            raise ValueError("Denominator in g_max is near zero.")
        return g_max_factor_num / g_max_factor_den

    def g_syn(self, u_pre: torch.Tensor, g_max_t: torch.Tensor) -> torch.Tensor:
        activation = torch.clamp(u_pre / self.R_op_range.T, min=0.0, max=1.0)
        return g_max_t * activation

    def _syn_connection(self, u_pre_0: torch.Tensor, u_pre_1: torch.Tensor, v_m_grid: torch.Tensor) -> torch.Tensor:

        u0_grid = u_pre_0.view(self.N, 1).expand(-1, self.N)
        u1_grid = u_pre_1.view(1, self.N).expand(self.N, -1)

        k0_grid = self.k_syn[0, :].view(self.N, 1).expand(-1, self.N)
        k1_grid = self.k_syn[1, :].view(1, self.N).expand(self.N, -1)

        g_max0 = self.g_max(k0_grid.flatten().unsqueeze(0))
        g_max1 = self.g_max(k1_grid.flatten().unsqueeze(0))

        g_syn0 = self.g_syn(u0_grid.flatten().unsqueeze(0), g_max0).view(self.N, self.N)
        g_syn1 = self.g_syn(u1_grid.flatten().unsqueeze(0), g_max1).view(self.N, self.N)

        i_syn_0 = g_syn0 * (self.e_syn_delta.view(self.N, self.N) - v_m_grid)
        i_syn_1 = g_syn1 * (self.e_syn_delta.view(self.N, self.N) - v_m_grid)

        i_row_col_syn_total = (i_syn_0 + i_syn_1).flatten().unsqueeze(1)

        return i_row_col_syn_total

    def _dv_dt(self, v_post, i_syn_values=None):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        return (self.i_bias + i_syn_values - i_leak) / self.c_m

    def _rk2(self, i_syn_total_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_syn_total_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_syn_total_values).squeeze(1)

        self.v_m += self.dt * k2

        return self.v_m

    def forward(self, u_pre_vector: List[torch.Tensor]):
        num_steps = u_pre_vector[0].shape[1]
        # Create an output tensor
        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        u_pre_0, u_pre_1 = u_pre_vector[0], u_pre_vector[1]

        for t in range(num_steps):
            u_pre_0_t, u_pre_1_t = u_pre_0[:, t], u_pre_1[:, t]

            v_m_grid = self.v_m.view(self.N, self.N)

            i_syn_total = self._syn_connection(u_pre_0=u_pre_0_t, u_pre_1=u_pre_1_t,
                                                       v_m_grid=v_m_grid)

            self.v_m = self._rk2(i_syn_total_values=i_syn_total)

            v_m_Post[:, t] = self.v_m

        return v_m_Post.view(self.N, self.N, num_steps)

class NonSpikingOutput(nn.Module):

    def __init__(self,
                 num_neurons: int,
                 num_steps: int,
                 num_outputs: int = 1,
                 parameters: dict = None,
                 device: torch.device = torch.device('cpu'),
                 k_syn_init: torch.Tensor = None,
                 e_syn_init: torch.Tensor = None,
                 ode_method: str = 'rk2'):

        super().__init__()

        self.device = device
        self.ode_method = ode_method
        self.num_outputs = num_outputs
        self.num_steps = num_steps

        # Default parameters
        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)
        c_m = parameters.get('c_m', 1.0)
        e_rest = parameters.get('e_rest', 0.0)
        g_m_leak = parameters.get('g_m_leak', 1.0)
        R_op_range = parameters.get('R_op_range', 1.0)
        i_bias = parameters.get('i_bias', 0.0)

        self.num_neurons = num_neurons  # Total  NSI neurons in the N x N grid

        self.expected_shape = (self.num_neurons, self.num_outputs)  # num_neuronsx1
        # Neuron parameters
        self.c_m = torch.full(self.expected_shape, c_m, device=self.device)
        self.e_rest = torch.full(self.expected_shape, e_rest, device=self.device)
        self.g_m_leak = torch.full(self.expected_shape, g_m_leak, device=self.device)
        self.R_op_range = torch.full(self.expected_shape, R_op_range, device=self.device)
        self.i_bias = torch.full(self.expected_shape, i_bias, device=self.device)

        self.v_m = torch.full(self.expected_shape, 0.0, device=self.device)

        self.e_syn_delta = torch.nn.Parameter(e_syn_init.to(device=self.device))

        self.k_syn = torch.nn.Parameter(k_syn_init.to(device=self.device))

    def _g_max(self, k_syn_gain_t: torch.Tensor, e_syn_delta, num_active_inputs: int = 2) -> torch.Tensor:
        g_max_factor_num = k_syn_gain_t * self.R_op_range
        g_max_factor_den = e_syn_delta - g_max_factor_num * num_active_inputs
        return g_max_factor_num / g_max_factor_den

    def _g_syn(self, u_pre: torch.Tensor, g_max: torch.Tensor) -> torch.Tensor:
        activation = torch.clamp(u_pre / self.R_op_range, min=self.R_op_range[0,0], max=self.R_op_range[0,0]*2.0)
        return g_max * activation

    def _dv_dt(self, v_post, i_syn_values):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        return (self.i_bias + i_syn_values - i_leak) / self.c_m

    def _steady_state(self, g_syn, e_syn):
        return (self.i_bias+g_syn*e_syn+self.e_rest * self.g_m_leak)/(self.g_m_leak+g_syn)

    def _connection(self, u_pre: torch.Tensor,
                    k_syn: torch.Tensor, e_syn: torch.Tensor) -> torch.Tensor:

        g_max = self._g_max(k_syn, e_syn)
        g_syn = self._g_syn(u_pre, g_max)
        i_syn_total = g_syn * (e_syn - self.v_m)

        self.v_m = self._steady_state(g_syn, e_syn)

        return i_syn_total

    def _rk2_forward_pass(self, i_syn_total_values: torch.Tensor):

        k1 = self._dv_dt(self.v_m, i_syn_total_values)
        k2 = self._dv_dt(self.v_m + 0.5 * self.dt * k1, i_syn_total_values)

        self.v_m += self.dt * k2

        return self.v_m

    def forward(self, u_pre_vector: torch.Tensor):

        v_m_Post = torch.zeros(self.num_outputs, self.num_steps, device=self.device)

        for t in range(self.num_steps):

            u_pre_t = u_pre_vector[:, :, t].flatten().unsqueeze(1).expand(self.expected_shape)

            i_syn_total = self._connection(u_pre=u_pre_t, k_syn=self.k_syn, e_syn=self.e_syn_delta)

            self.v_m = self._rk2_forward_pass(i_syn_total_values=i_syn_total)

            v_m_Post[:, t] = self.v_m.sum(dim=0)

        return v_m_Post.view(self.num_outputs, self.num_steps)