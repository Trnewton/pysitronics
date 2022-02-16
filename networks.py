from abc import ABC, abstractmethod
# from typing import Tuple
# from enum import Enum, auto

import numpy as np


#### Functions ####

def sigma_gen(N, g, p, seed=None) -> np.ndarray:
    ''''''
    if seed is not None:
        np.random.seed(seed)

    return g * (np.random.randn(N, N)) * (np.random.rand(N, N) < p)/(np.sqrt(N)*p)

def row_balance(matrix) -> None:
    ''''''
    for n, row in enumerate(matrix):
        matrix[n, row.nonzero()] -= np.sum(row) / matrix.shape[0]
    return

def omega_gen(N, q=2.0, shift=-0.5, seed=None, dim=1) -> np.ndarray:
    ''''''
    if seed is not None:
        np.random.seed(seed)

    if dim == 1:
        omega = q * (np.random.rand(N) + shift)
    else:
        omega = q * (np.random.rand(N, dim) + shift)

    return omega


#### Classes ####
#TODO: Create state methods

class Abstract_Network(ABC):
    '''Abstract class for all networks in framework'''

    @abstractmethod
    def step(self, dt: float):
        '''Intergrates network one time step then returns the output and neural activation of the network.'''

    @abstractmethod
    def simulate(self, dt: float, N: int) -> np.ndarray:
        '''Simulates network for N steps with time steps of size dt, returning the output.'''

    @abstractmethod
    def state(self) -> dict():
        '''Returns enough information about the state of the network to reproduce its dynamics.'''

    @abstractmethod
    def get_decoder(self) -> np.ndarray:
        '''Gets value of network decoder.'''

    @abstractmethod
    def set_decoder(self, phi) -> None:
        '''Updates the decocder.'''

    @abstractmethod
    def to_string(self) -> str:
        '''Expresses type and params of network as a string for recording.'''

class Rate(Abstract_Network):
    def __init__(self, N: int , sigma: np.ndarray, omega: np.ndarray, dim: int=1,
                z: np.ndarray=None, phi: np.ndarray=None) -> None:
        '''
          Initilizes the single-layer neural network

          Parameters
          ----------
          N : int
              The dimension of the network
          sigma : numpy 2-d array
              Network matrix
          omega : numpy 1-d array
              Feedback weight matrix
        '''

        # Network structure variables
        self.N = N
        self.dim = dim
        self.sigma = sigma
        self.omega = omega

        # Network state variables
        self.X = 0.5*np.ones(N)
        self.R = np.tanh(self.X)
        if z is not None:
            self.z = z
        else:
            if self.dim == 1:
                self.z = 0
            else:
                self.z = np.zeros(self.dim)
        if phi is not None:
            self.phi = phi
        else:
            if self.dim == 1:
                self.phi = np.zeros(self.N)
            else:
                self.phi = np.zeros((self.N, self.dim))

    def step(self, dt: float):
        '''
            Takes a step of size dt for the recursive network.
            ----------
            dt : float
                Size of integration time step

            Returns
            -------
            R : float
                Activation of network output
            Z : float
                Readout of network
        '''

        self.X = (1.0-dt) * self.X + dt * np.dot(self.sigma, self.R) + dt * \
            np.dot(self.omega, self.z)
        self.R = np.tanh(self.X)
        self.z = np.dot(self.R, self.phi)

        return self.z, self.R

    def simulate(self, dt: float, N: int) -> np.ndarray:
        ''''''

        z_out = np.zeros(N) if self.dim==1 else np.zeros((N, self.dim))

        for n in range(N):
            self.X = (1.0-dt) * self.X + dt * np.dot(self.sigma, self.R) + dt * \
            np.dot(self.omega, self.z)
            self.R = np.tanh(self.X)
            self.z = np.dot(self.R, self.phi)

            z_out[n] = self.z

        return z_out

    def state(self) -> dict():
        return super().state()

    def get_decoder(self) -> np.ndarray:
        return self.phi

    def set_decoder(self, phi) -> None:
        self.phi = phi

    def to_string(self) -> str:
        return super().to_string()

class LIF(Abstract_Network):
    def __init__(self, N:int , sigma:np.ndarray, omega:np.ndarray, t_m:float,\
                t_ref:float, v_reset:float, v_peak:float, i_bias:float,\
                t_r:float, t_d:float, dim:int=1) -> None:
        '''
        '''

        #### LIF Variables
        self.t_m = t_m
        self.t_ref = t_ref
        self.v_reset = v_reset
        self.v_peak = v_peak
        self.i_bias = i_bias # bias current
        self.t_r = t_r
        self.t_d = t_d
        self.dim = dim

        #### Topological Variables
        self.N = N
        self.sigma = sigma # resevoir connections
        self.omega = omega # feedback connections
        if dim == 1:
            self.phi = np.zeros(N) # decoder
        else:
            self.phi = np.zeros((self.dim, self.N)) # decoder

        #### State Variables
        self.refr_timer = np.zeros(N) # Timer for spike refractor periods
        self.v = self.v_reset + np.random.rand(N) * (30 - self.v_reset) # neuron voltage

        self.i_ps = np.zeros(N) # post synaptic current
        self.h = np.zeros(N) # current filter        self.i = np.zeros(N) # neuronal current

        self.h_rate = np.zeros(N) # spike rate filter
        self.R = np.zeros(N) # firing rates

        self.z = np.matmul(self.phi, self.R) # readout

    def step(self, dt: float):
        '''
            Takes a step of size dt for the recursive network.
            ----------
            dt : float
                Size of integration time step

            Returns
            -------
            R : float
                Activation of network output
            Z : float
                Readout of network
        '''

        if self.dim == 1:
            self.i = self.i_ps + self.omega * self.z + self.i_bias
        else:
            self.i = self.i_ps + np.matmul(self.omega, self.z) + self.i_bias

        # Compute neuronal voltages with refractory period
        self.refr_timer = np.maximum(0, self.refr_timer-dt)
        self.v += dt * ((self.refr_timer<=0) * (self.i - self.v) / self.t_m)

        # Get spiking neurons
        self.spikes = self.v >= self.v_peak
        self.spike_idx = np.argwhere(self.spikes)
        self.spike_current = np.matmul(self.sigma, self.spikes)

        # Set refractory timer
        self.refr_timer[self.spike_idx] = self.t_ref

        # Compute exponential filter
        if self.t_r == 0: # Single exponential filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_d) + self.spike_current * np.any(self.spikes) / self.t_d
            self.R = self.R * np.exp(-dt/self.t_d) + self.spikes / self.t_d
        else: # Double exponential filter
            # Current filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_r) + dt * self.h
            self.h = self.h * np.exp(-dt/self.t_d) + self.spike_current * np.any(self.spikes) / (self.t_d * self.t_r)

            # Spike rate filter
            self.R = self.R * np.exp(-dt/self.t_r) + dt * self.h_rate
            self.h_rate = self.h_rate * np.exp(-dt/self.t_d) + self.spikes / (self.t_r * self.t_d)

        ## interpolant implementation
        self.v[self.spike_idx] += (30 - self.v[self.spike_idx])
        self.v[self.v >= self.v_peak] += (self.v_reset - self.v[self.v >= self.v_peak])

        self.z = np.matmul(self.phi, self.R)

        return self.z, self.R

    def simulate(self, dt: float, N: int) -> np.ndarray:
        ''''''

        z_out = np.zeros(N) if self.dim==1 else np.zeros((N, self.dim))

        e_t_r = np.exp(-dt/self.t_r) if self.t_r != 0 else 0
        e_t_d = np.exp(-dt/self.t_d)

        for n in range(N):
            if self.dim == 1:
                self.i = self.i_ps + self.omega * self.z + self.i_bias
            else:
                self.i = self.i_ps + np.matmul(self.omega, self.z) + self.i_bias

            # Compute neuronal voltages with refractory period
            self.refr_timer = np.maximum(0, self.refr_timer-dt)
            self.v += dt * ((self.refr_timer<=0) * (self.i - self.v) / self.t_m)

            # Get spiking neurons
            self.spikes = self.v >= self.v_peak
            self.spike_idx = np.argwhere(self.spikes)
            self.spike_current = np.matmul(self.sigma, self.spikes)

            # Set refractory timer
            self.refr_timer[self.spike_idx] = self.t_ref

            # Compute exponential filter
            if self.t_r == 0: # Single exponential filter
                self.i_ps = self.i_ps * e_t_d + self.spike_current * np.any(self.spikes) / self.t_d
                self.R = self.R * e_t_d + self.spikes / self.t_d
            else: # Double exponential filter
                # Current filter
                self.i_ps = self.i_ps * e_t_r + dt * self.h
                self.h = self.h * e_t_d + self.spike_current * np.any(self.spikes) / (self.t_d * self.t_r)

                # Spike rate filter
                self.R = self.R * e_t_r + dt * self.h_rate
                self.h_rate = self.h_rate * e_t_d + self.spikes / (self.t_r * self.t_d)

            ## interpolant implementation
            self.v[self.spike_idx] += (30 - self.v[self.spike_idx])
            self.v[self.v >= self.v_peak] += (self.v_reset - self.v[self.v >= self.v_peak])

            self.z = np.matmul(self.phi, self.R)
            z_out[n] = self.z

        return z_out

    def state(self) -> dict():
        state = {}
        state['v'] = self.v
        state['refr_timer'] = self.refr_timer
        state['i_ps'] = self.i_ps
        state['h'] = self.h
        state['h_rate'] = self.h_rate
        state['R'] = self.R
        state['z'] = self.z

        return state

    def get_decoder(self) -> np.ndarray:
        return self.phi

    def set_decoder(self, phi) -> None:
        self.phi = phi

    def to_string(self) -> str:
        return super().to_string()

class LIF_Rate(Abstract_Network):
    def __init__(self, N:int, sigma:np.ndarray, omega:np.ndarray, t_m:float,
                t_ref:float, v_reset:float, v_peak:float, i_bias:float,\
                t_r:float, t_d:float, dim=1):
        ''''''

        #### LIF Variables
        self.t_m = t_m
        self.t_ref = t_ref
        self.v_reset = v_reset
        self.v_peak = v_peak
        self.i_bias = i_bias # bias current
        self.t_r = t_r
        self.t_d = t_d
        self.dim = dim

        #### Topological Variables
        self.N = N
        self.sigma = sigma # resevoir connections
        self.omega = omega # feedback connections
        if dim == 1:
            self.phi = np.zeros(N) # decoder
        else:
            self.phi = np.zeros((self.dim, self.N)) # decoder

        #### State Variables
        self.i_ps = self.i_bias*(np.random.rand(N)-0.5) # post synaptic current
        self.h = np.zeros(N) # current filter

        self.h_rate = np.zeros(N) # spike rate filter
        self.R = np.zeros(N) # firing rates

        self.z = np.matmul(self.phi, self.R) # readout

    def step(self, dt:float):
        ''''''

        if self.dim == 1:
            self.i = self.i_ps + self.omega * self.z + self.i_bias
        else:
            self.i = self.i_ps + np.matmul(self.omega, self.z) + self.i_bias

        self.rate = np.nan_to_num(1/(self.t_r - self.t_m * (np.log(self.i - self.v_peak) - np.log(self.i - self.v_reset))), copy=False)
        self.spike_current = dt * np.matmul(self.sigma, self.rate)

        # Compute exponential filter
        if self.t_r == 0: # Single exponential filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_d) + self.spike_current / self.t_d
            self.R = self.R * np.exp(-dt/self.t_d) + dt * self.rate / self.t_d
        else: # Double exponential filter
            # Current filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_r) + dt * self.h
            self.h = self.h * np.exp(-dt/self.t_d) + self.spike_current / (self.t_d * self.t_r)

            # Spike rate filter
            self.R = self.R * np.exp(-dt/self.t_r) + dt * self.h_rate
            self.h_rate = self.h_rate * np.exp(-dt/self.t_d) + dt * self.rate / (self.t_r * self.t_d)

        self.z = np.matmul(self.phi, self.R)

        return self.z, self.R

    def simulate(self, dt: float, N: int) -> np.ndarray:
        ''''''

        z_out = np.zeros(N) if self.dim==1 else np.zeros((N, self.dim))

        e_t_r = np.exp(-dt/self.t_r) if self.t_r != 0 else 0
        e_t_d = np.exp(-dt/self.t_d)

        for n in range(N):
            if self.dim == 1:
                self.i = self.i_ps + self.omega * self.z + self.i_bias
            else:
                self.i = self.i_ps + np.matmul(self.omega, self.z) + self.i_bias

            self.rate = np.nan_to_num(1/(self.t_r - self.t_m * (np.log(self.i - self.v_peak) - np.log(self.i - self.v_reset))), copy=False)
            self.spike_current = dt * np.matmul(self.sigma, self.rate)

            # Compute exponential filter
            if self.t_r == 0: # Single exponential filter
                self.i_ps = self.i_ps * e_t_d + self.spike_current / self.t_d
                self.R = self.R * e_t_d + dt * self.rate / self.t_d
            else: # Double exponential filter
                # Current filter
                self.i_ps = self.i_ps * e_t_r + dt * self.h
                self.h = self.h * e_t_d + self.spike_current / (self.t_d * self.t_r)

                # Spike rate filter
                self.R = self.R * e_t_r + dt * self.h_rate
                self.h_rate = self.h_rate * e_t_d + dt * self.rate / (self.t_r * self.t_d)

            self.z = np.matmul(self.phi, self.R)
            z_out[n] = self.z

        return z_out

    def state(self) -> dict():
        state = {}
        state['i_ps'] = self.i_ps
        state['h'] = self.h
        state['h_rate'] = self.h_rate
        state['R'] = self.R
        state['z'] = self.z

        return state

    def get_decoder(self) -> np.ndarray:
        return self.phi

    def set_decoder(self, phi) -> None:
        self.phi = phi

    def to_string(self) -> str:
        return super().to_string()


#### Factory ####

class Network_Factory:
    '''Factory for creating possible network types.'''
    def __init__(self, network_type:str, N:int, p:float, net_params:dict, row_balance:bool) -> None:
        if network_type == 'Rate':
            self.network_class = Rate
        elif network_type == 'LIF':
            self.network_class = LIF
        elif network_type == 'LIF_Rate':
            self.network_class = LIF_Rate

        self.N = N
        self.p = p
        self.net_params = net_params
        self.row_balance = row_balance

    def create_network(self, q:float, g:float, omega_seed:int=None, sigma_seed:int=None) -> Abstract_Network:
        '''Returns created network.'''
        omega = omega_gen(self.N, q, seed=omega_seed, dim=self.net_params['dim'])

        sigma = sigma_gen(self.N, g, p=self.p, seed=sigma_seed)

        if self.row_balance:
            row_balance(sigma)

        return self.network_class(self.N, sigma, omega, **self.net_params)

    def to_string(self) -> str:
        '''returns string representation of network factory for saving to file.'''
        return f'Network:{self.network_class.__name__}\n'\
                + f'\tN:{self.N}\n'\
                + f'\tp:{self.p}\n'\
                + ''.join([f'\t{param}:{val}\n' for param, val in self.net_params.items()])